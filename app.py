from flask import Flask, request, jsonify
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from datetime import datetime, timedelta
from groq import Groq
import pytz
import os
import json
from typing import Any, Dict, List
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)


def _debug_log(message: str) -> None:
    if os.getenv('DEBUG_LOG', '').lower() in {'1', 'true', 'yes', 'y'}:
        print(f"[voice-agent] {message}", flush=True)


def _is_affirmative(text: str) -> bool:
    if not text:
        return False
    normalized = ''.join(ch.lower() if ch.isalnum() or ch.isspace() else ' ' for ch in text)
    normalized = ' '.join(normalized.split())
    affirmations = {
        'yes', 'yeah', 'yep', 'yup', 'sure', 'ok', 'okay', 'correct', 'confirm',
        'confirmed', 'sounds good', 'that works', 'do it', 'please do', 'book it'
    }
    return normalized in affirmations


def _last_user_message(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            return (msg.get('content') or '').strip()
    return ''


def _extract_event_args_with_groq(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract name/datetime/title from conversation. Returns dict with keys.

    This is a reliability fallback: if the main assistant forgets to emit a tool call
    after the user confirms, we still produce a tool call payload for Vapi.
    """
    extraction_system = (
        "You are a strict information extraction engine. "
        "From the conversation, extract meeting details and output ONLY valid JSON. "
        "Schema: {\"name\": string, \"datetime\": string, \"title\": string}. "
        "datetime must be ISO 8601 like 2026-02-20T14:00:00 in UTC (no timezone suffix needed). "
        "If title is missing, set it to \"Meeting with {name}\". "
        "Do not include extra keys, markdown, or commentary."
    )
    extraction_messages = [
        {"role": "system", "content": extraction_system},
        {"role": "user", "content": json.dumps(messages)},
    ]

    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=extraction_messages,
        temperature=0,
        max_tokens=200,
        stream=False,
    )

    raw = (resp.choices[0].message.content or '').strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Best-effort: try to salvage JSON substring
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(raw[start:end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return {}

# Initialize Groq client once at startup
# Why: Avoids creating a new client on every request
groq_client = Groq(api_key=os.environ['GROQ_API_KEY'])
CALENDAR_ID = os.environ['CALENDAR_ID']


def get_system_prompt():
    """
    Generates system prompt with today's date AND current time injected.
    Why: LLM needs today's date and current time to:
    1. Resolve relative dates like 'tomorrow', 'next Monday' correctly
    2. Reject past dates and past times on today's date
    Generated fresh on every request so it's always accurate.
    """
    today = datetime.now(pytz.UTC)
    date_str = today.strftime('%A, %B %d, %Y')
    time_str = today.strftime('%I:%M %p')

    return f"""You are Tara, a friendly and professional scheduling assistant.
Today's current date is {date_str} and current time is {time_str} UTC.
Use this to:
- Calculate relative dates like "tomorrow", "next Monday", "this Thursday", "next week" etc.
- Reject any date or time that is in the past

Always resolve relative dates to the actual calendar date before confirming.

Follow this conversation flow strictly:

STEP 1: Greet the user warmly.
Example: "Hi! I'm Tara, your scheduling assistant. I'd love to help you book a meeting today!"

STEP 2: Ask for their full name.

STEP 3: Ask for the date they want.
- Accept natural language like "tomorrow", "next Monday", "this Thursday", "March 5th"
- Always resolve to the actual date. Example: if today is {date_str},
  calculate what "next Monday" or "this Thursday" actually falls on
- If ambiguous like "next week" with no day specified, ask which day
- If the date is in the past, say: "It looks like that date has already passed.
  Could you choose a date from today onwards?"

STEP 4: Ask for the time they prefer.
- Accept natural language like "2pm", "3:30 in the afternoon"
- Always clarify AM or PM if ambiguous
- Let the user know the time will be saved in UTC
- Example: "What time works for you? I'll save it in UTC —
  so if you're in Pakistan, 2pm PKT would be 9am UTC"
- If the user picks today's date, make sure the time is in the future.
  Current UTC time is {time_str}. If the time has already passed today, say:
  "That time has already passed today. Could you pick a later time,
  or would you prefer a different date?"

STEP 5: Ask for a meeting title (tell them it's optional).
- If they skip it, default to "Meeting with [their name]"

STEP 6: Read back ALL details clearly including UTC.
Example: "Just to confirm — I'll book 'Project Kickoff'
for John on Thursday February 20th 2026 at 9:00 AM UTC.
Does that sound right?"

STEP 7: Wait for confirmation.
- If YES: Immediately call the createCalendarEvent function
- If NO: Ask what they'd like to change and go back to that step

STEP 8: After function returns success, tell the user their event is booked
and wish them a great day.

IMPORTANT RULES:
- Always convert dates and times to ISO 8601 before calling the function
  Example: 2026-02-20T14:00:00
- Never call the function until the user explicitly confirms with yes
- Never accept a past date or a past time on today's date
- Be warm and conversational, not robotic
- If user provides unnecessary details or goes off topic, acknowledge
  briefly and warmly then steer back to collecting required information
- You are only a scheduling assistant. If asked about anything unrelated,
  politely say you can only help with booking calendar events
- Always say the resolved actual date out loud during confirmation so
  the user can catch any mistakes
"""


def get_calendar_service():
    """
    Creates authenticated Google Calendar service.
    Why: Uses environment variables so secrets are never in source code.
    """
    creds = Credentials(
        token=None,
        refresh_token=os.environ['GOOGLE_REFRESH_TOKEN'],
        client_id=os.environ['GOOGLE_CLIENT_ID'],
        client_secret=os.environ['GOOGLE_CLIENT_SECRET'],
        token_uri='https://oauth2.googleapis.com/token',
        scopes=['https://www.googleapis.com/auth/calendar']
    )
    creds.refresh(Request())
    return build('calendar', 'v3', credentials=creds)


@app.route('/')
def home():
    """Health check — confirms server is running."""
    return "Voice Scheduling Agent is running!"


from flask import Flask, request, jsonify, Response
import json

@app.route('/chat/completions', methods=['POST'])
def chat():
    """
    Custom LLM endpoint that Vapi calls for every conversation turn.

    Key logic:
    - Always call Groq in non-streaming mode first
    - If Groq wants to call a tool: return as complete JSON (non-streaming)
      Why: Tool calls must be complete JSON, not streamed chunks
    - If Groq returns text: stream it word by word
      Why: Streaming gives natural real-time speech in Vapi
    """
    try:
        data = request.json
        messages = data.get('messages', [])
        stream = data.get('stream', False)

        # Remove Vapi's system message - replace with ours
        messages = [m for m in messages if m.get('role') != 'system']

        # Inject dynamic system prompt with today's date and time
        messages = [{'role': 'system', 'content': get_system_prompt()}] + messages

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "createCalendarEvent",
                    "description": "Creates a Google Calendar event. Only call this after user confirms all details.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The user's full name"
                            },
                            "datetime": {
                                "type": "string",
                                "description": "Meeting date and time in ISO 8601 format. Example: 2026-02-20T14:00:00"
                            },
                            "title": {
                                "type": "string",
                                "description": "Meeting title. Default to 'Meeting with [name]' if not provided."
                            }
                        },
                        "required": ["name", "datetime"]
                    }
                }
            }
        ]

        # Always call Groq non-streaming first
        # Why: We need to inspect the response before deciding
        # whether to return a tool call or stream text
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=1000,
            stream=False
        )

        choice = response.choices[0]
        message = choice.message

        # CASE 1: Groq wants to call a tool
        # Return as complete non-streaming JSON
        # Why: Vapi needs complete tool call JSON to trigger /create-event
        if message.tool_calls:
            _debug_log("Groq returned tool_calls normally")
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]

            return jsonify({
                "id": response.id,
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": response.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": message.content or "",
                            "tool_calls": tool_calls,
                            "toolCalls": tool_calls
                        },
                        "finish_reason": "tool_calls"
                    }
                ]
            })

        # Reliability fallback:
        # If the user just confirmed ("yes") but the model returned plain text,
        # force a tool call by extracting arguments from the conversation.
        last_user = _last_user_message(messages)
        if _is_affirmative(last_user):
            _debug_log("User affirmed, but no tool_calls. Running extraction fallback.")
            args = _extract_event_args_with_groq(messages)
            name = (args.get('name') or '').strip()
            datetime_str = (args.get('datetime') or '').strip()
            title = (args.get('title') or '').strip()
            if name and datetime_str:
                if not title:
                    title = f"Meeting with {name}"
                tool_calls = [
                    {
                        "id": "call_manual_1",
                        "type": "function",
                        "function": {
                            "name": "createCalendarEvent",
                            "arguments": json.dumps({
                                "name": name,
                                "datetime": datetime_str,
                                "title": title,
                            })
                        }
                    }
                ]
                return jsonify({
                    "id": response.id,
                    "object": "chat.completion",
                    "created": int(datetime.now().timestamp()),
                    "model": response.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": tool_calls,
                                "toolCalls": tool_calls
                            },
                            "finish_reason": "tool_calls"
                        }
                    ]
                })

            _debug_log(f"Extraction fallback failed: {args}")
            # If we can't extract reliably, ask for the missing fields.
            return jsonify({
                "id": response.id,
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": response.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "I heard your confirmation, but I'm missing a clear name or date/time to book. Could you repeat your full name and the meeting date + time (UTC)?",
                            "tool_calls": None
                        },
                        "finish_reason": "stop"
                    }
                ]
            })

        # CASE 2: Groq returns text - stream it if Vapi asked for streaming
        # Why: Streaming gives natural word-by-word speech output in Vapi
        if stream:
            def generate():
                content = message.content or ""
                words = content.split(' ')

                for i, word in enumerate(words):
                    chunk_content = word + ('' if i == len(words) - 1 else ' ')
                    chunk_data = {
                        "id": response.id,
                        "object": "chat.completion.chunk",
                        "created": int(datetime.now().timestamp()),
                        "model": "llama-3.3-70b-versatile",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": chunk_content
                                },
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"

                # Final chunk signals end of stream
                done_data = {
                    "id": response.id,
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": "llama-3.3-70b-versatile",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(done_data)}\n\n"
                yield "data: [DONE]\n\n"

            return Response(
                generate(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )

        # CASE 3: Non-streaming fallback
        return jsonify({
            "id": response.id,
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": response.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": None
                    },
                    "finish_reason": choice.finish_reason
                }
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/create-event', methods=['POST'])
def create_event():
    """
    Called by Vapi when Groq triggers the createCalendarEvent tool.
    Extracts the arguments, validates the datetime, creates the Google
    Calendar event, and returns a confirmation message Vapi speaks to user.
    """
    tool_call_id = ''
    try:
        data = request.json

        # Vapi wraps tool call data in a specific structure
        tool_calls = data.get('message', {}).get('toolCalls', [])

        if tool_calls:
            args = tool_calls[0].get('function', {}).get('arguments', {})
            tool_call_id = tool_calls[0].get('id', '')
            # Arguments may be a JSON string — parse if needed
            if isinstance(args, str):
                args = json.loads(args)
        else:
            args = data

        name = args.get('name', 'Guest')
        datetime_str = args.get('datetime', '')
        title = args.get('title', f'Meeting with {name}')

        # Parse ISO datetime string
        try:
            start = datetime.fromisoformat(datetime_str)
        except ValueError:
            return jsonify({
                "results": [{
                    "toolCallId": tool_call_id,
                    "result": "I couldn't parse that date and time. Please try again."
                }]
            })

        # Make start timezone aware for comparison
        # Why: datetime.now(pytz.UTC) is timezone-aware, so start must be too
        if start.tzinfo is None:
            start = start.replace(tzinfo=pytz.UTC)

        # Validate: reject past dates and times
        # Why: Safety net in case LLM misses the validation
        now = datetime.now(pytz.UTC)
        if start < now:
            return jsonify({
                "results": [{
                    "toolCallId": tool_call_id,
                    "result": "I'm sorry, that date and time has already passed. Please choose a future date and time."
                }]
            })

        # Default meeting duration is 1 hour
        end = start + timedelta(hours=1)

        # Create calendar event
        service = get_calendar_service()

        event_body = {
            'summary': title,
            'description': f'Scheduled via Voice Agent for {name}',
            'start': {
                'dateTime': start.isoformat(),
                'timeZone': 'UTC',
            },
            'end': {
                'dateTime': end.isoformat(),
                'timeZone': 'UTC',
            },
        }

        service.events().insert(
            calendarId=CALENDAR_ID,
            body=event_body
        ).execute()

        confirmation = (
            f"Done! I've created '{title}' for {name} on "
            f"{start.strftime('%A, %B %d, %Y at %I:%M %p')} UTC. "
            f"You're all set!"
        )

        return jsonify({
            "results": [{
                "toolCallId": tool_call_id,
                "result": confirmation
            }]
        })

    except Exception as e:
        return jsonify({
            "results": [{
                "toolCallId": tool_call_id,
                "result": f"Sorry, there was an error creating your event: {str(e)}"
            }]
        }), 200


if __name__ == '__main__':
    app.run(debug=True)