from flask import Flask, request, jsonify, Response
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from datetime import datetime, timedelta
from groq import Groq
import pytz
import os
import json
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

groq_client = Groq(api_key=os.environ['GROQ_API_KEY'])
CALENDAR_ID = os.environ['CALENDAR_ID']


def get_system_prompt():
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
- Example: "What time works for you? I'll save it in UTC."
- If the user picks today's date, make sure the time is in the future.
  Current UTC time is {time_str}. If the time has already passed today, say:
  "That time has already passed today. Could you pick a later time,
  or would you prefer a different date?"

STEP 5: Ask for a meeting title (tell them it's optional).
- If they skip it, default to "Meeting with [their name]"

STEP 6: Read back ALL details clearly including UTC.
Example: "Just to confirm - I'll book 'Project Kickoff'
for John on Thursday February 20th 2026 at 2:00 PM UTC.
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


def create_calendar_event(args):
    """
    Creates a Google Calendar event from tool call arguments.
    Returns a confirmation string or error message.
    Called directly from /chat/completions when Groq triggers the tool.
    Why: Vapi's Custom LLM doesn't forward tool calls to server URLs,
    so we handle everything inside /chat/completions directly.
    """
    name = args.get('name', 'Guest')
    datetime_str = args.get('datetime', '')
    title = args.get('title', f'Meeting with {name}')

    try:
        start = datetime.fromisoformat(datetime_str)
    except ValueError:
        return "I couldn't parse that date and time. Please try again."

    if start.tzinfo is None:
        start = start.replace(tzinfo=pytz.UTC)

    now = datetime.now(pytz.UTC)
    if start < now:
        return "I'm sorry, that date and time has already passed. Please choose a future date and time."

    try:
        end = start + timedelta(hours=1)
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

        return (
            f"Done! I've created '{title}' for {name} on "
            f"{start.strftime('%A, %B %d, %Y at %I:%M %p')} UTC. "
            f"You're all set!"
        )

    except Exception as e:
        return f"Sorry, there was an error creating your event: {str(e)}"


def stream_text(text, response_id):
    """
    Streams text word by word in OpenAI SSE format.
    Why: Vapi needs streaming for real-time speech output.
    """
    words = text.split(' ')
    for i, word in enumerate(words):
        chunk_content = word + ('' if i == len(words) - 1 else ' ')
        chunk_data = {
            "id": response_id,
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

    # Final chunk
    done_data = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": "llama-3.3-70b-versatile",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(done_data)}\n\n"
    yield "data: [DONE]\n\n"


@app.route('/')
def home():
    return "Voice Scheduling Agent is running!"


@app.route('/chat/completions', methods=['POST'])
def chat():
    """
    Custom LLM endpoint for Vapi.

    Flow:
    1. Call Groq non-streaming to get full response
    2. If tool call detected → create calendar event directly here
       → return confirmation as streamed text to Vapi
    3. If regular text → stream it to Vapi
    
    Why handle tool calls here instead of /create-event:
    Vapi's Custom LLM provider does not forward tool calls
    from the LLM response to the tool's server URL (tested previously with multiple attempts and support confirmation).
    Everything must be handled inside this endpoint.
    """
    try:
        data = request.json
        messages = data.get('messages', [])
        stream = data.get('stream', False)

        # Replace Vapi's system message with our dynamic one
        messages = [m for m in messages if m.get('role') != 'system']
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

        # Always call Groq non-streaming first so we can inspect the response
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

        # CASE 1: Tool call detected
        # Create the calendar event directly here and return confirmation as text
        # Why: Vapi Custom LLM doesn't forward tool calls to server URLs
        if message.tool_calls:
            tc = message.tool_calls[0]
            args = json.loads(tc.function.arguments)
            confirmation = create_calendar_event(args)

            if stream:
                return Response(
                    stream_text(confirmation, response.id),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
                )

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
                            "content": confirmation,
                            "tool_calls": None
                        },
                        "finish_reason": "stop"
                    }
                ]
            })

        # CASE 2: Regular text response — stream it
        content = message.content or ""

        if stream:
            return Response(
                stream_text(content, response.id),
                mimetype='text/event-stream',
                headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
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
                        "content": content,
                        "tool_calls": None
                    },
                    "finish_reason": "stop"
                }
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# @app.route('/create-event', methods=['POST'])
# def create_event():
#     """
#     Kept as backup endpoint.
#     Primary calendar creation now happens inside /chat/completions.
#     """
#     tool_call_id = ''
#     try:
#         data = request.json
#         tool_calls = data.get('message', {}).get('toolCalls', [])

#         if tool_calls:
#             args = tool_calls[0].get('function', {}).get('arguments', {})
#             tool_call_id = tool_calls[0].get('id', '')
#             if isinstance(args, str):
#                 args = json.loads(args)
#         else:
#             args = data

#         confirmation = create_calendar_event(args)

#         return jsonify({
#             "results": [{
#                 "toolCallId": tool_call_id,
#                 "result": confirmation
#             }]
#         })

#     except Exception as e:
#         return jsonify({
#             "results": [{
#                 "toolCallId": tool_call_id,
#                 "result": f"Sorry, there was an error: {str(e)}"
#             }]
#         }), 200


if __name__ == '__main__':
    app.run(debug=True)