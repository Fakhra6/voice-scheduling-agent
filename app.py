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


def get_system_prompt(parsed_date=None, parsed_time=None, user_name=None):
    today = datetime.now(pytz.UTC)
    
    # Build context about what we know
    known_info = []
    if user_name:
        known_info.append(f"Name: {user_name}")
    if parsed_date:
        known_info.append(f"Date: {parsed_date.strftime('%A, %B %d, %Y')}")
    if parsed_time:
        known_info.append(f"Time: {parsed_time.strftime('%I:%M %p')} UTC")
    
    context = ""
    if known_info:
        context = f"\n\nEXTRACTED INFO (use these exact values):\n" + "\n".join(f"- {info}" for info in known_info)

    return f"""You are Tara, a friendly scheduling assistant for Google Calendar.
Today is {today.strftime('%A, %B %d, %Y')} at {today.strftime('%I:%M %p')} UTC.
{context}

Help users schedule calendar events by collecting: name, date, time, and optional title.

Be natural and conversational. Users may provide info all at once or gradually - adapt to their style.

When confirming, always state the full resolved date (e.g., "Tuesday, February 24th, 2026 at 4:00 PM UTC").
Only call createCalendarEvent after user explicitly confirms. Use EXTRACTED INFO values if available.
Times are in UTC. Never accept past dates/times."""


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


def extract_info_with_llm(messages):
    """
    Uses LLM to intelligently extract name, date, and time from the conversation.
    Simple and reliable - lets the LLM understand natural language contextually.
    """
    today = datetime.now(pytz.UTC)
    
    # Build conversation text for extraction
    conversation_text = "\n".join([
        f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}"
        for msg in messages
        if msg.get('role') in ['user', 'assistant'] and msg.get('content')
    ])
    
    extraction_prompt = f"""Today is {today.strftime('%A, %B %d, %Y')} at {today.strftime('%H:%M')} UTC.

From this conversation, extract what the USER explicitly stated:

{conversation_text}

Return JSON:
{{"name": "user's name or null", "date": "YYYY-MM-DD or null", "time": "HH:MM (24h) or null", "title": "meeting title or null"}}

Convert relative dates (tomorrow, next week, upcoming Monday, etc.) to actual dates. For ambiguous times like "4" in a meeting context, assume PM.
Only extract what was explicitly stated. Return ONLY valid JSON."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0,
            max_tokens=200
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON from response (handle markdown code blocks)
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        result_text = result_text.strip()
        
        extracted_json = json.loads(result_text)
        
        # Convert to datetime objects
        extracted = {
            'user_name': extracted_json.get('name'),
            'parsed_date': None,
            'parsed_time': None,
            'title': extracted_json.get('title')
        }
        
        date_str = extracted_json.get('date')
        time_str = extracted_json.get('time')
        
        if date_str:
            try:
                parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
                extracted['parsed_date'] = pytz.UTC.localize(parsed_date)
            except ValueError:
                pass
        
        if time_str and extracted['parsed_date']:
            try:
                hours, minutes = map(int, time_str.split(':'))
                extracted['parsed_time'] = extracted['parsed_date'].replace(
                    hour=hours, minute=minutes, second=0, microsecond=0
                )
            except (ValueError, AttributeError):
                pass
        elif time_str:
            # Have time but no date yet
            try:
                hours, minutes = map(int, time_str.split(':'))
                extracted['parsed_time'] = today.replace(
                    hour=hours, minute=minutes, second=0, microsecond=0
                )
            except (ValueError, AttributeError):
                pass
        
        return extracted
        
    except Exception as e:
        # Fallback: return empty extraction on error
        return {
            'user_name': None,
            'parsed_date': None,
            'parsed_time': None,
            'title': None
        }


def create_calendar_event(args, messages):
    """
    Creates a Google Calendar event.
    Uses the datetime from args which should already be corrected by server-side parsing.
    NEVER assumes or predicts dates/times - only uses what user explicitly provided.
    """
    name = args.get('name', 'Guest')
    datetime_str = args.get('datetime', '')
    title = args.get('title', f'Meeting with {name}')

    # Validation: Ensure we have both date and time
    if not datetime_str:
        return "I need both a date and time to create the event. Please provide the time."

    try:
        # Handle both with and without timezone
        if datetime_str.endswith('Z'):
            datetime_str = datetime_str.replace('Z', '+00:00')
        start = datetime.fromisoformat(datetime_str)
    except ValueError:
        return "I couldn't parse that date and time. Please provide a valid date and time."

    if start.tzinfo is None:
        start = start.replace(tzinfo=pytz.UTC)
    else:
        # Ensure it's in UTC
        start = start.astimezone(pytz.UTC)

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
    1. Extract information from conversation (name, date, time)
    2. Parse dates/times server-side to avoid LLM hallucination
    3. Inject parsed info into system prompt
    4. Call Groq non-streaming to get full response
    5. If tool call detected -> create calendar event directly here
       -> return confirmation as streamed text to Vapi
    6. If regular text -> stream it to Vapi

    Why handle tool calls here instead of /create-event:
    Vapi's Custom LLM provider does not forward tool calls
    from the LLM response to the tool server URL.
    Everything must be handled inside this endpoint.
    """
    try:
        data = request.json
        messages = data.get('messages', [])
        stream = data.get('stream', False)

        # Extract information from conversation using LLM (simple and reliable)
        extracted = extract_info_with_llm(messages)
        
        # Replace Vapi's system message with our dynamic one that includes parsed info
        messages = [m for m in messages if m.get('role') != 'system']
        messages = [{'role': 'system', 'content': get_system_prompt(
            parsed_date=extracted['parsed_date'],
            parsed_time=extracted['parsed_time'],
            user_name=extracted['user_name']
        )}] + messages

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "createCalendarEvent",
                    "description": "Creates a calendar event. Call only after user confirms the date, time, and details.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "User's name"},
                            "datetime": {"type": "string", "description": "ISO 8601 datetime (YYYY-MM-DDTHH:MM:SS). Use EXTRACTED INFO values."},
                            "title": {"type": "string", "description": "Meeting title (default: 'Meeting with [name]')"}
                        },
                        "required": ["name", "datetime"]
                    }
                }
            }
        ]

        # Always call Groq non-streaming first so we can inspect the response
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
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
        if message.tool_calls:
            # Check if event was already created in this conversation (prevent duplicates)
            for msg in messages:
                content = msg.get('content', '')
                if msg.get('role') == 'assistant' and "Done! I've created" in content:
                    # Event already created - just acknowledge
                    already_done = "Your event has already been created! Is there anything else I can help you with?"
                    if stream:
                        return Response(
                            stream_text(already_done, response.id),
                            mimetype='text/event-stream',
                            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
                        )
                    return jsonify({
                        "id": response.id,
                        "object": "chat.completion",
                        "created": int(datetime.now().timestamp()),
                        "model": response.model,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": already_done}, "finish_reason": "stop"}]
                    })
            
            tc = message.tool_calls[0]
            args = json.loads(tc.function.arguments)
            
            # ALWAYS use the LLM-extracted date/time - it's reliable and understands context
            if extracted['parsed_time']:
                # We have both date and time from LLM extraction - use them
                args['datetime'] = extracted['parsed_time'].isoformat()
            elif extracted['parsed_date']:
                # We have date but no time - try to get time from the tool call args
                datetime_str = args.get('datetime', '')
                try:
                    parsed_from_args = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                    combined_datetime = extracted['parsed_date'].replace(
                        hour=parsed_from_args.hour,
                        minute=parsed_from_args.minute,
                        second=0,
                        microsecond=0
                    )
                    args['datetime'] = combined_datetime.isoformat()
                except:
                    pass
            
            # Use extracted name and title if available
            if extracted['user_name']:
                args['name'] = extracted['user_name']
            if extracted.get('title'):
                args['title'] = extracted['title']
            
            confirmation = create_calendar_event(args, messages)

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


if __name__ == '__main__':
    app.run(debug=True)