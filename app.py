from flask import Flask, request, jsonify, Response
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from datetime import datetime, timedelta
from groq import Groq
import pytz
import os
import json
import re
from dotenv import load_dotenv
import dateparser
load_dotenv()

app = Flask(__name__)

groq_client = Groq(api_key=os.environ['GROQ_API_KEY'])
CALENDAR_ID = os.environ['CALENDAR_ID']

# In-memory store for conversation state
# In production, use Redis or a database
conversation_state = {}


def get_system_prompt(parsed_date=None, parsed_time=None, user_name=None):
    today = datetime.now(pytz.UTC)
    date_str = today.strftime('%A, %B %d, %Y')
    time_str = today.strftime('%I:%M %p')
    
    # Build context about what we know
    known_info = []
    missing_info = []
    
    if user_name:
        known_info.append(f"User's name: {user_name}")
    else:
        missing_info.append("user's name")
    
    if parsed_date:
        known_info.append(f"Meeting date: {parsed_date.strftime('%A, %B %d, %Y')} ({parsed_date.strftime('%Y-%m-%d')})")
    else:
        missing_info.append("meeting date")
    
    if parsed_time:
        known_info.append(f"Meeting time: {parsed_time.strftime('%I:%M %p')} UTC")
    else:
        missing_info.append("meeting time")
    
    context_section = ""
    if known_info:
        context_section = f"\nINFORMATION YOU HAVE:\n" + "\n".join(f"- {info}" for info in known_info)
        context_section += "\n\nIMPORTANT: When you reference dates or times in your responses, use the EXACT values above. Do NOT recalculate them.\n"
    
    if missing_info:
        context_section += f"\nINFORMATION STILL NEEDED: {', '.join(missing_info)}\n"

    return f"""You are Tara, a friendly and professional scheduling assistant for Google Calendar.
Today is {date_str} and the current time is {time_str} UTC.
{context_section}
Your goal is to help users schedule a calendar event. You need to collect:
1. The user's name
2. The meeting date and time
3. A meeting title (optional - default to "Meeting with [name]" if not provided)

CONVERSATION STYLE:
- Be natural, warm, and conversational - adapt to how the user speaks
- Users can provide information in ANY order - they might say "Schedule a meeting with John next Monday at 2pm" all at once
- Or they might provide information gradually across multiple messages
- Follow the user's lead - if they're direct, be direct; if they're chatty, be friendly
- Don't force a rigid question-answer format unless the user seems to prefer it

EXTRACTING INFORMATION:
- Listen for the user's name in any form: "I'm Nancy", "It's John Smith", "Call me Sarah"
- Listen for dates in natural language: "tomorrow", "next Monday", "March 5th", "in 2 weeks"
- Listen for times: "2pm", "3:30 in the afternoon", "10 AM", "morning"
- Listen for meeting titles: "Project Kickoff", "Team Meeting", etc.
- Users might provide everything at once or piece by piece - handle both naturally

WHEN YOU HAVE ALL REQUIRED INFO:
- Read back the details clearly: "Just to confirm, I'll book '[title]' for [name] on [date] at [time] UTC. Does that sound right?"
- Wait for explicit confirmation (yes/okay/sounds good/etc.)
- Only call createCalendarEvent after the user confirms

WHEN INFORMATION IS MISSING:
- Naturally ask for what's missing without being robotic
- If the user provides partial info, acknowledge it and ask for the rest naturally
- Example: User says "Next Monday" → "Got it, next Monday. What time works for you?"
- Example: User says "2pm" but no date → "2pm works! What date should we schedule this for?"

IMPORTANT RULES:
- Times are saved in UTC - mention this when relevant
- Never accept past dates or past times for today
- If EXTRACTED INFORMATION is provided above, use those EXACT values - don't recalculate
- Be conversational and flexible - adapt to the user's communication style
- If asked about unrelated topics, politely redirect: "I'm here to help you schedule calendar events. What would you like to book?"
- When calling createCalendarEvent, use ISO 8601 format: YYYY-MM-DDTHH:MM:SS
- If you have parsed date/time from EXTRACTED INFORMATION, use those exact values in the function call
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


def parse_date_from_text(text, reference_date=None):
    """
    Parses a date from natural language text using dateparser.
    Handles relative dates like "next Monday", "tomorrow", etc.
    """
    if not text:
        return None
    
    if reference_date is None:
        reference_date = datetime.now(pytz.UTC)
    
    # Clean up text
    text = text.strip()
    
    # Try parsing with dateparser
    # Use settings that prefer future dates for relative phrases
    parsed = dateparser.parse(
        text,
        settings={
            'RELATIVE_BASE': reference_date.replace(tzinfo=None),
            'PREFER_DATES_FROM': 'future',
            'RETURN_AS_TIMEZONE_AWARE': False,
            'PREFER_DAY_OF_MONTH': 'first'
        }
    )
    
    if parsed:
        # Make timezone aware in UTC
        if parsed.tzinfo is None:
            parsed = pytz.UTC.localize(parsed)
        else:
            parsed = parsed.astimezone(pytz.UTC)
        
        # Ensure it's a future date
        now = datetime.now(pytz.UTC)
        if parsed.date() < now.date():
            # If we got a past date but text suggests future, try again with more context
            if any(word in text.lower() for word in ['next', 'tomorrow', 'upcoming', 'coming']):
                # Add a week to reference date for "next" phrases
                future_ref = reference_date + timedelta(days=7)
                parsed = dateparser.parse(
                    text,
                    settings={
                        'RELATIVE_BASE': future_ref.replace(tzinfo=None),
                        'PREFER_DATES_FROM': 'future',
                        'RETURN_AS_TIMEZONE_AWARE': False
                    }
                )
                if parsed:
                    if parsed.tzinfo is None:
                        parsed = pytz.UTC.localize(parsed)
                    else:
                        parsed = parsed.astimezone(pytz.UTC)
        
        # Final check: if still in past, return None
        if parsed and parsed.date() < now.date():
            return None
        
        return parsed
    
    return None


def parse_time_from_text(text, reference_date=None):
    """
    Parses a time from natural language text.
    Returns a datetime with the parsed time on the reference_date.
    """
    if not text:
        return None
    
    if reference_date is None:
        reference_date = datetime.now(pytz.UTC)
    
    # Extract time patterns
    time_patterns = [
        r'(\d{1,2}):(\d{2})\s*(am|pm|AM|PM)',
        r'(\d{1,2})\s*(am|pm|AM|PM)',
        r'(\d{1,2}):(\d{2})',
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, text.lower())
        if match:
            groups = match.groups()
            if len(groups) == 3:  # HH:MM AM/PM
                hour = int(groups[0])
                minute = int(groups[1])
                am_pm = groups[2].lower()
                if am_pm == 'pm' and hour != 12:
                    hour += 12
                elif am_pm == 'am' and hour == 12:
                    hour = 0
            elif len(groups) == 2 and groups[1] in ['am', 'pm', 'AM', 'PM']:  # HH AM/PM
                hour = int(groups[0])
                minute = 0
                am_pm = groups[1].lower()
                if am_pm == 'pm' and hour != 12:
                    hour += 12
                elif am_pm == 'am' and hour == 12:
                    hour = 0
            elif len(groups) == 2:  # HH:MM (24-hour)
                hour = int(groups[0])
                minute = int(groups[1])
            else:
                continue
            
            # Create datetime with parsed time
            result = reference_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
            return result
    
    # Try dateparser for time
    parsed = dateparser.parse(text, settings={'PREFER_DATES_FROM': 'future'})
    if parsed:
        if parsed.tzinfo is None:
            parsed = pytz.UTC.localize(parsed)
        else:
            parsed = parsed.astimezone(pytz.UTC)
        # Use the time from parsed, but keep the reference date
        result = reference_date.replace(
            hour=parsed.hour,
            minute=parsed.minute,
            second=parsed.second,
            microsecond=0
        )
        return result
    
    return None


def extract_info_from_conversation(messages, conversation_id=None):
    """
    Extracts name, date, and time from conversation history using flexible parsing.
    Works with natural conversation flow - information can come in any order.
    """
    extracted = {
        'parsed_date': None,
        'parsed_time': None,
        'user_name': None
    }
    
    # Get conversation state if available (persist across requests)
    if conversation_id and conversation_id in conversation_state:
        state = conversation_state[conversation_id]
        extracted['parsed_date'] = state.get('parsed_date')
        extracted['parsed_time'] = state.get('parsed_time')
        extracted['user_name'] = state.get('user_name')
    
    # Scan ALL messages (both user and assistant) for information
    # This allows us to catch information mentioned anywhere in the conversation
    today = datetime.now(pytz.UTC)
    
    # Combine all user messages into one text for better context
    all_user_text = " ".join([
        msg.get('content', '') for msg in messages 
        if msg.get('role') == 'user'
    ])
    
    # Try to extract name from any user message
    if not extracted['user_name']:
        # More flexible name patterns
        name_patterns = [
            r"(?:it'?s|i'?m|my name is|this is|call me|i am|name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"(?:schedule|meeting|book)\s+(?:with|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)$"
        ]
        for pattern in name_patterns:
            match = re.search(pattern, all_user_text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Validate it looks like a name (1-3 words, mostly alphabetic)
                words = name.split()
                if 1 <= len(words) <= 3:
                    # Check if it's not a common word that might be misidentified
                    common_words = {'me', 'you', 'the', 'a', 'an', 'and', 'or', 'but', 'for', 'with'}
                    if name.lower() not in common_words:
                        extracted['user_name'] = name
                        break
    
    # Try to parse date from any message - be more aggressive
    # Parse the entire conversation text, not just individual messages
    if not extracted['parsed_date']:
        # Try parsing the full conversation context
        parsed_date = parse_date_from_text(all_user_text, today)
        if parsed_date and parsed_date.date() >= today.date():
            extracted['parsed_date'] = parsed_date
    
    # If still no date, try individual messages
    if not extracted['parsed_date']:
        for msg in messages:
            if msg.get('role') != 'user':
                continue
            content = msg.get('content', '').strip()
            parsed_date = parse_date_from_text(content, today)
            if parsed_date and parsed_date.date() >= today.date():
                extracted['parsed_date'] = parsed_date
                break
    
    # Try to parse time - use full context first
    if not extracted['parsed_time']:
        ref_date = extracted['parsed_date'] if extracted['parsed_date'] else today
        parsed_time = parse_time_from_text(all_user_text, ref_date)
        if parsed_time:
            if extracted['parsed_date']:
                parsed_time = extracted['parsed_date'].replace(
                    hour=parsed_time.hour,
                    minute=parsed_time.minute,
                    second=0,
                    microsecond=0
                )
            extracted['parsed_time'] = parsed_time
    
    # If still no time, try individual messages
    if not extracted['parsed_time']:
        for msg in messages:
            if msg.get('role') != 'user':
                continue
            content = msg.get('content', '').strip()
            ref_date = extracted['parsed_date'] if extracted['parsed_date'] else today
            parsed_time = parse_time_from_text(content, ref_date)
            if parsed_time:
                if extracted['parsed_date']:
                    parsed_time = extracted['parsed_date'].replace(
                        hour=parsed_time.hour,
                        minute=parsed_time.minute,
                        second=0,
                        microsecond=0
                    )
                extracted['parsed_time'] = parsed_time
                break
    
    # If we got a new date but already had a time, update the time's date
    if extracted['parsed_date'] and extracted['parsed_time']:
        # Ensure the time uses the correct date
        if extracted['parsed_time'].date() != extracted['parsed_date'].date():
            extracted['parsed_time'] = extracted['parsed_date'].replace(
                hour=extracted['parsed_time'].hour,
                minute=extracted['parsed_time'].minute,
                second=0,
                microsecond=0
            )
    
    # Update conversation state (persist extracted info)
    if conversation_id:
        if conversation_id not in conversation_state:
            conversation_state[conversation_id] = {}
        # Update with new information (don't overwrite with None)
        for key in ['parsed_date', 'parsed_time', 'user_name']:
            if extracted[key] is not None:
                conversation_state[conversation_id][key] = extracted[key]
    
    return extracted


def create_calendar_event(args, messages):
    """
    Creates a Google Calendar event.
    Uses the datetime from args which should already be corrected by server-side parsing.
    """
    name = args.get('name', 'Guest')
    datetime_str = args.get('datetime', '')
    title = args.get('title', f'Meeting with {name}')

    try:
        # Handle both with and without timezone
        if datetime_str.endswith('Z'):
            datetime_str = datetime_str.replace('Z', '+00:00')
        start = datetime.fromisoformat(datetime_str)
    except ValueError:
        return "I couldn't parse that date and time. Please try again."

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
        
        # Get or generate conversation ID
        # Vapi might send a user ID or we can use a hash of messages
        conversation_id = data.get('user', None) or str(hash(str(messages[:3])))
        
        # Extract information from conversation (server-side parsing)
        extracted = extract_info_from_conversation(messages, conversation_id)
        
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
                    "description": """Creates a Google Calendar event. 
                    
                    Call this function ONLY after:
                    1. You have collected the user's name, date, and time
                    2. The user has explicitly confirmed the details (said yes/okay/sounds good/etc.)
                    
                    Extract information naturally from the conversation - users may provide it in any order.
                    If you have parsed date/time from the EXTRACTED INFORMATION section, use those exact values.
                    """,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The user's full name as mentioned in the conversation"
                            },
                            "datetime": {
                                "type": "string",
                                "description": "Meeting date and time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS). Use the exact date/time from EXTRACTED INFORMATION if available. Example: 2026-02-23T14:00:00"
                            },
                            "title": {
                                "type": "string",
                                "description": "Meeting title. Use what the user specified, or default to 'Meeting with [name]' if not provided."
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
        if message.tool_calls:
            tc = message.tool_calls[0]
            args = json.loads(tc.function.arguments)
            
            # Override with server-parsed values (source of truth to prevent hallucinations)
            # Priority: server-parsed > LLM-extracted
            if extracted['parsed_date']:
                if extracted['parsed_time']:
                    # We have both date and time from server parsing - use them
                    combined_datetime = extracted['parsed_time']
                else:
                    # We have date but not time - try to get time from LLM's function call
                    datetime_str = args.get('datetime', '')
                    try:
                        parsed_from_args = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                        # Use server-parsed date with LLM-extracted time
                        combined_datetime = extracted['parsed_date'].replace(
                            hour=parsed_from_args.hour,
                            minute=parsed_from_args.minute,
                            second=parsed_from_args.second,
                            microsecond=0
                        )
                    except:
                        # If LLM's time is invalid, use a default time (2 PM)
                        combined_datetime = extracted['parsed_date'].replace(hour=14, minute=0, second=0, microsecond=0)
                
                args['datetime'] = combined_datetime.isoformat()
            elif extracted['parsed_time']:
                # We have time but not date - use LLM's date with server-parsed time
                datetime_str = args.get('datetime', '')
                try:
                    parsed_from_args = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                    combined_datetime = parsed_from_args.replace(
                        hour=extracted['parsed_time'].hour,
                        minute=extracted['parsed_time'].minute,
                        second=0,
                        microsecond=0
                    )
                    args['datetime'] = combined_datetime.isoformat()
                except:
                    pass  # If parsing fails, let LLM's value stand
            
            # Override name if we have it from server parsing
            if extracted['user_name']:
                args['name'] = extracted['user_name']
            
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