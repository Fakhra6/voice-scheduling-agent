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
        # Format date in a clear, readable way for the LLM to use
        day_name = parsed_date.strftime('%A')
        month = parsed_date.strftime('%B')
        day = parsed_date.strftime('%d').lstrip('0')
        year = parsed_date.strftime('%Y')
        # Add ordinal suffix (1st, 2nd, 3rd, etc.)
        if day.endswith('1') and day != '11':
            suffix = 'st'
        elif day.endswith('2') and day != '12':
            suffix = 'nd'
        elif day.endswith('3') and day != '13':
            suffix = 'rd'
        else:
            suffix = 'th'
        formatted_date = f"{day_name}, {month} {day}{suffix}, {year}"
        known_info.append(f"Meeting date: {formatted_date} ({parsed_date.strftime('%Y-%m-%d')})")
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
- CRITICAL: Always state the EXACT resolved date clearly before asking for confirmation
- Example: "Got it! So that's Monday, February 23rd, 2026 at 2:00 PM UTC. Just to confirm, I'll book '[title]' for [name] on Monday, February 23rd, 2026 at 2:00 PM UTC. Does that sound right?"
- Always include both the day name AND the full date (e.g., "Monday, February 23rd, 2026") so the user can verify
- Wait for explicit confirmation (yes/okay/sounds good/etc.)
- Only call createCalendarEvent after the user confirms
- When calling the function, use the EXACT ISO date from EXTRACTED INFORMATION - never recalculate

WHEN INFORMATION IS MISSING:
- Naturally ask for what's missing without being robotic
- If the user provides partial info, acknowledge it and ask for the rest naturally
- Example: User says "Next Monday" → "Got it, next Monday. That would be Monday, February 23rd, 2026. What time works for you?"
- Example: User says "2pm" but no date → "2pm works! What date should we schedule this for?"
- Always resolve relative dates to actual dates when acknowledging them

IMPORTANT RULES:
- CRITICAL: NEVER predict, assume, or guess dates or times
- ONLY use dates and times that the user has explicitly provided
- If the user hasn't provided a time, ask for it - DO NOT assume a default time
- If the user hasn't provided a date, ask for it - DO NOT assume a default date
- Times are saved in UTC - mention this when relevant
- Never accept past dates or past times for today
- If EXTRACTED INFORMATION is provided above, use those EXACT values - don't recalculate
- Be conversational and flexible - adapt to the user's communication style
- If asked about unrelated topics, politely redirect: "I'm here to help you schedule calendar events. What would you like to book?"
- When calling createCalendarEvent, use ISO 8601 format: YYYY-MM-DDTHH:MM:SS
- If you have parsed date/time from EXTRACTED INFORMATION, use those exact values in the function call
- DO NOT call createCalendarEvent unless you have BOTH date AND time explicitly from the user
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
    Dynamically parses a date from natural language using dateparser.
    Handles "next Monday", "tomorrow", etc. with explicit calculation for "next [day]" patterns.
    """
    if not text:
        return None
    
    if reference_date is None:
        reference_date = datetime.now(pytz.UTC)
    
    text_lower = text.lower().strip()
    now = datetime.now(pytz.UTC)
    
    # Strategy 1: Handle "next [day]" patterns explicitly for accuracy
    # This is dynamic - we detect day names without rigid patterns
    days_map = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6,
        'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 6
    }
    
    # Check for "next [day]" or "upcoming [day]" patterns dynamically
    # "upcoming" = next occurrence (this week or next week)
    # "next" = the one after upcoming (add extra week)
    if 'next' in text_lower or 'upcoming' in text_lower:
        for day_name, day_num in days_map.items():
            if day_name in text_lower:
                # Calculate occurrence of that day
                current_weekday = reference_date.weekday()
                days_ahead = day_num - current_weekday
                
                if days_ahead <= 0:
                    # Day already passed this week, go to next week
                    days_ahead += 7
                
                # "next" means add another week (the week after upcoming)
                # "upcoming" means just the next occurrence (this week or next week)
                if 'next' in text_lower:
                    days_ahead += 7
                
                target_date = reference_date + timedelta(days=days_ahead)
                result = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
                
                # Validate it's in the future
                if result.date() >= now.date():
                    return result
    
    # Strategy 2: Use dateparser for other patterns
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
        
        # Validate the parsed date makes sense
        # If dateparser gave us a wrong day of week for "next [day]" or "upcoming [day]", recalculate
        if 'next' in text_lower or 'upcoming' in text_lower:
            for day_name, day_num in days_map.items():
                if day_name in text_lower:
                    # Check if parsed date matches the expected day
                    if parsed.weekday() != day_num:
                        # dateparser got it wrong, recalculate
                        current_weekday = reference_date.weekday()
                        days_ahead = day_num - current_weekday
                        if days_ahead <= 0:
                            days_ahead += 7
                        # "next" means add another week, "upcoming" doesn't
                        if 'next' in text_lower:
                            days_ahead += 7
                        target_date = reference_date + timedelta(days=days_ahead)
                        result = target_date.replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0)
                        if result.date() >= now.date():
                            return result
        
        # If we got a past date but text suggests future, try again
        if parsed.date() < now.date() and any(word in text_lower for word in ['next', 'tomorrow', 'upcoming']):
            # Try with a future reference point
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
        
        # Final validation: must be future date
        if parsed and parsed.date() >= now.date():
            return parsed
    
    # Strategy 3: Try with cleaned text (remove filler words)
    cleaned_text = text_lower
    for word in ['on', 'for', 'the', 'a', 'an']:
        cleaned_text = cleaned_text.replace(f' {word} ', ' ')
    
    if cleaned_text != text_lower:
        parsed = dateparser.parse(
            cleaned_text,
            settings={
                'RELATIVE_BASE': reference_date.replace(tzinfo=None),
                'PREFER_DATES_FROM': 'future',
                'RETURN_AS_TIMEZONE_AWARE': False
            }
        )
        if parsed:
            if parsed.tzinfo is None:
                parsed = pytz.UTC.localize(parsed)
            else:
                parsed = parsed.astimezone(pytz.UTC)
            if parsed.date() >= now.date():
                return parsed
    
    return None


def parse_time_from_text(text, reference_date=None):
    """
    Dynamically parses a time from natural language using dateparser.
    Returns a datetime with the parsed time on the reference_date.
    Handles single digits like "4" as 4 PM (afternoon) - common for meetings.
    """
    if not text:
        return None
    
    if reference_date is None:
        reference_date = datetime.now(pytz.UTC)
    
    text_lower = text.lower().strip()
    
    # Handle single digit times (e.g., "4" without AM/PM)
    # For voice conversations, single digits usually mean PM (afternoon meetings)
    # Let dateparser try first, but if it fails or gives midnight, interpret as PM
    parsed = dateparser.parse(
        text,
        settings={
            'RELATIVE_BASE': reference_date.replace(tzinfo=None),
            'PREFER_DATES_FROM': 'future',
            'RETURN_AS_TIMEZONE_AWARE': False
        }
    )
    
    if parsed:
        if parsed.tzinfo is None:
            parsed = pytz.UTC.localize(parsed)
        else:
            parsed = parsed.astimezone(pytz.UTC)
        
        # Check if it's just a single digit (like "4")
        # If dateparser gave us midnight (00:00) for a single digit, assume PM
        if parsed.hour == 0 and parsed.minute == 0:
            # Check if text is just a number 1-12
            try:
                hour_num = int(text_lower.strip())
                if 1 <= hour_num <= 12:
                    # Single digit without AM/PM - assume PM for meetings
                    hour_24 = 12 if hour_num == 12 else hour_num + 12
                    result = reference_date.replace(hour=hour_24, minute=0, second=0, microsecond=0)
                    return result
            except ValueError:
                pass
        
        # Extract just the time components and apply to reference date
        result = reference_date.replace(
            hour=parsed.hour,
            minute=parsed.minute,
            second=0,
            microsecond=0
        )
        return result
    
    # If dateparser failed, try interpreting single digits as PM
    try:
        hour_num = int(text_lower.strip())
        if 1 <= hour_num <= 12:
            # Single digit without AM/PM - assume PM for meetings
            hour_24 = 12 if hour_num == 12 else hour_num + 12
            result = reference_date.replace(hour=hour_24, minute=0, second=0, microsecond=0)
            return result
    except ValueError:
        pass

    return None


def extract_info_from_conversation(messages, conversation_id=None):
    """
    Dynamically extracts information from conversation using LLM extraction + server-side validation.
    No regex patterns - relies on LLM to extract naturally, then validates with dateparser.
    CRITICAL: Always re-parse from recent messages to catch date/time changes.
    """
    extracted = {
        'parsed_date': None,
        'parsed_time': None,
        'user_name': None
    }
    
    today = datetime.now(pytz.UTC)
    
    # Combine all user messages for context-aware parsing
    all_user_text = " ".join([
        msg.get('content', '') for msg in messages 
        if msg.get('role') == 'user'
    ])
    
    # CRITICAL: Always re-parse dates from recent messages first
    # This ensures we catch when user changes the date (e.g., "no i want it on upcoming tuesday")
    # Check messages in reverse order (most recent first) to catch changes
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            content = msg.get('content', '').strip()
            if content:
                # Check if this message contains a date reference
                # Use dateparser to detect date mentions naturally
                parsed_date = parse_date_from_text(content, today)
                if parsed_date:
                    # Found a date in recent message - use it (overrides any previous date)
                    extracted['parsed_date'] = parsed_date
                    break
    
    # Fallback: If no date found in individual messages, try full context
    if not extracted['parsed_date'] and all_user_text:
        parsed_date = parse_date_from_text(all_user_text, today)
        if parsed_date:
            extracted['parsed_date'] = parsed_date
    
    # Parse time dynamically from user messages
    # CRITICAL: Always re-parse from recent messages to catch time changes
    ref_date = extracted['parsed_date'] if extracted['parsed_date'] else today
    
    # Check messages in reverse order (most recent first) to catch changes
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            content = msg.get('content', '').strip()
            if content:
                parsed_time = parse_time_from_text(content, ref_date)
                if parsed_time:
                    # Found a time in recent message - use it (overrides any previous time)
                    if extracted['parsed_date']:
                        parsed_time = extracted['parsed_date'].replace(
                            hour=parsed_time.hour,
                            minute=parsed_time.minute,
                            second=0,
                            microsecond=0
                        )
                    extracted['parsed_time'] = parsed_time
                    break
    
    # Fallback: If no time found in individual messages, try full context
    if not extracted['parsed_time'] and all_user_text:
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
    
    # Ensure date and time are synchronized
    if extracted['parsed_date'] and extracted['parsed_time']:
        if extracted['parsed_time'].date() != extracted['parsed_date'].date():
            extracted['parsed_time'] = extracted['parsed_date'].replace(
                hour=extracted['parsed_time'].hour,
                minute=extracted['parsed_time'].minute,
                second=0,
                microsecond=0
            )
    
    # Update conversation state
    if conversation_id:
        if conversation_id not in conversation_state:
            conversation_state[conversation_id] = {}
        for key in ['parsed_date', 'parsed_time', 'user_name']:
            if extracted[key] is not None:
                conversation_state[conversation_id][key] = extracted[key]
    
    return extracted


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
                    1. You have collected the user's name, date, AND time (all explicitly provided by user)
                    2. The user has explicitly confirmed the details (said yes/okay/sounds good/etc.)
                    
                    CRITICAL RULES:
                    - NEVER assume, predict, or guess dates or times
                    - ONLY use dates/times the user explicitly provided
                    - If user hasn't given a time, ask for it - DO NOT call this function
                    - If user hasn't given a date, ask for it - DO NOT call this function
                    
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
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
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
            
            # CRITICAL: Always override with server-parsed values if available
            # This prevents LLM date hallucinations
            datetime_str = args.get('datetime', '')
            
            # Priority 1: Use server-parsed date/time (most reliable - prevents hallucinations)
            # CRITICAL: Never assume or predict times - only use what user explicitly provided
            if extracted['parsed_date']:
                if extracted['parsed_time']:
                    # We have both date and time from server parsing - ALWAYS use them
                    combined_datetime = extracted['parsed_time']
                    args['datetime'] = combined_datetime.isoformat()
                else:
                    # We have date but NOT time from server parsing
                    # Check if LLM extracted time from user input (not assumed)
                    try:
                        parsed_from_args = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                        # Use server-parsed date with LLM-extracted time
                        # The LLM should only provide times the user explicitly mentioned
                        combined_datetime = extracted['parsed_date'].replace(
                            hour=parsed_from_args.hour,
                            minute=parsed_from_args.minute,
                            second=0,
                            microsecond=0
                        )
                        args['datetime'] = combined_datetime.isoformat()
                    except:
                        # Cannot parse time - this should not happen if LLM followed instructions
                        # Let create_calendar_event handle the error
                        pass
            elif extracted['parsed_time']:
                # We have time but not date - use LLM's date with server-parsed time
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
            else:
                # No server-parsed values - validate LLM's datetime
                # Try to re-parse from conversation as a safety check
                try:
                    llm_datetime = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                    now = datetime.now(pytz.UTC)
                    
                    # Re-parse from conversation to validate
                    all_user_text = " ".join([
                        msg.get('content', '') for msg in messages 
                        if msg.get('role') == 'user'
                    ])
                    
                    if all_user_text:
                        # Try to parse date from conversation
                        validated_date = parse_date_from_text(all_user_text, now)
                        if validated_date:
                            # If validated date differs significantly from LLM's date, use validated
                            # This catches hallucinations even if date is in the future
                            llm_date = llm_datetime.replace(tzinfo=pytz.UTC).date()
                            validated_date_only = validated_date.date()
                            
                            # If dates differ by more than 1 day, use validated date
                            if abs((validated_date_only - llm_date).days) > 1:
                                # Use validated date with LLM's time
                                combined_datetime = validated_date.replace(
                                    hour=llm_datetime.hour,
                                    minute=llm_datetime.minute,
                                    second=0,
                                    microsecond=0
                                )
                                args['datetime'] = combined_datetime.isoformat()
                            elif llm_datetime.replace(tzinfo=pytz.UTC) < now:
                                # LLM gave us a past date - use validated date
                                combined_datetime = validated_date.replace(
                                    hour=llm_datetime.hour,
                                    minute=llm_datetime.minute,
                                    second=0,
                                    microsecond=0
                                )
                                args['datetime'] = combined_datetime.isoformat()
                except:
                    pass  # If validation fails, proceed with LLM's value
            
            # Override name if we have it from server parsing (though LLM should handle this)
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