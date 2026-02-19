from flask import Flask, request, jsonify
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


@app.route('/chat/completions', methods=['POST'])
def chat():
    """
    Custom LLM endpoint that Vapi calls for every conversation turn.
    Why: By intercepting here we can inject today's date and current time
    into the system prompt dynamically before forwarding to Groq.
    Vapi expects an OpenAI-compatible response format.
    """
    try:
        data = request.json

        # Extract conversation history from Vapi's request
        messages = data.get('messages', [])

        # Remove any system message Vapi sends — we replace with ours
        messages = [m for m in messages if m.get('role') != 'system']

        # Inject our dynamic system prompt with today's date and time
        messages = [{'role': 'system', 'content': get_system_prompt()}] + messages

        # Define the calendar tool so Groq knows when and how to call it
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

        # Forward to Groq
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.3,  # Low temp = consistent, predictable behavior
            max_tokens=1000
        )

        choice = response.choices[0]
        message = choice.message

        # Build tool_calls in OpenAI format if present
        tool_calls = None
        if message.tool_calls:
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

        # Return in OpenAI-compatible format that Vapi expects
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
                        "tool_calls": tool_calls
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