from flask import Flask, request, jsonify, Response
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from datetime import datetime, timedelta
from groq import Groq
import dateparser
import pytz
import os
import json
import uuid
import re
from dotenv import load_dotenv
load_dotenv()

# dateparser settings used everywhere:
# - PREFER_DATES_FROM: future ensures "Monday" means next Monday, not last
# - RETURN_AS_TIMEZONE_AWARE: always get a tz-aware datetime
# - TIMEZONE: treat bare times as UTC
DATEPARSER_SETTINGS = {
    "PREFER_DATES_FROM": "future",
    "RETURN_AS_TIMEZONE_AWARE": True,
    "TIMEZONE": "UTC",
    "PREFER_DAY_OF_MONTH": "first",
}

app = Flask(__name__)

groq_client = Groq(api_key=os.environ['GROQ_API_KEY'])
CALENDAR_ID = os.environ['CALENDAR_ID']

# ─── In-memory session store ──────────────────────────────────────────────────
# Keyed by call_id (from Vapi's message.call.id)
# Each session holds:
#   step       : current step name
#   name       : confirmed user name
#   date_iso   : confirmed date string "YYYY-MM-DD"
#   date_label : human label e.g. "Monday, February 23rd 2026"
#   time_iso   : confirmed time string "HH:MM" (24h UTC)
#   time_label : human label e.g. "2:00 PM UTC"
#   title      : confirmed meeting title
sessions = {}

STEPS = ["greet", "name", "date", "time", "title", "confirm", "done"]


def now_utc():
    return datetime.now(pytz.UTC)


def get_call_id(data):
    """Extract a stable call identifier from Vapi's request."""
    try:
        return data["call"]["id"]
    except (KeyError, TypeError):
        # Fallback: use first user message as fingerprint (less ideal)
        messages = data.get("messages", [])
        for m in messages:
            if m.get("role") == "user":
                return str(hash(m.get("content", "")))
        return "default"


def get_session(call_id):
    if call_id not in sessions:
        sessions[call_id] = {"step": "greet"}
    return sessions[call_id]


# ─── Step-specific system prompts ────────────────────────────────────────────
# Each prompt is laser-focused on ONE task. No state bleed, no confusion.

def prompt_greet():
    return (
        "You are Tara, a friendly scheduling assistant. "
        "Greet the user warmly and ask for their full name. "
        "Keep it to 2 sentences max."
    )

def prompt_name():
    return (
        "You are Tara, a scheduling assistant. "
        "The user has just responded. Extract their full name from their message. "
        "Once you have a name, confirm it warmly and tell them you'll now help pick a date. "
        "If you can't find a name, ask again politely."
    )

def prompt_date(name):
    today = now_utc()
    return (
        f"You are Tara, a scheduling assistant helping {name} book a meeting. "
        f"Today is {today.strftime('%A, %B %d, %Y')} UTC. "
        "Ask the user what date they'd like. "
        "When they answer, resolve any relative date (tomorrow, next Monday, etc.) "
        "to the exact calendar date. "
        "Confirm it out loud as 'Day, Month DDth YYYY' and include the ISO date in parentheses like (2026-02-23). "
        "If the date is in the past, politely ask for a future date. "
        "Do not ask for the time yet."
    )

def prompt_time(name, date_label):
    today = now_utc()
    current_time = today.strftime('%I:%M %p')
    return (
        f"You are Tara, a scheduling assistant. The meeting for {name} is confirmed for {date_label}. "
        f"Current UTC time is {current_time}. "
        "Ask what time they'd like. Accept natural language (2pm, 3:30 in the afternoon). "
        "Always clarify AM/PM if ambiguous. Confirm the time in UTC. "
        "If they picked today and the time has already passed, ask for a later time or different date. "
        "Do not ask for the title yet."
    )

def prompt_title(name, date_label, time_label):
    return (
        f"You are Tara, a scheduling assistant. "
        f"The meeting for {name} is set for {date_label} at {time_label}. "
        "Ask for a meeting title. Tell them it's optional — if they skip it, "
        f"you'll use 'Meeting with {name}'. Do not book anything yet."
    )

def prompt_confirm(name, date_label, time_label, title):
    return (
        f"You are Tara, a scheduling assistant. "
        f"Read back all details and ask for confirmation:\n"
        f"- Name: {name}\n"
        f"- Date: {date_label}\n"
        f"- Time: {time_label}\n"
        f"- Title: {title}\n"
        "Say something like: 'Just to confirm, I'll book [title] for [name] on [date] at [time]. Does that sound right?' "
        "Wait for yes or no. Do not call any functions."
    )


# ─── State extraction helpers ─────────────────────────────────────────────────

def extract_name_from_reply(user_text, last_assistant_text):
    """Ask LLM to extract just the name from a short exchange."""
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": (
                "Extract the person's full name from the user message. "
                "Return ONLY the name, nothing else. "
                "If you cannot find a name, return the empty string."
            )},
            {"role": "user", "content": f"Assistant asked: {last_assistant_text}\nUser replied: {user_text}"}
        ],
        temperature=0,
        max_tokens=20,
    )
    return resp.choices[0].message.content.strip()


def ordinal(day: int) -> str:
    """Return day with ordinal suffix, e.g. 23 -> '23rd'."""
    if 11 <= day <= 13:
        return f"{day}th"
    return f"{day}{['th','st','nd','rd','th','th','th','th','th','th'][day % 10]}"


def extract_date_from_reply(user_text: str) -> tuple[str, str]:
    """
    Use dateparser to parse a date from free-form user text.

    Returns (iso_date, human_label) e.g. ("2026-02-23", "Monday, February 23rd 2026")
    or ("", "") if parsing fails or date is in the past.

    Why dateparser beats an LLM extraction call here:
    - Deterministic: same input always gives same output
    - No hallucination risk
    - Handles dozens of natural language formats out of the box:
      "tomorrow", "next Monday", "March 5th", "in 3 days", "the 23rd", etc.
    - PREFER_DATES_FROM=future means "Monday" = next Monday, never last Monday
    """
    parsed = dateparser.parse(user_text, settings=DATEPARSER_SETTINGS)
    if not parsed:
        return "", ""

    # Normalize to midnight UTC so we compare dates cleanly
    parsed = parsed.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=pytz.UTC)
    today_midnight = now_utc().replace(hour=0, minute=0, second=0, microsecond=0)

    if parsed < today_midnight:
        return "", ""   # past date — caller will ask user to retry

    iso = parsed.strftime("%Y-%m-%d")
    label = f"{parsed.strftime('%A, %B')} {ordinal(parsed.day)} {parsed.year}"
    return iso, label


def extract_time_from_reply(user_text: str, date_iso: str) -> tuple[str, str]:
    """
    Use dateparser to parse a time from free-form user text.

    Anchors the parse to the already-confirmed date so that "2pm"
    resolves to the correct day.  Returns (HH:MM 24h, human_label)
    or ("", "") if ambiguous, unparseable, or already in the past.

    dateparser handles:
    - "2pm", "2 PM", "14:00", "half past two", "2:30 in the afternoon"
    - Prefers future times when PREFER_DATES_FROM=future

    One edge case dateparser can't solve alone: "3" is ambiguous (AM or PM).
    We catch that below and return "" so the LLM conversational layer can
    ask the user to clarify — the LLM is still used for the *conversation*,
    just not for date/time *parsing*.
    """
    # Build a combined string so dateparser has both date and time context
    combined = f"{date_iso} {user_text}"
    parsed = dateparser.parse(combined, settings=DATEPARSER_SETTINGS)

    if not parsed:
        # Try parsing the time expression alone (dateparser sometimes
        # rejects compound strings it can't fully tokenize)
        parsed = dateparser.parse(user_text, settings=DATEPARSER_SETTINGS)
        if not parsed:
            return "", ""
        # Graft the confirmed date onto the parsed time
        confirmed_date = datetime.fromisoformat(date_iso)
        parsed = parsed.replace(
            year=confirmed_date.year,
            month=confirmed_date.month,
            day=confirmed_date.day,
            tzinfo=pytz.UTC,
        )

    parsed = parsed.replace(tzinfo=pytz.UTC)

    # Detect bare hour with no AM/PM (e.g. user said "3" or "seven")
    # dateparser defaults to AM in that case — ambiguous, better to ask
    bare_hour_pattern = re.compile(
        r'^\s*\d{1,2}\s*$|^\s*(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*$',
        re.IGNORECASE
    )
    if bare_hour_pattern.match(user_text.strip()):
        return "", ""   # ambiguous — conversational LLM will ask AM/PM

    # Reject times already in the past
    if parsed < now_utc():
        return "", ""

    hhmm = parsed.strftime("%H:%M")
    label = parsed.strftime("%-I:%M %p") + " UTC"   # e.g. "2:00 PM UTC"
    return hhmm, label


def extract_title_from_reply(user_text, default_title):
    """Return meeting title from user reply, or default."""
    skip_phrases = ["no", "skip", "default", "that's fine", "sure", "you can", "use that", "go ahead"]
    lower = user_text.lower().strip()
    if any(p in lower for p in skip_phrases) or len(lower) < 3:
        return default_title
    # Use the user's text directly if it's short enough to be a title
    if len(user_text.strip()) < 80:
        return user_text.strip().strip('"').strip("'")
    return default_title


def is_confirmation(user_text):
    yes_phrases = ["yes", "yeah", "yep", "correct", "that's right", "sounds good",
                   "looks good", "confirmed", "book it", "go ahead", "sure", "absolutely"]
    lower = user_text.lower()
    return any(p in lower for p in yes_phrases)


# ─── Calendar ────────────────────────────────────────────────────────────────

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


def create_calendar_event(session):
    """Build ISO datetime from locked state and create the event."""
    date_iso = session["date_iso"]      # "YYYY-MM-DD"
    time_hhmm = session["time_iso"]    # "HH:MM"
    name = session["name"]
    title = session["title"]

    datetime_str = f"{date_iso}T{time_hhmm}:00"
    start = datetime.fromisoformat(datetime_str).replace(tzinfo=pytz.UTC)
    end = start + timedelta(hours=1)

    # Sanity check — never book in the past
    if start < now_utc():
        return "I'm sorry, that time has already passed. Let me know a new time."

    try:
        service = get_calendar_service()
        service.events().insert(
            calendarId=CALENDAR_ID,
            body={
                'summary': title,
                'description': f'Scheduled via Voice Agent for {name}',
                'start': {'dateTime': start.isoformat(), 'timeZone': 'UTC'},
                'end': {'dateTime': end.isoformat(), 'timeZone': 'UTC'},
            }
        ).execute()
        return (
            f"Done! I've booked '{title}' for {name} on "
            f"{start.strftime('%A, %B %d, %Y')} at {start.strftime('%I:%M %p')} UTC. "
            "You're all set — have a great day!"
        )
    except Exception as e:
        return f"Sorry, there was an error creating your event: {str(e)}"


# ─── Streaming helper ─────────────────────────────────────────────────────────

def stream_text(text, response_id):
    words = text.split(' ')
    for i, word in enumerate(words):
        chunk_content = word + ('' if i == len(words) - 1 else ' ')
        yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': int(now_utc().timestamp()), 'model': 'llama-3.3-70b-versatile', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': chunk_content}, 'finish_reason': None}]})}\n\n"
    yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': int(now_utc().timestamp()), 'model': 'llama-3.3-70b-versatile', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"


def build_response(text, response_id, stream):
    if stream:
        return Response(
            stream_text(text, response_id),
            mimetype='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
        )
    return jsonify({
        "id": response_id,
        "object": "chat.completion",
        "created": int(now_utc().timestamp()),
        "model": "llama-3.3-70b-versatile",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text, "tool_calls": None}, "finish_reason": "stop"}]
    })


# ─── LLM call (conversational only, no tools) ────────────────────────────────

def llm_respond(system_prompt, user_text, prior_assistant_text=None):
    """
    Single-turn or two-turn LLM call.
    We only send the system prompt + the immediate user message (+ optional
    prior assistant turn for context). Never the whole history.
    """
    messages = [{"role": "system", "content": system_prompt}]
    if prior_assistant_text:
        messages.append({"role": "assistant", "content": prior_assistant_text})
    messages.append({"role": "user", "content": user_text})

    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.4,
        max_tokens=200,
        stream=False,
    )
    return resp.choices[0].message.content.strip(), resp.id


# ─── Main endpoint ────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return "Voice Scheduling Agent is running!"


@app.route('/chat/completions', methods=['POST'])
def chat():
    try:
        data = request.json
        stream = data.get('stream', False)
        call_id = get_call_id(data)
        session = get_session(call_id)

        # Get latest user message
        messages = data.get('messages', [])
        user_messages = [m for m in messages if m.get('role') == 'user']
        assistant_messages = [m for m in messages if m.get('role') == 'assistant']
        user_text = user_messages[-1]['content'] if user_messages else ""
        last_assistant_text = assistant_messages[-1]['content'] if assistant_messages else ""

        step = session.get("step", "greet")
        response_id = str(uuid.uuid4())

        # ── GREET ────────────────────────────────────────────────────────────
        if step == "greet":
            reply, rid = llm_respond(prompt_greet(), "Hello")
            session["step"] = "name"
            session["last_assistant"] = reply
            return build_response(reply, rid, stream)

        # ── NAME ─────────────────────────────────────────────────────────────
        elif step == "name":
            name = extract_name_from_reply(user_text, last_assistant_text)
            if name:
                session["name"] = name
                session["step"] = "date"
                reply, rid = llm_respond(prompt_date(name), user_text, last_assistant_text)
                session["last_assistant"] = reply
                return build_response(reply, rid, stream)
            else:
                # Couldn't find name — ask again
                reply, rid = llm_respond(prompt_name(), user_text, last_assistant_text)
                session["last_assistant"] = reply
                return build_response(reply, rid, stream)

        # ── DATE ─────────────────────────────────────────────────────────────
        elif step == "date":
            name = session.get("name", "")
            date_iso, date_label = extract_date_from_reply(user_text)

            if date_iso:
                # Validate not in the past
                parsed = datetime.fromisoformat(date_iso).replace(tzinfo=pytz.UTC)
                today_date = now_utc().replace(hour=0, minute=0, second=0, microsecond=0)
                if parsed < today_date:
                    reply, rid = llm_respond(
                        prompt_date(name),
                        "That date is in the past. Please ask for a future date.",
                        last_assistant_text
                    )
                    session["last_assistant"] = reply
                    return build_response(reply, rid, stream)

                session["date_iso"] = date_iso
                session["date_label"] = date_label
                session["step"] = "time"
                reply, rid = llm_respond(prompt_time(name, date_label), user_text, last_assistant_text)
                session["last_assistant"] = reply
                return build_response(reply, rid, stream)
            else:
                # Couldn't resolve date — ask again
                reply, rid = llm_respond(prompt_date(name), user_text, last_assistant_text)
                session["last_assistant"] = reply
                return build_response(reply, rid, stream)

        # ── TIME ─────────────────────────────────────────────────────────────
        elif step == "time":
            name = session.get("name", "")
            date_iso = session.get("date_iso", "")
            date_label = session.get("date_label", "")
            time_hhmm, time_label = extract_time_from_reply(user_text, date_iso)

            if time_hhmm:
                session["time_iso"] = time_hhmm
                session["time_label"] = time_label
                session["step"] = "title"
                reply, rid = llm_respond(prompt_title(name, date_label, time_label), user_text, last_assistant_text)
                session["last_assistant"] = reply
                return build_response(reply, rid, stream)
            else:
                # Time in the past or ambiguous
                reply, rid = llm_respond(prompt_time(name, date_label), user_text, last_assistant_text)
                session["last_assistant"] = reply
                return build_response(reply, rid, stream)

        # ── TITLE ────────────────────────────────────────────────────────────
        elif step == "title":
            name = session.get("name", "")
            date_label = session.get("date_label", "")
            time_label = session.get("time_label", "")
            default_title = f"Meeting with {name}"
            title = extract_title_from_reply(user_text, default_title)
            session["title"] = title
            session["step"] = "confirm"
            reply, rid = llm_respond(prompt_confirm(name, date_label, time_label, title), user_text)
            session["last_assistant"] = reply
            return build_response(reply, rid, stream)

        # ── CONFIRM ──────────────────────────────────────────────────────────
        elif step == "confirm":
            if is_confirmation(user_text):
                session["step"] = "done"
                result = create_calendar_event(session)
                return build_response(result, response_id, stream)
            else:
                # User wants to change something — detect which field
                lower = user_text.lower()
                if any(w in lower for w in ["date", "day", "monday", "tuesday", "wednesday",
                                             "thursday", "friday", "saturday", "sunday",
                                             "tomorrow", "next", "january", "february",
                                             "march", "april", "may", "june", "july",
                                             "august", "september", "october", "november", "december"]):
                    session["step"] = "date"
                    session.pop("date_iso", None)
                    session.pop("date_label", None)
                    session.pop("time_iso", None)
                    session.pop("time_label", None)
                    session.pop("title", None)
                    reply, rid = llm_respond(prompt_date(session["name"]), user_text)
                elif any(w in lower for w in ["time", "am", "pm", "o'clock", "hour"]):
                    session["step"] = "time"
                    session.pop("time_iso", None)
                    session.pop("time_label", None)
                    session.pop("title", None)
                    reply, rid = llm_respond(
                        prompt_time(session["name"], session.get("date_label", "")), user_text
                    )
                elif any(w in lower for w in ["title", "name", "call it", "meeting"]):
                    session["step"] = "title"
                    session.pop("title", None)
                    reply, rid = llm_respond(
                        prompt_title(session["name"], session.get("date_label", ""), session.get("time_label", "")),
                        user_text
                    )
                else:
                    # Generic "no" — ask what to change
                    reply = "No problem! What would you like to change — the date, time, or title?"
                    rid = response_id
                session["last_assistant"] = reply
                return build_response(reply, rid, stream)

        # ── DONE (repeat booking or anything after) ──────────────────────────
        else:
            reply = "Your meeting is already booked! Is there anything else I can help you with?"
            return build_response(reply, response_id, stream)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)