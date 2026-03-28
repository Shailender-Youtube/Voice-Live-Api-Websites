"""
Course Atlas Voice Server
WebSocket bridge between Azure VoiceLive and the React frontend.

Filter updates use function calling (set_filters tool) — the AI silently calls
the function whenever it wants to update the catalog view. Arguments are
forwarded to the browser as real-time filter WebSocket messages.
"""
from __future__ import annotations

import asyncio
import base64
import difflib
import json
import logging
import os
import queue
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from azure.core.credentials_async import AsyncTokenCredential
from azure.identity.aio import AzureCliCredential
from azure.ai.voicelive.aio import connect
from azure.ai.voicelive.models import (
    AvatarConfig,
    AvatarConfigTypes,
    AudioEchoCancellation,
    AudioInputTranscriptionOptions,
    AudioNoiseReduction,
    AzureStandardVoice,
    FunctionCallOutputItem,
    FunctionTool,
    InputAudioFormat,
    Modality,
    OutputAudioFormat,
    RequestSession,
    ServerEventType,
    ServerVad,
)
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import pyaudio
import uvicorn

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv("./.env", override=True)

# ── Logging ──────────────────────────────────────────────────────────────────

if not os.path.exists("logs"):
    os.makedirs("logs")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(
    handlers=[
        logging.FileHandler(f"logs/{timestamp}_server.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    format="%(asctime)s:%(name)s:%(levelname)s:%(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

ENDPOINT = os.environ.get("AZURE_VOICELIVE_ENDPOINT", "")
MODEL = os.environ.get("AZURE_VOICELIVE_MODEL", "gpt-5-mini")
VOICE = os.environ.get("AZURE_VOICELIVE_VOICE", "en-US-Ava:DragonHDLatestNeural")
AVATAR_CHARACTER = os.environ.get("AZURE_AVATAR_CHARACTER", "layla")
AVATAR_MODEL = os.environ.get("AZURE_AVATAR_MODEL", "vasa-1")
VAD_THRESHOLD = float(os.environ.get("VOICE_VAD_THRESHOLD", "0.74"))
VAD_PREFIX_PADDING_MS = int(os.environ.get("VOICE_VAD_PREFIX_PADDING_MS", "220"))
VAD_SILENCE_MS = int(os.environ.get("VOICE_VAD_SILENCE_MS", "900"))
MIC_DUCK_SECONDS = float(os.environ.get("VOICE_MIC_DUCK_SECONDS", "0.45"))
# Minimum seconds of mic silence to add after a response finishes, on top of the estimated speech duration
MIC_DUCK_POST_RESPONSE = float(os.environ.get("VOICE_MIC_DUCK_POST_RESPONSE", "1.5"))
MIC_DUCK_GRACE_SECONDS = float(os.environ.get("VOICE_MIC_DUCK_GRACE_SECONDS", "0.3"))
MIC_DUCK_MIN_TAIL_SECONDS = float(os.environ.get("VOICE_MIC_DUCK_MIN_TAIL_SECONDS", "1.8"))
MIC_DUCK_MAX_TAIL_SECONDS = float(os.environ.get("VOICE_MIC_DUCK_MAX_TAIL_SECONDS", "4.0"))
ECHO_SUPPRESS_WINDOW_SECONDS = float(os.environ.get("VOICE_ECHO_SUPPRESS_WINDOW_SECONDS", "8.0"))
ECHO_SUPPRESS_SIMILARITY = float(os.environ.get("VOICE_ECHO_SUPPRESS_SIMILARITY", "0.8"))
ENABLE_LOCAL_PLAYBACK = os.environ.get("VOICE_ENABLE_LOCAL_PLAYBACK", "false").lower() in {
    "1", "true", "yes", "on"
}
ALLOWED_ORIGINS = {
    "http://localhost:5173",
    "http://localhost:4173",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:4173",
}
GENERIC_CERT_QUERY_TERMS = {
    "first certification",
    "no certification",
    "no certifications",
    "no certifications yet",
    "beginner certification",
    "starter certification",
    "entry certification",
}

SYSTEM_INSTRUCTIONS = """You are a friendly, proactive voice mentor for Course Atlas — an online course catalog.

Goal:
Help users discover the best learning path, not just run one-shot search commands.

Conversation style:
- Be warm, clear, and concise.
- Ask one focused follow-up question at a time.
- Keep discovery responses short and natural.
- Stay strictly on course discovery and learning guidance.

Required flow (follow this order exactly unless the user already gave the answer):
1) Ask for category only: AI, Cloud, Data, DevOps, Security, or Development.
2) Ask for level only: Beginner, Intermediate, or Advanced.
3) Ask for training type only: Certification Path, Self-paced, or Instructor-led.
4) After you know category + level + training type, call set_filters and show the matching grounded courses.
5) Read out up to 3 matching courses briefly and ask the user to pick one.
6) If the user names one course or describes one clearly enough to identify it, call set_filters again with query terms that narrow to that course.
7) After one course remains, say that the course is selected on screen and explain why it fits.
8) Then offer next-step help such as creating a study plan.

Suggestion behavior:
- After understanding category + level + training type, suggest up to 3 good options briefly.
- Mention why each option fits.
- Invite the user to pick one or refine further.

Catalog grounding rules (strict):
- You must only recommend courses that exist in the provided function output for set_filters.
- Do not invent course names, providers, levels, certifications, durations, or links.
- If there are zero matches, clearly say no exact match was found and ask a narrowing question.
- When listing options, quote the exact course titles from function output.
- If exactly one grounded course remains, treat it as selected and tell the user it is selected on screen.

Filter behavior (critical):
- Use set_filters whenever user intent implies a useful filter/search update.
- Prefer broad filters first, then narrow as user gives more detail.
- If user is unsure, set broader filters and ask a clarifying question.
- Every time the user adds or changes a preference, call set_filters again before answering.
- Treat these as filter-changing signals: category/domain, level, format/training type, provider, course title mention, certification intent, exam code, and keyword/topic refinement.
- If the user mentions a specific course title, keep the current category/level when known and set query to the most identifying title words.
- If the user mentions certification intent for a selected course, call set_filters again and use query to preserve the course/certification context.
- Do not continue with generic spoken advice after the user narrows the request unless you have refreshed grounding with set_filters for that turn.
- A plain certification-history answer like "no", "not yet", or "this is my first certification" should NOT add a query by itself.
- Only use query for certification when the user names a real certification, exam code, or specific course/certification title.
- Map "self-paced training" to format "Self-paced".
- Map "certification course" to format "Certification Path".
- Map "instructor-led training" to format "Instructor-led".

Available filter values:
    category: "All", "AI", "Cloud", "Data", "DevOps", "Security", "Development"
    level: "All", "Beginner", "Intermediate", "Advanced"
    format: "All", "Self-paced", "Instructor-led", "Certification Path"
    provider: "" (any), "Microsoft Learn", "AWS Skill Builder", "Google Cloud Skills Boost",
                        "Coursera", "Udemy", "Pluralsight", "Frontend Masters"
    query: free-text keyword (empty string = no keyword filter)

Mapping rules:
- Use "All" for category/level when unknown.
- Use "All" for format when unknown.
- Use empty string "" for provider/query when unknown.
- Do not map generic certification-history phrases into query.
- Use query for explicit certification identifiers only (example: "AZ-900", "SC-900", or a full certification/course title).
- Map mentions like "identity fundamentals", "security fundamentals", or exact course names into query terms, not just spoken text.

Session start requirement:
- Use an interactive first line such as: "Hello there, I am the Course Atlas assistant. I can help you find the right course for you."
- In the first turn, ask only: "Which field are you looking for: AI, Cloud, Data, DevOps, Security, or Development?"
- After user answers category, ask only level in the next turn.
- After user answers level, ask only training type in the next turn: certification course, self-paced training, or instructor-led.
- Do not ask provider, certification history, or extra questions before those first 3 steps are complete.
- Immediately apply a broad starter filter with set_filters (All/All/All/""/"") so the catalog is populated.
"""


def _extract_str(block: str, field: str) -> str:
    match = re.search(rf"{field}:\s*'([^']*)'", block)
    return match.group(1) if match else ""


def _extract_tags(block: str) -> list[str]:
    tags_match = re.search(r"tags:\s*\[([^\]]*)\]", block, flags=re.DOTALL)
    if not tags_match:
        return []
    return re.findall(r"'([^']*)'", tags_match.group(1))


def _load_catalog_from_ts() -> list[dict]:
    # Reuse frontend catalog as the single source of truth for grounded answers.
    ts_path = Path(__file__).resolve().parent.parent / "src" / "data" / "courses.ts"
    try:
        content = ts_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not load catalog file: %s", exc)
        return []

    pattern = re.compile(r"\{\s*id:\s*'[^']+'(.*?)\n\s*\}\s*,?", flags=re.DOTALL)
    items: list[dict] = []
    for block in pattern.findall(content):
        title = _extract_str(block, "title")
        if not title:
            continue
        items.append(
            {
                "title": title,
                "provider": _extract_str(block, "provider"),
                "category": _extract_str(block, "category"),
                "level": _extract_str(block, "level"),
                "duration": _extract_str(block, "duration"),
                "format": _extract_str(block, "format"),
                "summary": _extract_str(block, "summary"),
                "audience": _extract_str(block, "audience"),
                "tags": _extract_tags(block),
            }
        )
    return items


CATALOG = _load_catalog_from_ts()

# ── Tool definition ───────────────────────────────────────────────────────────

SET_FILTERS_TOOL = FunctionTool(
    name="set_filters",
    description="Update the course catalog filters so the user sees matching courses in real-time.",
    parameters={
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["All", "AI", "Cloud", "Data", "DevOps", "Security", "Development"],
                "description": "Course category, or 'All'.",
            },
            "level": {
                "type": "string",
                "enum": ["All", "Beginner", "Intermediate", "Advanced"],
                "description": "Experience level, or 'All'.",
            },
            "format": {
                "type": "string",
                "enum": ["All", "Self-paced", "Instructor-led", "Certification Path"],
                "description": "Training type / course format, or 'All'.",
            },
            "provider": {
                "type": "string",
                "description": "Training provider name, or empty string for any.",
            },
            "query": {
                "type": "string",
                "description": "Free-text keyword, or empty string.",
            },
        },
        "required": ["category", "level", "format", "provider", "query"],
    },
)

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Course Atlas Voice Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def voice_endpoint(websocket: WebSocket):
    origin = websocket.headers.get("origin", "")
    if origin and origin not in ALLOWED_ORIGINS:
        await websocket.close(code=1008, reason="Origin not allowed")
        return

    await websocket.accept()
    logger.info("Frontend connected from '%s'", origin)

    session = VoiceSession(websocket)
    try:
        await session.run()
    except WebSocketDisconnect:
        logger.info("Frontend disconnected")
    except Exception as exc:
        logger.exception("Session error: %s", exc)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
    finally:
        session.cleanup()


# ── Voice session ─────────────────────────────────────────────────────────────

class VoiceSession:
    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.connection = None
        self.ap: Optional[AudioProc] = None
        self._running = True
        self._active_response = False
        self._response_api_done = False
        self._audio_transcript_acc = ""
        self._last_spoken_chars = 0
        self._last_assistant_text = ""
        self._last_assistant_spoke_at = 0.0
        self._pending_barge_in = False
        self._avatar_answer_event = asyncio.Event()
        self._avatar_server_sdp: Optional[str] = None

    @staticmethod
    def _norm_text(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()

    def _is_probable_echo(self, transcript: str) -> bool:
        if not transcript or not self._last_assistant_text:
            return False
        if (time.monotonic() - self._last_assistant_spoke_at) > ECHO_SUPPRESS_WINDOW_SECONDS:
            return False

        user_norm = self._norm_text(transcript)
        ai_norm = self._norm_text(self._last_assistant_text)
        if not user_norm or not ai_norm:
            return False

        # Fast path: direct containment catches phrase repeats like "Let's get you started".
        if len(user_norm) >= 8 and user_norm in ai_norm:
            return True

        ratio = difflib.SequenceMatcher(None, user_norm, ai_norm).ratio()
        return ratio >= ECHO_SUPPRESS_SIMILARITY

    def _filter_catalog(self, args: dict) -> list[dict]:
        category = (args.get("category") or "All").strip()
        level = (args.get("level") or "All").strip()
        course_format = (args.get("format") or "All").strip()
        provider = (args.get("provider") or "").strip()
        query = (args.get("query") or "").strip().lower()

        matches: list[dict] = []
        for course in CATALOG:
            if category != "All" and course.get("category") != category:
                continue
            if level != "All" and course.get("level") != level:
                continue
            if course_format != "All" and course.get("format") != course_format:
                continue
            if provider and course.get("provider") != provider:
                continue

            if query:
                haystack = " ".join(
                    [
                        course.get("title", ""),
                        course.get("provider", ""),
                        course.get("summary", ""),
                        course.get("audience", ""),
                        " ".join(course.get("tags", [])),
                    ]
                ).lower()
                if query not in haystack:
                    continue

            matches.append(
                {
                    "title": course.get("title", ""),
                    "provider": course.get("provider", ""),
                    "category": course.get("category", ""),
                    "level": course.get("level", ""),
                    "duration": course.get("duration", ""),
                    "format": course.get("format", ""),
                    "summary": course.get("summary", ""),
                }
            )
        return matches

    @staticmethod
    def _normalize_filter_args(args: dict) -> dict:
        normalized = dict(args)
        if not normalized.get("format"):
            normalized["format"] = "All"
        query = (normalized.get("query") or "").strip().lower()
        if query in GENERIC_CERT_QUERY_TERMS:
            normalized["query"] = ""
        return normalized

    async def _send(self, msg: dict) -> None:
        try:
            await self.ws.send_text(json.dumps(msg))
        except Exception:
            pass

    async def run(self) -> None:
        await self._send({"type": "status", "value": "connecting"})
        credential = AzureCliCredential()

        async with connect(endpoint=ENDPOINT, credential=credential, model=MODEL) as conn:
            self.connection = conn
            self.ap = AudioProc(conn)

            await self._configure_session()
            if ENABLE_LOCAL_PLAYBACK:
                self.ap.start_playback()
                logger.info("Local playback enabled on backend")
            else:
                logger.info("Local playback disabled on backend (avatar/browser audio only)")
            await self._send({"type": "connected"})

            frontend_task = asyncio.create_task(self._listen_frontend())
            try:
                async for event in conn:
                    if not self._running:
                        break
                    await self._handle(event)
            finally:
                frontend_task.cancel()

    async def _listen_frontend(self) -> None:
        try:
            while self._running:
                raw = await self.ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "stop":
                    self._running = False
                    break
                if msg.get("type") == "avatar_offer":
                    await self._send({"type": "avatar_connecting"})
                    client_sdp = msg.get("sdp", "")
                    logger.info("Received avatar_offer from frontend")
                    server_sdp = await self._send_avatar_connect(client_sdp)
                    if server_sdp:
                        logger.info("Sending avatar_answer to frontend")
                        await self._send({"type": "avatar_answer", "sdp": server_sdp})
                    else:
                        logger.warning("No avatar SDP answer received from VoiceLive")
                        await self._send({
                            "type": "avatar_error",
                            "message": "Could not establish avatar connection for this endpoint/avatar.",
                        })
        except Exception:
            pass

    async def _send_avatar_connect(self, client_sdp: str) -> Optional[str]:
        conn = self.connection
        if conn is None:
            return None

        self._avatar_server_sdp = None
        self._avatar_answer_event.clear()

        try:
            await conn.send({"type": "session.avatar.connect", "client_sdp": client_sdp})
            await asyncio.wait_for(self._avatar_answer_event.wait(), timeout=30.0)
            return self._avatar_server_sdp
        except Exception as exc:
            logger.warning("Avatar connect failed: %s", exc)
            return None

    async def _configure_session(self) -> None:
        voice: Union[AzureStandardVoice, str] = (
            AzureStandardVoice(name=VOICE) if "-" in VOICE else VOICE
        )
        
        assert self.connection is not None
        await self.connection.session.update(
            session=RequestSession(
                modalities=[Modality.TEXT, Modality.AUDIO, Modality.AVATAR],
                instructions=SYSTEM_INSTRUCTIONS,
                voice=voice,
                input_audio_format=InputAudioFormat.PCM16,
                output_audio_format=OutputAudioFormat.PCM16,
                turn_detection=ServerVad(
                    threshold=VAD_THRESHOLD,
                    prefix_padding_ms=VAD_PREFIX_PADDING_MS,
                    silence_duration_ms=VAD_SILENCE_MS,
                ),
                input_audio_echo_cancellation=AudioEchoCancellation(),
                input_audio_noise_reduction=AudioNoiseReduction(
                    type="azure_deep_noise_suppression"
                ),
                input_audio_transcription=AudioInputTranscriptionOptions(model="whisper-1"),
                avatar=AvatarConfig(
                    type=AvatarConfigTypes.PHOTO_AVATAR,
                    character=AVATAR_CHARACTER,
                    model=AVATAR_MODEL,
                ),
                tools=[SET_FILTERS_TOOL],
            )
        )
        logger.info(
            "Session configured — set_filters tool registered, photo_avatar=%s (model=%s)",
            AVATAR_CHARACTER,
            AVATAR_MODEL,
        )

    async def _handle(self, event) -> None:
        ap = self.ap
        conn = self.connection
        assert ap is not None and conn is not None

        etype = event.type
        logger.info("EVENT: %s", etype)

        if etype == ServerEventType.SESSION_UPDATED:
            session_avatar = getattr(getattr(event, "session", None), "avatar", None)
            ice_servers = []
            if session_avatar and getattr(session_avatar, "ice_servers", None):
                for ice in session_avatar.ice_servers:
                    ice_servers.append(
                        {
                            "urls": getattr(ice, "urls", []),
                            "username": getattr(ice, "username", ""),
                            "credential": getattr(ice, "credential", ""),
                        }
                    )
            await self._send({"type": "avatar_ice_servers", "servers": ice_servers})
            logger.info("Avatar ICE servers sent to frontend: %d", len(ice_servers))

            ap.start_capture()
            await self._send({"type": "status", "value": "ready"})

        elif etype == ServerEventType.SESSION_AVATAR_CONNECTING or "avatar" in str(etype).lower():
            server_sdp = None
            if hasattr(event, "as_dict"):
                event_dict = event.as_dict()
                server_sdp = event_dict.get("server_sdp")
            if not server_sdp:
                server_sdp = getattr(event, "server_sdp", None)
            if server_sdp:
                self._avatar_server_sdp = server_sdp
                self._avatar_answer_event.set()
                await self._send({"type": "avatar_ready"})

        elif etype == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
            if ap.is_capture_ducked(grace_seconds=MIC_DUCK_GRACE_SECONDS) and not (
                self._active_response and not self._response_api_done
            ):
                logger.info("Ignoring speech_started during mic duck window (likely echo)")
                return
            ap.skip_pending_audio()
            await self._send({"type": "status", "value": "listening"})
            if self._active_response and not self._response_api_done:
                # Defer cancellation until we receive a non-echo user transcription.
                # This prevents false cancels from speaker leakage.
                self._pending_barge_in = True

        elif etype == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED:
            await self._send({"type": "status", "value": "processing"})

        elif etype == ServerEventType.RESPONSE_CREATED:
            self._active_response = True
            self._response_api_done = False
            self._pending_barge_in = False
            self._audio_transcript_acc = ""
            ap.duck_capture(max(MIC_DUCK_SECONDS, 1.0))
            await self._send({"type": "status", "value": "speaking"})

        elif etype == ServerEventType.RESPONSE_AUDIO_DELTA:
            # Keep extending capture ducking during assistant playback to avoid feedback loops.
            ap.duck_capture(0.6)
            if ENABLE_LOCAL_PLAYBACK:
                ap.queue_audio(event.delta)

        # ── AI spoken text — accumulate deltas, send on done ──────────────
        elif etype == ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DELTA:
            # Avatar mode may not emit continuous audio.delta events; keep ducking active from transcript stream.
            # Keep short rolling duck windows; long windows block barge-in.
            ap.duck_capture(1.0)
            self._audio_transcript_acc += getattr(event, "delta", "") or ""

        elif etype == ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DONE:
            text = getattr(event, "transcript", None) or self._audio_transcript_acc
            self._last_spoken_chars = len(text)  # save before clearing for RESPONSE_DONE
            self._last_assistant_text = text or ""
            self._last_assistant_spoke_at = time.monotonic()
            self._audio_transcript_acc = ""
            # Apply tail duck: audio.done fires here but browser still has buffered audio to play.
            ap.duck_capture(MIC_DUCK_POST_RESPONSE + 0.8)
            if text:
                logger.info("AI spoke: %.120s", text)
                await self._send({"type": "transcript", "role": "assistant", "text": text})

        # ── Function call — set_filters ───────────────────────────────────
        elif etype == ServerEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE:
            func_name = getattr(event, "name", "")
            call_id = getattr(event, "call_id", "")
            raw_args = getattr(event, "arguments", "{}") or "{}"
            logger.info("Function call '%s' args: %s", func_name, raw_args)

            if func_name == "set_filters":
                try:
                    args = json.loads(raw_args)
                    args = self._normalize_filter_args(args)
                    await self._send({"type": "filter", **args})
                    logger.info("Filter sent to browser: %s", args)
                except json.JSONDecodeError as exc:
                    logger.warning("Bad JSON in function args: %s — %s", raw_args, exc)
                    args = {
                        "category": "All",
                        "level": "All",
                        "format": "All",
                        "provider": "",
                        "query": "",
                    }

                grounded_matches = self._filter_catalog(args)
                if not grounded_matches and (args.get("query") or "").strip().lower() in GENERIC_CERT_QUERY_TERMS:
                    args["query"] = ""
                    grounded_matches = self._filter_catalog(args)
                logger.info("Grounded course matches: %d", len(grounded_matches))

                # Tell the AI the function completed so it continues speaking
                try:
                    await conn.conversation.item.create(
                        item=FunctionCallOutputItem(
                            call_id=call_id,
                            output=json.dumps(
                                {
                                    "status": "ok",
                                    "filters": args,
                                    "total_matches": len(grounded_matches),
                                    "available_courses": grounded_matches[:12],
                                    "note": "Recommend only courses from available_courses.",
                                }
                            ),
                        )
                    )
                    # Explicitly ask the model to continue after tool output.
                    # Without this, the response may end right after function execution.
                    await conn.response.create()
                except Exception as exc:
                    logger.warning("Could not return function result: %s", exc)

        # ── User speech transcript ────────────────────────────────────────
        elif etype == ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED:
            transcript = getattr(event, "transcript", "")
            if transcript:
                if self._is_probable_echo(transcript):
                    logger.info("Suppressed probable echo transcript: %.120s", transcript)
                    self._pending_barge_in = False
                    return
                if self._pending_barge_in and self._active_response and not self._response_api_done:
                    try:
                        await conn.response.cancel()
                        logger.info("Cancelled active response on confirmed barge-in transcript")
                    except Exception:
                        pass
                    self._pending_barge_in = False
                logger.info("User said: %.120s", transcript)
                await self._send({"type": "transcript", "role": "user", "text": transcript})

        elif etype == ServerEventType.RESPONSE_DONE:
            self._active_response = False
            self._response_api_done = True
            self._pending_barge_in = False
            # Keep post-response duck bounded so interruptions remain possible quickly.
            spoken_chars = self._last_spoken_chars
            tail_guard = min(
                max(MIC_DUCK_POST_RESPONSE + 0.8, MIC_DUCK_MIN_TAIL_SECONDS),
                MIC_DUCK_MAX_TAIL_SECONDS,
            )
            ap.duck_capture(tail_guard)
            logger.info("Post-response mic duck: %.1fs (transcript ~%d chars)", tail_guard, spoken_chars)
            await self._send({"type": "status", "value": "ready"})

        elif etype == ServerEventType.RESPONSE_ANIMATION_BLENDSHAPES_DELTA:
            # Avatar blendshape animation data — silently handled by the connection
            pass

        elif etype == ServerEventType.RESPONSE_ANIMATION_VISEME_DELTA:
            # Avatar viseme (mouth shape) animation data — silently handled
            pass

        elif etype == ServerEventType.ERROR:
            msg = getattr(event.error, "message", str(event.error))
            if "no active response" not in msg.lower():
                logger.error("VoiceLive error: %s", msg)
                await self._send({"type": "error", "message": msg})

    def cleanup(self) -> None:
        self._running = False
        if self.ap:
            self.ap.shutdown()


# ── Audio processor ───────────────────────────────────────────────────────────

class AudioProc:
    loop: asyncio.AbstractEventLoop

    class _Pkt:
        __slots__ = ("seq_num", "data")

        def __init__(self, seq: int, data) -> None:
            self.seq_num = seq
            self.data = data

    def __init__(self, conn) -> None:
        self.conn = conn
        self.pa = pyaudio.PyAudio()
        self.fmt = pyaudio.paInt16
        self.ch = 1
        self.rate = 24000
        self.chunk = 1200
        self.input_stream = None
        self.output_stream = None
        self.pb_queue: queue.Queue = queue.Queue()
        self.pb_base = 0
        self._seq = 0
        self._duck_capture_until = 0.0

    def duck_capture(self, seconds: float) -> None:
        self._duck_capture_until = max(self._duck_capture_until, time.monotonic() + max(seconds, 0.0))

    def is_capture_ducked(self, grace_seconds: float = 0.0) -> bool:
        return time.monotonic() < (self._duck_capture_until + max(grace_seconds, 0.0))

    def start_capture(self) -> None:
        if self.input_stream:
            return
        self.loop = asyncio.get_event_loop()

        def _cb(in_data, _fc, _ti, _sf):
            if time.monotonic() < self._duck_capture_until:
                return (None, pyaudio.paContinue)
            b64 = base64.b64encode(in_data).decode("utf-8")
            asyncio.run_coroutine_threadsafe(
                self.conn.input_audio_buffer.append(audio=b64), self.loop
            )
            return (None, pyaudio.paContinue)

        self.input_stream = self.pa.open(
            format=self.fmt, channels=self.ch, rate=self.rate,
            input=True, frames_per_buffer=self.chunk, stream_callback=_cb,
        )

    def start_playback(self) -> None:
        if self.output_stream:
            return
        remaining = bytes()

        def _cb(_in, frame_count, _ti, _sf):
            nonlocal remaining
            need = frame_count * pyaudio.get_sample_size(pyaudio.paInt16)
            out = remaining[:need]
            remaining = remaining[need:]
            while len(out) < need:
                try:
                    pkt = self.pb_queue.get_nowait()
                except queue.Empty:
                    out += bytes(need - len(out))
                    continue
                if not pkt or not pkt.data:
                    break
                if pkt.seq_num < self.pb_base:
                    remaining = bytes()
                    continue
                take = need - len(out)
                out += pkt.data[:take]
                remaining = pkt.data[take:]
            return (out, pyaudio.paContinue if len(out) >= need else pyaudio.paComplete)

        self.output_stream = self.pa.open(
            format=self.fmt, channels=self.ch, rate=self.rate,
            output=True, frames_per_buffer=self.chunk, stream_callback=_cb,
        )

    def _next_seq(self) -> int:
        s = self._seq
        self._seq += 1
        return s

    def queue_audio(self, data) -> None:
        self.pb_queue.put(self._Pkt(self._next_seq(), data))

    def skip_pending_audio(self) -> None:
        self.pb_base = self._next_seq()

    def shutdown(self) -> None:
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None
        if self.output_stream:
            self.skip_pending_audio()
            self.queue_audio(None)
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None
        if self.pa:
            self.pa.terminate()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Course Atlas Voice Server")
    print("=" * 40)
    print("WebSocket : ws://localhost:8765/ws")
    print("Health    : http://localhost:8765/health")
    print("Logs      : voice/logs/")
    print("Press Ctrl+C to stop")
    print()
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
