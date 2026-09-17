import os
import json
import queue
import threading
import asyncio
import base64
import time as _time
from typing import Iterator, AsyncIterator

_SARVAM_KEY = os.getenv("SARVAM_API_KEY")
_SARVAM_STT_WS = "wss://api.sarvam.ai/speech-to-text/ws"
_SARVAM_TTS_WS = "wss://api.sarvam.ai/text-to-speech/ws"
DEFAULT_LANG = "en-US"

try:
    import websockets
    _HAS_WS = True
except ImportError:
    _HAS_WS = False

try:
    import sounddevice as sd
    import numpy as np
    _HAS_SOUND = True
except ImportError:
    _HAS_SOUND = False

try:
    import pyttsx3
    _HAS_TTS_LOCAL = True
except ImportError:
    _HAS_TTS_LOCAL = False

try:
    import speech_recognition as sr
    _HAS_STT_LOCAL = True
except ImportError:
    _HAS_STT_LOCAL = False


class SarvamVoice:
    """Full-duplex voice I/O using Sarvam AI.

    STT: Saaras v2 — real-time Hindi/Indian language ASR, <250ms latency,
         barge-in support, speaker diarization-ready.
    TTS: Bulbul — multilingual expressive voices, ~400ms streaming,
        emotion + speed control, barge-in support.

    WebSocket protocol:
      - Auth: header 'api-subscription-key: <key>'
      - STT: send PCM 16-bit mono @16kHz interleaved; receive JSON with
        { " transcript": "...", " is_final": bool, ... }
      - TTS: send {"text": "...", "language_code": "en-US", "voice": "meera"};
        receive base64 PCM audio chunks.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or _SARVAM_KEY
        self.lang = os.getenv("SARVAM_LANG", DEFAULT_LANG)
        self._interrupt_flag = threading.Event()
        self._tts_thread: threading.Thread | None = None
        self._last_transcript = queue.Queue(maxsize=1)

    @property
    def available(self) -> bool:
        return self.api_key is not None and _HAS_WS and _HAS_SOUND

    # --- STT (Saaras) ---

    def listen(self, timeout: float = 10.0, phrase_limit: float = 8.0) -> str:
        """Capture speech via mic and transcribe with Saaras STT.

        Supports barge-in: if TTS is playing, it is interrupted first.
        Falls back to local speech_recognition or keyboard if Sarvam unavailable.
        """
        self._interrupt_flag.set()  # Signal TTS to stop (barge-in)
        if self._tts_thread is not None and self._tts_thread.is_alive():
            self._tts_thread.join(timeout=0.5)
        self._interrupt_flag.clear()
        self._tts_thread = None
        if self.available:
            return self._listen_sarvam(timeout, phrase_limit)
        elif _HAS_STT_LOCAL:
            return self._listen_local()
        else:
            return self._listen_keyboard()

    def _listen_sarvam(self, timeout: float, phrase_limit: float) -> str:
        """WebSocket streaming STT with Saaras."""
        transcript = ""

        async def _run():
            headers = {"api-subscription-key": self.api_key}
            async with websockets.connect(_SARVAM_STT_WS, extra_headers=headers) as ws:
                await ws.send(json.dumps({"config": {"model": "saaras:v2.1", "language_code": self.lang}}))

                stop_rec = threading.Event()

                async def send_audio():
                    if not _HAS_SOUND:
                        return
                    sample_rate = 16000
                    chunk_size = int(sample_rate * 0.1)
                    for _ in range(int(phrase_limit * 10)):
                        if self._interrupt_flag.is_set() or stop_rec.is_set():
                            break
                        chunk = sd.rec(chunk_size, samplerate=sample_rate, channels=1, dtype="int16")
                        sd.wait()
                        try:
                            await ws.send(bytes(chunk.tobytes()))
                        except Exception:
                            break

                async def receive():
                    nonlocal transcript
                    while True:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                            data = json.loads(msg)
                            if data.get("is_final", False):
                                transcript = data.get("transcript", "")
                                if self._last_transcript.full():
                                    self._last_transcript.get_nowait()
                                self._last_transcript.put_nowait(transcript)
                                stop_rec.set()
                                break
                        except (asyncio.TimeoutError, websockets.ConnectionClosed):
                            break

                await asyncio.gather(send_audio(), receive())

        try:
            asyncio.run(_run())
        except Exception as e:
            print(f"[STT] Sarvam error: {e}")

        result = transcript.strip()
        if result:
            return result
        elif _HAS_STT_LOCAL:
            return self._listen_local()
        else:
            return self._listen_keyboard()

    def _listen_local(self) -> str:
        """Local fallback using speech_recognition + Google Web API."""
        if not _HAS_STT_LOCAL:
            return ""
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=8)
                return r.recognize_google(audio)
            except Exception:
                return self._listen_keyboard()

    def _listen_keyboard(self) -> str:
        """Keyboard fallback — type your reply and press Enter."""
        try:
            return input("[keyboard] Reply (Enter to submit): ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    # --- TTS (Bulbul) ---

    def speak(self, text: str, interrupt: bool = True):
        """Block until speech completes. Interrupt if interrupt=True."""
        if interrupt:
            self.interrupt()
        if self.available and text.strip():
            self._speak_sarvam(text)
        elif _HAS_TTS_LOCAL:
            self._speak_local(text)
        else:
            print(f"[TTS] {text}")

    def speak_async(self, text: str) -> threading.Thread:
        """Non-blocking speech output on a daemon thread."""
        self._interrupt_flag.clear()
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        self._tts_thread = t
        t.start()
        return t

    def interrupt(self):
        """Signal ongoing STT/TTS to cut off (barge-in)."""
        self._interrupt_flag.set()

    def _speak_sarvam(self, text: str, voice: str = "meera", speed: float = 1.0):
        """WebSocket streaming TTS with Bulbul.

        Plays audio in chunks for interruptible barge-in support.
        """
        import time as _time
        async def _run():
            headers = {"api-subscription-key": self.api_key}
            async with websockets.connect(_SARVAM_TTS_WS, extra_headers=headers) as ws:
                msg = json.dumps({
                    "text": text,
                    "language_code": self.lang,
                    "voice": voice,
                    "speed": speed,
                    "sample_rate": 16000,
                })
                await ws.send(msg)
                all_audio = b""
                while True:
                    try:
                        chunk = await asyncio.wait_for(ws.recv(), timeout=3.0)
                        if isinstance(chunk, str):
                            data = json.loads(chunk)
                            if data.get("type") == "audio":
                                audio_bytes = base64.b64decode(data["audio"])
                                all_audio += audio_bytes
                        elif isinstance(chunk, bytes):
                            all_audio += chunk
                    except (asyncio.TimeoutError, websockets.ConnectionClosed):
                        break

                if all_audio and _HAS_SOUND:
                    # Play in chunks — check interrupt between each chunk
                    audio = np.frombuffer(all_audio, dtype=np.int16)
                    chunk_size = 1600  # ~100ms at 16kHz
                    offset = 0
                    sd.play(audio, 16000)
                    while offset < len(audio):
                        if self._interrupt_flag.is_set():
                            sd.stop()
                            break
                        _time.sleep(0.05)
                        offset += chunk_size

        try:
            asyncio.run(_run())
        except Exception as e:
            print(f"[TTS] Sarvam error: {e}")
            self._speak_local(text)

    def _speak_local(self, text: str):
        """Local fallback using pyttsx3."""
        if _HAS_TTS_LOCAL:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        else:
            print(f"[TTS] {text}")

    # --- STT Token Streaming (for true full-duplex) ---

    def stream_transcript(self) -> AsyncIterator[str]:
        """Yield partial transcripts as they arrive (for live captioning).

        Must be called after listen() starts to receive partial results.
        """
        while True:
            try:
                transcript = self._last_transcript.get_nowait()
                yield transcript
            except queue.Empty:
                yield ""
