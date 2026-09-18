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

# pyttsx3 engines run a COM/loop per runAndWait() call; calling runAndWait()
# from two threads at once raises "RuntimeError: run loop already started".
# One shared engine behind a lock serializes all local-fallback playback.
_LOCAL_TTS_LOCK = threading.Lock()
_LOCAL_TTS_ENGINE = None

# Languages Sarvam TTS accepts as target_language_code (pipecat-verified map).
_SARVAM_TTS_LANGS = {"en-IN", "hi-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN",
                     "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN"}

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
      - TTS: Bulbul v3 protocol (verified against api.sarvam.ai 2026-09):
          1. send {"type":"config","data":{...}}  voice/model/audio format
          2. send {"type":"text","data":{"text": ...}}
          3. send {"type":"flush"}
          4. receive {"type":"audio","data":{"audio":"<b64 PCM16>"}} chunks
             until {"type":"event","data":{"event_type":"final"}}
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or _SARVAM_KEY
        self.lang = os.getenv("SARVAM_LANG", DEFAULT_LANG)
        self._interrupt_flag = threading.Event()
        self._tts_gen = 0  # generation counter — bumped on interrupt/new utterance
        self._tts_thread: threading.Thread | None = None
        self._last_transcript = queue.Queue(maxsize=1)
        # Sentence-level speech queue (talkative upgrade, Phase 0)
        self._speech_queue: queue.Queue | None = None
        self._speech_thread: threading.Thread | None = None
        self._speech_stop = threading.Event()
        # Echo guard hook: called with True/False while TTS audio plays so
        # MicVAD can suppress ARIA's own voice (pseudo-duplex).
        self._on_speaking = None
        # UI hooks (web dashboard): on_say fires for every spoken utterance,
        # on_heard for every transcribed user turn.
        self._on_say = None
        self._on_heard = None
        # Set while the SpeechQueue worker is actively playing an utterance
        self._speech_busy = threading.Event()

    def available(self) -> bool:
        return self.api_key is not None and _HAS_WS

    # --- STT (Saaras) ---

    def listen(self, timeout: float = 10.0, phrase_limit: float = 8.0, mic=None) -> str:
        """Capture speech and transcribe it.

        With `mic` (a MicVAD), uses energy-VAD end-of-turn detection: capture
        starts at speech onset and ends after ~600ms of quiet — no fixed
        8-second windows. ARIA speaking does NOT interrupt the mic here; the
        mic suppresses its own echo instead (pseudo-duplex, laptop speakers).
        Without a mic, legacy half-duplex semantics apply: in-flight TTS is
        interrupted first, then a fixed capture window is recorded.
        """
        if mic is not None:
            result = self._listen_vad(mic, timeout, phrase_limit)
        else:
            self.interrupt()  # Barge-in: bump generation so in-flight TTS aborts
            if self._tts_thread is not None and self._tts_thread.is_alive():
                self._tts_thread.join(timeout=0.5)
            self._interrupt_flag.clear()
            self._tts_thread = None
            if self.available():
                result = self._listen_sarvam(timeout, phrase_limit)
            elif _HAS_STT_LOCAL:
                result = self._listen_local()
            else:
                result = self._listen_keyboard()
        if result and self._on_heard is not None:
            try:
                self._on_heard(result)
            except Exception:
                pass
        return result

    def _listen_sarvam(self, timeout: float, phrase_limit: float) -> str:
        """WebSocket streaming STT with Saaras."""
        transcript = ""

        async def _run():
            headers = {"api-subscription-key": self.api_key}
            connect_kwargs = {"additional_headers": headers}
            if websockets.__version__ < "13.0":
                connect_kwargs = {"extra_headers": headers}
            async with websockets.connect(_SARVAM_STT_WS, **connect_kwargs) as ws:
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

    def _listen_vad(self, mic, timeout: float, phrase_limit: float) -> str:
        """VAD-gated listening: record only actual speech, end on quiet.

        Echo safety (laptop mic + speakers): MicVAD knows when ARIA is talking
        and ignores that audio entirely, so ARIA never transcribes itself.
        """
        pcm = mic.wait_for_utterance(max_wait=timeout, max_utterance=phrase_limit)
        if self._interrupt_flag.is_set():
            return ""
        if pcm is None or len(pcm) == 0:
            return ""
        if self.available():
            text = self._transcribe_pcm(pcm.tobytes())
            if text:
                return text
        if _HAS_STT_LOCAL:
            try:
                import speech_recognition as sr
                audio = sr.AudioData(pcm.tobytes(), 16000, 2)
                return sr.Recognizer().recognize_google(audio)
            except Exception:
                return ""
        return ""

    def _transcribe_pcm(self, pcm_bytes: bytes) -> str:
        """Transcribe an already-captured PCM16/16k buffer via Saaras STT."""
        transcript = ""

        async def _run():
            nonlocal transcript
            headers = {"api-subscription-key": self.api_key}
            connect_kwargs = {"additional_headers": headers}
            if websockets.__version__ < "13.0":
                connect_kwargs = {"extra_headers": headers}
            async with websockets.connect(_SARVAM_STT_WS, **connect_kwargs) as ws:
                await ws.send(json.dumps({"config": {"model": "saaras:v2.1", "language_code": self.lang}}))
                step = 3200  # 0.1s of 16kHz int16 mono
                for i in range(0, len(pcm_bytes), step):
                    await ws.send(pcm_bytes[i:i + step])
                deadline = _time.time() + 6.0
                while _time.time() < deadline:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    except (asyncio.TimeoutError, websockets.ConnectionClosed):
                        break
                    data = json.loads(msg)
                    if data.get("transcript"):
                        transcript = data["transcript"]
                    if data.get("is_final", False):
                        break

        try:
            asyncio.run(_run())
        except Exception as e:
            print(f"[STT] Sarvam transcribe error: {e}")
        return transcript.strip()

    def _listen_local(self) -> str:
        """Local fallback using sounddevice for audio capture (no pyaudio needed).

        Records audio at 16kHz mono via sounddevice, then transcribes via
        Google Web Speech API through speech_recognition's AudioData.
        """
        if not _HAS_STT_LOCAL or not _HAS_SOUND:
            return self._listen_keyboard()

        try:
            import speech_recognition as sr
        except ImportError:
            return self._listen_keyboard()

        sample_rate = 16000
        duration = 8.0

        frames = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                        channels=1, dtype="int16")
        sd.wait()

        audio = sr.AudioData(frames.tobytes(), sample_rate, 2)
        try:
            return sr.Recognizer().recognize_google(audio)
        except Exception:
            return self._listen_keyboard()

    def _listen_keyboard(self) -> str:
        """Keyboard fallback — type your reply and press Enter."""
        try:
            return input("[keyboard] Reply (Enter to submit): ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    # --- TTS (Bulbul) ---

    def set_speaking_callback(self, cb):
        """Register a callback(bool) fired while TTS audio is playing.

        run.py wires this to MicVAD.speaking so the mic ignores ARIA's own
        voice through the laptop speakers (echo suppression).
        """
        self._on_speaking = cb

    def set_ui_hooks(self, on_say=None, on_heard=None):
        """Register transcript hooks for the web UI.

        on_say(text) fires for every utterance ARIA speaks; on_heard(text)
        for every user turn that survives transcription.
        """
        if on_say is not None:
            self._on_say = on_say
        if on_heard is not None:
            self._on_heard = on_heard

    def speak(self, text: str, interrupt: bool = True):
        """Block until speech completes. Interrupt any older utterance first."""
        cb = self._on_speaking
        if cb:
            cb(True)
        if self._on_say is not None and text and text.strip():
            try:
                self._on_say(text)
            except Exception:
                pass
        try:
            if interrupt:
                self.interrupt()
            my_gen = self._tts_gen  # captured AFTER the interrupt bump
            if self.available() and text.strip():
                self._speak_sarvam(text, my_gen)
            elif _HAS_TTS_LOCAL:
                self._speak_local(text)
            else:
                print(f"[TTS] {text}")
        finally:
            if cb:
                cb(False)

    def speak_async(self, text: str) -> threading.Thread:
        """Non-blocking speech output on a daemon thread.

        Supersedes any in-flight utterance and waits briefly for the previous
        speaker to wind down so two TTS streams never overlap.
        """
        self._tts_gen += 1
        self._interrupt_flag.clear()
        old = self._tts_thread
        if old is not None and old.is_alive():
            old.join(timeout=1.0)
        t = threading.Thread(target=self.speak, args=(text, False), daemon=True)
        self._tts_thread = t
        t.start()
        return t

    def interrupt(self):
        """Signal ongoing STT/TTS to cut off (barge-in).

        Also flushes the sentence speech queue — a barge-in kills not just
        the current sentence but the sentences ARIA was ABOUT to say.
        """
        self._tts_gen += 1  # invalidates any speaker holding an older generation
        self._interrupt_flag.set()
        if self._speech_queue is not None:
            try:
                while True:
                    self._speech_queue.get_nowait()
            except queue.Empty:
                pass

    def _speak_sarvam(self, text: str, my_gen: int | None = None,
                      voice: str | None = None, speed: float = 1.0):
        """WebSocket STREAMING TTS with Bulbul v3.

        Protocol (verified live against api.sarvam.ai with this project's key):
        a config envelope must be sent FIRST, then text + flush; audio arrives
        as {"type":"audio","data":{"audio":<b64>}} chunks. The old code sent
        the REST payload as the first message, so the API answered 422 'Input
        parameters has to be a valid dictionary' and every utterance silently
        fell back to pyttsx3.

        Audio still plays through an OutputStream as chunks ARRIVE (Phase 0
        streaming), and aborts early on generation mismatch (barge-in).
        """
        def _stale() -> bool:
            return my_gen is not None and my_gen != self._tts_gen

        speaker = voice or os.getenv("SARVAM_TTS_SPEAKER", "ritu")
        # self.lang may hold an STT-style code like en-US; TTS needs one of
        # Sarvam's regional target_language_codes.
        lang = self.lang if self.lang in _SARVAM_TTS_LANGS else "en-IN"
        pace = max(0.5, min(2.0, speed))  # bulbul:v3 accepts 0.5-2.0
        stream = None
        played_any = False

        def _sink(audio_bytes: bytes) -> bool:
            nonlocal stream, played_any
            if not audio_bytes or not _HAS_SOUND or _stale():
                return False
            if stream is None:
                stream = sd.OutputStream(samplerate=16000, channels=1, dtype="int16")
                stream.start()
            stream.write(np.frombuffer(audio_bytes, dtype=np.int16))
            played_any = True
            return True

        async def _run():
            headers = {"api-subscription-key": self.api_key}
            connect_kwargs = {"additional_headers": headers}
            if websockets.__version__ < "13.0":
                connect_kwargs = {"extra_headers": headers}
            url = _SARVAM_TTS_WS + "?model=bulbul:v3"
            async with websockets.connect(url, **connect_kwargs) as ws:
                await ws.send(json.dumps({
                    "type": "config",
                    "data": {
                        "target_language_code": lang,
                        "speaker": speaker,
                        "model": "bulbul:v3",
                        "speech_sample_rate": "16000",
                        "output_audio_codec": "linear16",
                        "pace": pace,
                    },
                }))
                await ws.send(json.dumps({"type": "text", "data": {"text": text}}))
                await ws.send(json.dumps({"type": "flush"}))
                while True:
                    try:
                        chunk = await asyncio.wait_for(ws.recv(), timeout=6.0)
                    except (asyncio.TimeoutError, websockets.ConnectionClosed):
                        break
                    if _stale():
                        return
                    if isinstance(chunk, str):
                        data = json.loads(chunk)
                        if data.get("type") == "audio":
                            b64 = (data.get("data") or {}).get("audio", "")
                            if b64:
                                _sink(base64.b64decode(b64))
                        elif data.get("type") == "error":
                            raise RuntimeError(
                                (data.get("data") or {}).get("message", "sarvam tts error"))
                        elif data.get("type") == "event":
                            if (data.get("data") or {}).get("event_type") == "final":
                                break
                    elif isinstance(chunk, bytes):
                        _sink(chunk)

        try:
            asyncio.run(_run())
        except Exception as e:
            print(f"[TTS] Sarvam error: {e}")
            self._speak_local(text)
            return
        finally:
            if stream is not None:
                _time.sleep(0.3)  # let the output buffer drain (tail of last word)
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
        if not played_any and not _stale():
            self._speak_local(text)

    def _speak_local(self, text: str):
        """Local fallback using pyttsx3 (thread-safe).

        pyttsx3's run loop is not re-entrant: calling runAndWait() from two
        threads at once raises "RuntimeError: run loop already started". All
        fallback playback is serialized through one shared engine + lock.
        """
        global _LOCAL_TTS_ENGINE
        if _HAS_TTS_LOCAL:
            with _LOCAL_TTS_LOCK:
                try:
                    if _LOCAL_TTS_ENGINE is None:
                        _LOCAL_TTS_ENGINE = pyttsx3.init()
                    _LOCAL_TTS_ENGINE.say(text)
                    _LOCAL_TTS_ENGINE.runAndWait()
                except Exception:
                    print(f"[TTS] {text}")
        else:
            print(f"[TTS] {text}")

    # --- SpeechQueue: sentence-level spoken output -------------------

    def start_speech_queue(self):
        """Start the sentence-level speech worker.

        Enqueued sentences play back-to-back; enqueue_speech() returns
        immediately so the LLM keeps streaming while ARIA talks (the
        GPT-Live "speak first, think in parallel" behavior).
        """
        if self._speech_thread is None or not self._speech_thread.is_alive():
            self._speech_stop.clear()
            self._speech_queue = queue.Queue()
            self._speech_thread = threading.Thread(
                target=self._speech_worker, daemon=True, name="SpeechQueue")
            self._speech_thread.start()

    def enqueue_speech(self, text: str):
        """Queue one sentence for playback; returns immediately."""
        text = (text or "").strip()
        if not text:
            return
        self.start_speech_queue()
        self._speech_queue.put(text)

    def wait_speech_done(self, timeout: float | None = None):
        """Block until the speech queue is drained AND the last utterance's
        audio has finished playing.

        Polls the queue + busy flag; never joins the worker thread (it stays
        alive waiting for the next sentence).
        """
        q = self._speech_queue
        if q is None:
            return
        deadline = None if timeout is None else _time.time() + timeout
        while (not q.empty()) or self._speech_busy.is_set():
            if deadline is not None and _time.time() > deadline:
                return
            _time.sleep(0.05)

    def _speech_worker(self):
        while not self._speech_stop.is_set():
            try:
                text = self._speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._speech_stop.is_set():
                break
            self._speech_busy.set()
            try:
                self.speak(text, interrupt=False)  # generation-checked barge-in
            finally:
                self._speech_busy.clear()

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
