"""MicVAD — always-on microphone with energy-based voice activity detection.

Phase 1 of the talkative-ARIA upgrade (OpenAI GPT-Live-style turn taking,
pseudo-duplex edition for laptop mic + speakers):

- A dedicated thread reads the mic in 30ms frames (16kHz, mono, int16)
- Energy VAD with adaptive noise floor marks speech onsets/offsets
- An utterance = speech onset, then silence >= 600ms (configurable), or a
  hard cap (max_utterance). wait_for_utterance() blocks until one is ready.
- Echo safety: while ARIA is SPEAKING the VAD is suppressed (frames during
  playback never open an utterance), so ARIA never transcribes its own voice
  through the laptop speakers. It "yields" to the user the instant they
  actually speak, without ARIA hearing itself.

This replaces fixed 8-12s record windows: capture starts at speech onset and
ends at the natural end of the phrase, so responses feel immediate.
"""
import os
import threading
import time
from collections import deque

import numpy as np

try:
    import sounddevice as sd
    _HAS_SOUND = True
except ImportError:
    _HAS_SOUND = False

# Env-tunable VAD tuning (tune_vad.py writes these into .env)
DEFAULT_SILENCE_MS = float(os.getenv("ARIA_VAD_SILENCE_MS", "600"))
DEFAULT_THRESHOLD = float(os.getenv("ARIA_VAD_THRESHOLD", "2.6"))
DEFAULT_MIN_SPEECH_MS = float(os.getenv("ARIA_VAD_MIN_SPEECH_MS", "180"))


class MicVAD:
    """Background mic reader producing complete speech utterances."""

    SAMPLE_RATE = 16000
    FRAME_MS = 30

    def __init__(self,
                 silence_ms: float = DEFAULT_SILENCE_MS,
                 speech_threshold: float = DEFAULT_THRESHOLD,
                 pre_roll_ms: float = 240.0,
                 min_speech_ms: float = DEFAULT_MIN_SPEECH_MS,
                 debug: bool = False):
        self.silence_ms = silence_ms        # quiet needed to end a turn
        self.speech_threshold = speech_threshold  # x noise floor = speech
        self.pre_roll_frames = int(pre_roll_ms / self.FRAME_MS)
        self.min_speech_frames = int(min_speech_ms / self.FRAME_MS)
        self.debug = debug                  # print live VAD levels
        self.in_speech = False              # True while an utterance is open

        self.speaking = False          # True while ARIA's TTS plays (echo guard)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._utterances: deque[np.ndarray] = deque(maxlen=4)
        self._utterance_event = threading.Event()
        self._lock = threading.Lock()

        # Adaptive noise floor
        self._noise_floor = 150.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        if not _HAS_SOUND:
            print("[MicVAD] sounddevice unavailable — voice input disabled")
            return False
        if self._thread is not None and self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="MicVAD")
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def wait_for_utterance(self, max_wait: float = 12.0,
                           max_utterance: float = 8.0) -> np.ndarray | None:
        """Block until a complete utterance is available (or timeout).

        Returns int16 mono PCM at 16kHz, or None on timeout/no speech.
        """
        deadline = time.time() + max_wait
        # Wait for a fresh utterance
        while time.time() < deadline:
            with self._lock:
                if self._utterances:
                    utt = self._utterances.popleft()
                    dur = len(utt) / float(self.SAMPLE_RATE)
                    print(f"[MicVAD] Utterance dequeued: {dur:.2f}s ({len(utt)} samples, remaining in queue: {len(self._utterances)})")
                    return utt
            self._utterance_event.wait(timeout=0.1)
            self._utterance_event.clear()
        return None

    def clear_pending(self):
        """Drop queued utterances (e.g. after a state reset)."""
        with self._lock:
            self._utterances.clear()
        self._utterance_event.clear()

    # ------------------------------------------------------------------
    # VAD loop
    # ------------------------------------------------------------------

    def _loop(self):
        frame_len = int(self.SAMPLE_RATE * self.FRAME_MS / 1000)
        pre_roll = deque(maxlen=self.pre_roll_frames)
        ringing = deque(maxlen=30)  # ~0.9s of recent RMS for the noise floor

        in_speech = False
        speech_frames = 0
        silence_frames = 0
        max_frames = int(8.0 * 1000 / self.FRAME_MS)  # hard cap 8s
        buf: list[np.ndarray] = []

        try:
            stream = sd.InputStream(samplerate=self.SAMPLE_RATE, channels=1,
                                    dtype="int16", blocksize=frame_len)
            stream.start()
        except Exception as e:
            print(f"[MicVAD] cannot open mic: {e}")
            return

        while not self._stop.is_set():
            try:
                audio, _ = stream.read(frame_len)
            except Exception:
                time.sleep(0.05)
                continue
            frame = audio[:, 0]
            rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
            if self.speaking:
                # Echo guard: ARIA is talking — reset turn state and keep only
                # the pre-roll so a user interrupt mid-speech is still caught.
                if in_speech:
                    print("[MicVAD] Echo guard active: suppressing in-flight utterance because ARIA is speaking")
                in_speech = False
                speech_frames = silence_frames = 0
                buf = []
                pre_roll.clear()
                continue

            is_speech = rms > self._noise_floor * self.speech_threshold
            self.in_speech = in_speech

            # Only adapt the noise floor during ambient silence, never during active
            # speech or speech-onset candidates. Otherwise the floor tracks the
            # speaker's voice, the gate skyrockets (e.g. 80 -> 3500 -> 9000), and
            # subsequent words in the same sentence are falsely cut off as "silence".
            if not in_speech and not is_speech:
                ringing.append(rms)
                if len(ringing) >= 10:
                    self._noise_floor = max(80.0, float(np.median(ringing)))

            if self.debug and int(time.time() * 5) != getattr(self, "_dbg_tick", -1):
                self._dbg_tick = int(time.time() * 5)
                print(f"[VAD] rms={rms:7.1f} floor={self._noise_floor:6.1f} "
                      f"gate={self._noise_floor * self.speech_threshold:7.1f} "
                      f"speech={in_speech}")

            if not in_speech:
                pre_roll.append(frame)
                if is_speech:
                    speech_frames += 1
                    if speech_frames >= self.min_speech_frames:
                        # Speech onset confirmed — open the utterance
                        in_speech = True
                        silence_frames = 0
                        buf = list(pre_roll)
                        pre_roll.clear()
                        print(f"[MicVAD] Speech onset confirmed (rms={rms:.1f}, floor={self._noise_floor:.1f}, gate={self._noise_floor * self.speech_threshold:.1f})")
                else:
                    speech_frames = 0
            else:
                buf.append(frame)
                silence_frames = 0 if is_speech else silence_frames + 1
                if silence_frames * self.FRAME_MS >= self.silence_ms:
                    emitted = buf[:-silence_frames] if silence_frames else buf
                    dur = len(emitted) * self.FRAME_MS / 1000.0
                    print(f"[MicVAD] Silence detected ({silence_frames * self.FRAME_MS:.0f}ms >= {self.silence_ms:.0f}ms). Closing utterance ({dur:.2f}s).")
                    self._emit(emitted)
                    in_speech, buf = False, []
                    speech_frames = silence_frames = 0
                elif len(buf) >= max_frames:
                    dur = len(buf) * self.FRAME_MS / 1000.0
                    print(f"[MicVAD] Max utterance length reached ({dur:.2f}s). Closing utterance.")
                    self._emit(buf)
                    in_speech, buf = False, []
                    speech_frames = silence_frames = 0

        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

    def _emit(self, frames: list[np.ndarray]):
        if not frames:
            return
        pcm = np.concatenate(frames).astype(np.int16)
        # Cheap trim of leading/trailing near-silence
        rms = np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2, axis=1)
                      if pcm.ndim > 1 else (pcm.astype(np.float32) / 32768.0) ** 2)
        if float(np.max(rms)) <= 0:
            print("[MicVAD] Discarded near-zero RMS utterance")
            return
        with self._lock:
            self._utterances.append(pcm)
        dur = len(pcm) / float(self.SAMPLE_RATE)
        print(f"[MicVAD] Emitted utterance to queue: {dur:.2f}s ({len(pcm)} samples, queue_len={len(self._utterances)})")
        self._utterance_event.set()
