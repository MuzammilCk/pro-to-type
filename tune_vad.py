"""tune_vad.py — interactive VAD calibration for YOUR room.

Measures your room's silence floor and speech level, computes a signal
margin (speech / silence ratio), recommends VAD settings, writes them to
.env, then live-monitors so you can verify detection end-to-end.

Usage:
    python tune_vad.py            # full flow: measure → recommend → verify
    python tune_vad.py --monitor  # skip calibration, just live-monitor
"""
import os
import sys
import time

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice not installed — cannot calibrate")
    sys.exit(1)

SR = 16000


def read_seconds(seconds: float, label: str) -> np.ndarray:
    n = int(SR * seconds)
    print(f"\n  >> {label} ({seconds:.0f}s)...", flush=True)
    frames = []
    remaining = n
    while remaining > 0:
        block = min(1600, remaining)  # 0.1s blocks
        data = sd.rec(block, samplerate=SR, channels=1, dtype="int16")
        sd.wait()
        frames.append(data)
        remaining -= block
    return np.concatenate(frames)[:, 0].astype(np.float32)


def rms_curve(pcm: np.ndarray, frame_ms: int = 30) -> np.ndarray:
    fl = int(SR * frame_ms / 1000)
    n = len(pcm) // fl
    vals = []
    for i in range(n):
        seg = pcm[i * fl:(i + 1) * fl]
        vals.append(float(np.sqrt(np.mean(seg ** 2))))
    return np.array(vals)


def percentile(x, p):
    return float(np.percentile(x, p))


def main():
    monitor_only = "--monitor" in sys.argv

    env_path = ".env"
    existing = {}
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                existing[k.strip()] = v.strip()

    silence_ms = float(existing.get("ARIA_VAD_SILENCE_MS", "600"))
    threshold = float(existing.get("ARIA_VAD_THRESHOLD", "2.6"))

    if not monitor_only:
        print("=" * 62)
        print("ARIA VAD calibration — laptop mic, your room")
        print("=" * 62)
        print("  1) SIT STILL & QUIET (typing, fan, AC = fine, they're noise)")
        print("  2) SPEAK NORMALLY (a few sentences, like talking to ARIA)")

        input("\nPress ENTER when you're ready to measure SILENCE... ")
        sil = read_seconds(5.0, "stay quiet")
        sil_rms = rms_curve(sil)
        floor_p50, floor_p90 = percentile(sil_rms, 50), percentile(sil_rms, 90)
        print(f"  silence floor: p50={floor_p50:.1f}  p90={floor_p90:.1f}")

        input("\nPress ENTER when ready to SPEAK (normal voice)... ")
        sp = read_seconds(5.0, "talk normally")
        sp_rms = rms_curve(sp)
        # Speech frames = above the silence p90 (skip inter-word gaps)
        gate = max(floor_p90, 50.0)
        speech_frames = sp_rms[sp_rms > gate]
        speech_p50 = percentile(speech_frames, 50) if len(speech_frames) else gate * 2
        print(f"  speech level:  p50={speech_p50:.1f}  "
              f"({len(speech_frames)}/{len(sp_rms)} frames above gate)")

        # Margin = how far speech sits above the noise floor (median-based)
        margin = speech_p50 / max(floor_p50, 1.0)
        print(f"\n  signal margin: {margin:.2f}x "
              f"({'tight' if margin < 3 else 'good' if margin < 8 else 'excellent'})")

        # Threshold: put the gate geometric mean between floor and speech,
        # expressed as a multiple of the adaptive noise floor.
        # floor is adaptive (median of last ~0.9s), so use margin-based rules:
        if margin >= 8:
            threshold = 2.6          # plenty of headroom
        elif margin >= 4:
            threshold = 2.0
        elif margin >= 2.5:
            threshold = 1.6
        elif margin >= 1.8:
            threshold = 1.35
        else:
            threshold = 1.2          # noisy room — lean on the adaptive floor
            print("  !! Very noisy room: consider a quieter spot or headset mic")

        # End-of-turn silence: noisy rooms need slightly longer windows so
        # inter-word pauses don't split a sentence into pieces.
        silence_ms = 600 if margin >= 3 else 750 if margin >= 1.8 else 900

        print("\n  RECOMMENDED for your room:")
        print(f"    ARIA_VAD_THRESHOLD={threshold}")
        print(f"    ARIA_VAD_SILENCE_MS={int(silence_ms)}")
        print(f"    ARIA_VAD_MIN_SPEECH_MS=180")

        # Apply — append calibration lines; run.py's loader uses
        # setdefault, so later duplicate keys win over earlier ones.
        existing["ARIA_VAD_THRESHOLD"] = f"{threshold}"
        existing["ARIA_VAD_SILENCE_MS"] = f"{int(silence_ms)}"
        existing.setdefault("ARIA_VAD_MIN_SPEECH_MS", "180")
        with open(env_path, "a") as f:
            f.write("\n# --- ARIA VAD calibration (tune_vad.py) ---\n")
            for k in ("ARIA_VAD_THRESHOLD", "ARIA_VAD_SILENCE_MS", "ARIA_VAD_MIN_SPEECH_MS"):
                f.write(f"{k}={existing[k]}\n")
        print("  -> written to .env (appended; later lines win)")

    else:
        print("Live VAD monitor — calibration values from .env apply")

    # ------------------------------------------------------------------
    # Live verification monitor
    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("LIVE MONITOR — talk to your laptop like you would to ARIA")
    print("  [VAD]    = live level vs gate (rms / floor / gate / speech)")
    print("  [TURN]   = end-of-turn detected (ARIA would start replying)")
    print("  Ctrl+C   = stop")
    print("=" * 62)

    from interaction import mic_vad
    # Re-import env values (mic_vad reads env at import; we just wrote .env
    # but this process's env is stale — pass explicit values instead)
    env = {}
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    thr = float(env.get("ARIA_VAD_THRESHOLD", "2.6"))
    sil = float(env.get("ARIA_VAD_SILENCE_MS", "600"))
    print(f"  using threshold={thr}  silence_ms={sil}")

    mic = mic_vad.MicVAD(silence_ms=sil, speech_threshold=thr, debug=True)

    # Wrap _emit to print turn events
    orig_emit = mic._emit
    def emit_printer(frames):
        print("  [TURN] utterance captured")
        return orig_emit(frames)
    mic._emit = emit_printer

    mic.start()
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        mic.stop()


if __name__ == "__main__":
    main()
