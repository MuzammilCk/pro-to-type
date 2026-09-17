"""Benchmark every viable FREE OpenRouter model for ARIA-style conversation.

Measures time-to-first-token (TTFT — the "does she feel alive" metric),
total time, and reply quality. One-off decision tool; delete after use.

Run: python bench_models.py
"""
import os
import time
import json
import httpx


def load_env_key():
    if os.getenv("OPENROUTER_API_KEY"):
        return os.getenv("OPENROUTER_API_KEY")
    try:
        with open(".env", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return None


SYSTEM = (
    "You are ARIA, a warm, talkative companion AI speaking out loud through "
    "speakers. Reply in 1-2 short spoken sentences. No markdown, no lists, "
    "no emoji. Be natural and personable."
)
USER = "hey aria, what's up?"

MODELS = [
    "nex-agi/nex-n2.5-pro:free",           # current default
    "google/gemma-4-26b-a4b-it:free",      # current fallback
    "liquid/lfm-2.5-2.6b:free",            # current voice-brain opt-in
    "z-ai/glm-5.2:free",
    "nvidia/nemotron-3.5-lightning:free",  # name says fast
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",  # likely smart but slow
    "inclusionai/ling-3.0-flash-sante:free",
]


def bench(key: str, model: str) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
        "stream": True,
        "max_tokens": 80,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "http://localhost:8080",
        "X-Title": "ARIA-bench",
        "Content-Type": "application/json",
    }
    t0 = time.perf_counter()
    ttft = None
    text = ""
    try:
        with httpx.Client(timeout=25) as client:
            with client.stream("POST", "https://openrouter.ai/api/v1/chat/completions",
                               headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    body = resp.read().decode("utf-8", "replace")[:120]
                    return {"model": model, "error": f"HTTP {resp.status_code}: {body}"}
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        text += delta
        if ttft is None:
            return {"model": model, "error": "no tokens returned (empty reply)"}
        total = time.perf_counter() - t0
        return {"model": model, "ttft": ttft, "total": total,
                "words": len(text.split()), "reply": " ".join(text.split())[:100]}
    except Exception as e:  # noqa: BLE001
        return {"model": model, "error": f"{type(e).__name__}: {str(e)[:80]}"}


def main():
    key = load_env_key()
    if not key:
        print("No OPENROUTER_API_KEY found (.env or environment).")
        return
    results = []
    for model in MODELS:
        print(f"  testing {model} ...", flush=True)
        r = bench(key, model)
        results.append(r)
        if "error" in r:
            print(f"    ERROR  {r['error']}")
        else:
            print(f"    TTFT {r['ttft']*1000:6.0f}ms | total {r['total']:5.1f}s | "
                  f"{r['words']} words | \"{r['reply'][:70]}\"")

    ok = [r for r in results if "error" not in r]
    ok.sort(key=lambda r: r["ttft"])
    print("\n=== Ranked by TTFT (time to first token) ===")
    for i, r in enumerate(ok, 1):
        print(f"{i}. {r['ttft']*1000:6.0f}ms TTFT | {r['total']:5.1f}s total | {r['model']}")
    print("\n=== Errors ===")
    for r in results:
        if "error" in r:
            print(f"- {r['model']}: {r['error']}")


if __name__ == "__main__":
    main()
