"""Presence state manager — turns raw face tracking into clean events.

Based on ARIA_ARCHITECTURE.md:
- Maintain a table of track IDs -> {name-or-unknown, first_seen, last_seen, greeted}
- Emit exactly ONE event per state transition
- Cooldown: a person already greeted in the last N minutes shouldn't re-trigger

Events: person_entered, person_recognized, person_unrecognized, person_left
"""
import time
import queue
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrackState:
    track_id: str
    identity: str = "unknown"  # "unknown" or person's name
    authorized: bool = False
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_score: float = 0.0
    greeted: bool = False
    greeting_time: float = 0.0
    left: bool = False


class PresenceManager:
    """Consumes face tracking results and emits de-duplicated presence events.

    Architecture: vision thread calls update(), which may push events to
    the conversation queue. Event-driven, not per-frame.
    """

    GREETING_COOLDOWN = 300.0  # 5 minutes — don't re-greet same person

    def __init__(self, event_queue: queue.Queue):
        self.tracks: dict[str, TrackState] = {}
        self.event_queue = event_queue
        self._next_id = 0
        # B9 (identity-level cooldown): track IDs are ephemeral — a person who
        # steps out and back in gets a NEW track_id, which used to bypass the
        # per-track greeting cooldown entirely. Gate greetings by identity too.
        self._identity_greeted: dict[str, float] = {}  # name -> last greeting time

    def update(self, face_results: list[dict], now: float | None = None):
        """Process face detection results and emit state-transition events."""
        now = now or time.time()
        active_ids = set()

        for fr in face_results:
            track_id = fr.get("track_id", f"face_{self._next_id}")
            self._next_id += 1
            active_ids.add(track_id)

            if track_id not in self.tracks:
                self.tracks[track_id] = TrackState(
                    track_id=track_id,
                    identity=fr["identity"],
                    authorized=fr.get("authorized", False),
                    first_seen=now,
                    last_seen=now,
                    last_score=fr.get("distance", 0.0),
                )
                self._emit("person_entered", track_id, fr)

                if fr.get("authorized", False):
                    self._emit_recognized(track_id, fr)
                else:
                    self._emit_unrecognized(track_id, fr)

            else:
                track = self.tracks[track_id]
                track.last_seen = now
                track.last_score = fr.get("distance", 0.0)

                # Phase 7 — SYMMETRIC transitions:
                #   UNKNOWN -> AUTHORIZED : only on fresh authorization
                #   AUTHORIZED -> UNKNOWN : tracker revoked -> downgrade NOW
                # The PresenceManager never preserves a previous identity
                # once the tracker says authorization is gone.
                if fr.get("authorized", False):
                    new_id = fr.get("identity", "unknown")
                    if not track.authorized or track.identity != new_id:
                        prev = track.identity
                        if prev not in (None, "unknown", new_id):
                            # Identity CHANGED on a live track (fresh evidence
                            # says this is someone else): revoke the old name
                            # first so any conversation with it terminates.
                            self._emit_unrecognized(track_id, fr, previous_identity=prev)
                        track.identity = new_id
                        track.authorized = True
                        self._emit_recognized(track_id, fr)
                elif track.authorized or track.identity != "unknown":
                    # Tracker explicitly revoked / reclassified this track.
                    prev = track.identity
                    track.authorized = False
                    track.identity = "unknown"
                    self._emit_unrecognized(track_id, fr, previous_identity=prev)

                if track.greeted and now - track.greeting_time > self.GREETING_COOLDOWN:
                    track.greeted = False

        for tid in list(self.tracks.keys()):
            if tid not in active_ids:
                track = self.tracks[tid]
                if now - track.last_seen > 2.0 and not track.left:
                    track.left = True
                    self._emit("person_left", tid, {
                        "track_id": tid,
                        "identity": track.identity,
                        "duration": now - track.first_seen,
                    })
                elif now - track.last_seen > 10.0:
                    del self.tracks[tid]

    def mark_greeted(self, track_id: str, now: float | None = None):
        """Mark a track as greeted so we don't re-trigger the same event."""
        if track_id in self.tracks:
            ts = now or time.time()
            track = self.tracks[track_id]
            track.greeted = True
            track.greeting_time = ts
            if track.identity and track.identity != "unknown":
                self._identity_greeted[track.identity] = ts

    def should_greet(self, track_id: str) -> bool:
        """Check if this track should be greeted (haven't greeted recently)."""
        if track_id not in self.tracks:
            return False
        track = self.tracks[track_id]
        if track.greeted:
            return time.time() - track.greeting_time > self.GREETING_COOLDOWN
        return True

    def _emit(self, event_type: str, track_id: str, data: dict):
        self.event_queue.put({
            "type": event_type,
            "track_id": track_id,
            "timestamp": time.time(),
            **data,
        })

    def _emit_recognized(self, track_id: str, fr: dict):
        # Identity-level cooldown: suppress the event if this person was
        # greeted recently, even under a different track_id.
        name = fr.get("identity", "unknown")
        now = time.time()
        track = self.tracks.get(track_id)
        if name and name != "unknown":
            last = self._identity_greeted.get(name)
            if last is not None and now - last < self.GREETING_COOLDOWN:
                if track:
                    track.greeted = True
                    track.greeting_time = last
                return
        if track:
            track.greeted = False
        self._emit("person_recognized", track_id, {
            "name": name,
            "score": fr.get("distance", 0.0),
            "face_bbox": fr.get("face_bbox", (0, 0, 0, 0)),
        })

    def _emit_unrecognized(self, track_id: str, fr: dict,
                           previous_identity: str | None = None):
        track = self.tracks.get(track_id)
        if track:
            track.identity = "unknown"
            track.greeted = False
        self._emit("person_unrecognized", track_id, {
            "score": fr.get("distance", 0.0),
            "face_bbox": fr.get("face_bbox", (0, 0, 0, 0)),
            "quality": fr.get("quality", 0.0),
            # Phase 8: who JUST lost the track — lets the conversation layer
            # terminate an authenticated dialogue for exactly that identity.
            "previous_identity": previous_identity,
        })
