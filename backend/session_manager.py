"""
Session Manager — In-memory pipeline status tracking.
"""

import threading
from typing import Optional


class SessionManager:
    """Thread-safe in-memory store for pipeline session statuses."""

    VALID_STATUSES = [
        "pending",
        "extracting_dicom",
        "preprocessing",
        "segmenting",
        "super_resolution",
        "volumetric_features",
        "depth_estimation",
        "pointcloud",
        "hologram_generation",
        "completed",
        "error",
    ]

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: dict[str, str] = {}

    def register(self, session_id: str):
        with self._lock:
            self._sessions[session_id] = "pending"

    def update_status(self, session_id: str, status: str):
        if status not in self.VALID_STATUSES:
            raise ValueError(f"Unknown status: {status}")
        with self._lock:
            self._sessions[session_id] = status

    def get_status(self, session_id: str) -> Optional[str]:
        with self._lock:
            return self._sessions.get(session_id, "unknown")

    def remove(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)

    def all_sessions(self) -> dict[str, str]:
        with self._lock:
            return dict(self._sessions)

