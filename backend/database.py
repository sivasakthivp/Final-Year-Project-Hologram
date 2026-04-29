"""
Database Module — Async SQL (SQLite dev / PostgreSQL prod)
Handles patient records and session persistence.
"""

import json
import logging
from typing import Optional, Any
from datetime import datetime

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = "data/hologram_medical.db"


class Database:
    def __init__(self):
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self):
        self._conn = await aiosqlite.connect(DB_PATH)
        self._conn.row_factory = aiosqlite.Row
        await self._create_tables()
        logger.info("Database connected.")

    async def disconnect(self):
        if self._conn:
            await self._conn.close()

    async def _create_tables(self):
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                name TEXT,
                dob TEXT,
                gender TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                patient_id TEXT,
                file_path TEXT,
                status TEXT DEFAULT 'pending',
                features TEXT,
                metadata TEXT,
                output_path TEXT,
                error TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            );
        """)
        await self._conn.commit()

    async def create_session(self, session_id: str, file_path: str, patient_id: Optional[str]):
        await self._conn.execute(
            """INSERT INTO sessions (id, patient_id, file_path, status)
               VALUES (?, ?, ?, 'pending')""",
            (session_id, patient_id, file_path),
        )
        await self._conn.commit()

    async def update_session(self, session_id: str, data: dict[str, Any]):
        fields = []
        values = []
        for key, val in data.items():
            if isinstance(val, (dict, list)):
                val = json.dumps(val)
            fields.append(f"{key} = ?")
            values.append(val)
        fields.append("updated_at = ?")
        values.append(datetime.utcnow().isoformat())
        values.append(session_id)
        query = f"UPDATE sessions SET {', '.join(fields)} WHERE id = ?"
        await self._conn.execute(query, values)
        await self._conn.commit()

    async def get_session(self, session_id: str) -> Optional[dict]:
        cursor = await self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        result = dict(row)
        for field in ("features", "metadata"):
            if result.get(field):
                try:
                    result[field] = json.loads(result[field])
                except Exception:
                    pass
        return result

    async def delete_session(self, session_id: str):
        await self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await self._conn.commit()

    async def list_patients(self) -> list[dict]:
        cursor = await self._conn.execute("SELECT * FROM patients ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def upsert_patient(self, patient_id: str, name: str, dob: str, gender: str):
        await self._conn.execute(
            """INSERT INTO patients (id, name, dob, gender)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET name=excluded.name,
               dob=excluded.dob, gender=excluded.gender""",
            (patient_id, name, dob, gender),
        )
        await self._conn.commit()

