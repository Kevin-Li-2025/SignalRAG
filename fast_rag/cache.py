from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class PageCache:
    def __init__(self, path: Path, ttl_seconds: int = 60 * 60 * 24) -> None:
        self.path = path
        self.ttl_seconds = ttl_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pages (
                url TEXT PRIMARY KEY,
                status INTEGER,
                content_type TEXT,
                body TEXT,
                created_at REAL
            )
            """
        )
        self._conn.commit()

    def get(self, url: str) -> tuple[int | None, str | None, str] | None:
        row = self._conn.execute(
            "SELECT status, content_type, body, created_at FROM pages WHERE url = ?",
            (url,),
        ).fetchone()
        if not row:
            return None
        status, content_type, body, created_at = row
        if time.time() - float(created_at) > self.ttl_seconds:
            return None
        return status, content_type, body

    def set(self, url: str, status: int | None, content_type: str | None, body: str) -> None:
        self._conn.execute(
            """
            INSERT INTO pages (url, status, content_type, body, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                status = excluded.status,
                content_type = excluded.content_type,
                body = excluded.body,
                created_at = excluded.created_at
            """,
            (url, status, content_type, body, time.time()),
        )
        self._conn.commit()

