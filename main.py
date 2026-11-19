import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field, field_validator


DATABASE_PATH = Path(
    os.getenv("SEARCH_HISTORY_DB", "search_history.sqlite3")
)


class SearchCreate(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=2000)

    @field_validator("user_id", "query")
    @classmethod
    def strip_non_empty_values(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This field must not be blank.")
        return value


class SearchRecord(SearchCreate):
    id: int
    created_at: datetime


def initialize_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title="Skyterbird Search History API",
    lifespan=lifespan,
)


@app.post("/searches", response_model=SearchRecord, status_code=201)
def save_search(search: SearchCreate) -> SearchRecord:
    created_at = datetime.now(timezone.utc)
    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.execute(
            "INSERT INTO searches"
            "(user_id, query, created_at) VALUES (?, ?, ?)",
            (search.user_id, search.query, created_at.isoformat()),
        )
        search_id = cursor.lastrowid
        if search_id is None:
            raise RuntimeError("The search record was not assigned an ID.")

    return SearchRecord(
        id=search_id,
        user_id=search.user_id,
        query=search.query,
        created_at=created_at,
    )


@app.get("/searches", response_model=list[SearchRecord])
def list_searches(
    user_id: Annotated[str, Query(min_length=1, max_length=100)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[SearchRecord]:
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, user_id, query, created_at
            FROM searches
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    return [
        SearchRecord(
            id=row["id"],
            user_id=row["user_id"],
            query=row["query"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
        for row in rows
    ]
