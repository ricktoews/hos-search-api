import json
import math
import os
from typing import Optional

import mysql.connector
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

CONFIG_PATH = "/etc/hearts-of-space/mysql.json"
MODEL = "text-embedding-3-small"
DEFAULT_LIMIT = 10

PROGRAM_CACHE = []

# OpenAI key
with open(os.path.expanduser("/etc/hearts-of-space/.openai_key"), "r", encoding="utf-8") as f:
    os.environ["OPENAI_API_KEY"] = f.read().strip()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://hos-search.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()


class SearchQuery(BaseModel):
    text: Optional[str] = None
    program_name: Optional[str] = None
    genre: Optional[str] = None
    program_content: Optional[str] = None
    limit: int = DEFAULT_LIMIT


def get_db_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_connection():
    return mysql.connector.connect(**get_db_config())


def normalize(value):
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot / (norm_a * norm_b)


@app.on_event("startup")
def load_program_cache():
    global PROGRAM_CACHE

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT
          p.id,
          p.program_number,
          p.title,
          p.short_description,
          p.description,
          p.weather_report,
          p.program_date,
          p.producer,
          p.gallery_url,
          pe.embedding
        FROM programs p
        JOIN program_embeddings pe ON pe.program_id = p.id
        ORDER BY p.program_number
    """)

    PROGRAM_CACHE = []

    for row in cursor.fetchall():
        row["embedding_vector"] = json.loads(row["embedding"])
        del row["embedding"]
        PROGRAM_CACHE.append(row)

    cursor.close()
    conn.close()

    print(f"Loaded {len(PROGRAM_CACHE)} program embeddings into memory.")


def get_filtered_program_ids(program_name=None, genre=None, program_content=None):
    program_name = normalize(program_name)
    genre = normalize(genre)
    program_content = normalize(program_content)

    if not program_name and not genre and not program_content:
        return None

    where_clauses = []
    params = []

    if program_name:
        where_clauses.append("p.title LIKE %s")
        params.append(f"%{program_name}%")

    if genre:
        where_clauses.append("""
            EXISTS (
              SELECT 1
              FROM program_genres pg
              JOIN genres g ON g.id = pg.genre_id
              WHERE pg.program_id = p.id
                AND g.name = %s
            )
        """)
        params.append(genre)

    if program_content:
        where_clauses.append("""
            (
              p.description LIKE %s
              OR p.short_description LIKE %s
              OR p.weather_report LIKE %s
              OR EXISTS (
                SELECT 1
                FROM program_tracks pt2
                JOIN tracks t2 ON t2.id = pt2.track_id
                WHERE pt2.program_id = p.id
                  AND t2.title LIKE %s
              )
              OR EXISTS (
                SELECT 1
                FROM program_tracks pt3
                JOIN track_artists ta3 ON ta3.track_id = pt3.track_id
                JOIN artists a3 ON a3.id = ta3.artist_id
                WHERE pt3.program_id = p.id
                  AND a3.name LIKE %s
              )
              OR EXISTS (
                SELECT 1
                FROM program_tracks pt4
                JOIN albums al4 ON al4.id = pt4.album_id
                WHERE pt4.program_id = p.id
                  AND al4.title LIKE %s
              )
            )
        """)
        like = f"%{program_content}%"
        params.extend([like, like, like, like, like, like])

    where_sql = "WHERE " + " AND ".join(where_clauses)

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(f"""
        SELECT p.id
        FROM programs p
        {where_sql}
    """, params)

    ids = {row["id"] for row in cursor.fetchall()}

    cursor.close()
    conn.close()

    return ids


def fetch_tracks_for_programs(program_ids):
    if not program_ids:
        return {}

    placeholders = ",".join(["%s"] * len(program_ids))

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(f"""
        SELECT
          pt.program_id,
          pt.start_position_in_stream,
          GROUP_CONCAT(DISTINCT a.name ORDER BY a.name SEPARATOR ', ') AS artist,
          t.title AS track,
          al.title AS album,
          pt.duration,
          pt.attributes
        FROM program_tracks pt
        JOIN tracks t ON t.id = pt.track_id
        LEFT JOIN albums al ON al.id = pt.album_id
        LEFT JOIN track_artists ta ON ta.track_id = t.id
        LEFT JOIN artists a ON a.id = ta.artist_id
        WHERE pt.program_id IN ({placeholders})
        GROUP BY
          pt.program_id,
          pt.track_id,
          pt.start_position_in_stream,
          t.title,
          al.title,
          pt.duration,
          pt.attributes
        ORDER BY pt.program_id, pt.start_position_in_stream
    """, program_ids)

    tracks_by_program = {}

    for row in cursor.fetchall():
        tracks_by_program.setdefault(row["program_id"], []).append({
            "start_position_in_stream": row["start_position_in_stream"],
            "artist": row["artist"],
            "track": row["track"],
            "album": row["album"],
            "duration": row["duration"],
            "attributes": row["attributes"],
        })

    cursor.close()
    conn.close()

    return tracks_by_program


def build_result(row, score=None):
    return {
        "id": row["id"],
        "score": round(score, 4) if score is not None else None,
        "program_number": row["program_number"],
        "title": row["title"],
        "description": row["description"],
        "short_description": row["short_description"],
        "producer": row["producer"],
        "program_date": row["program_date"],
        "gallery_url": row["gallery_url"],
        "weather_report": row["weather_report"],
    }


def get_program_by_number(program_number):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT
          p.id,
          p.program_number,
          p.title,
          p.short_description,
          p.description,
          p.weather_report,
          p.program_date,
          p.producer,
          p.gallery_url
        FROM programs p
        WHERE p.program_number = %s
        LIMIT 1
    """, (program_number,))

    row = cursor.fetchone()

    cursor.close()
    conn.close()

    if not row:
        return None

    result = build_result(row)
    tracks_by_program = fetch_tracks_for_programs([result["id"]])
    result["tracks"] = tracks_by_program.get(result["id"], [])
    del result["id"]

    return result


def run_search(
    text=None,
    program_name=None,
    genre=None,
    program_content=None,
    limit=DEFAULT_LIMIT
):
    text = normalize(text)
    limit = max(1, min(limit or DEFAULT_LIMIT, 100))

    filtered_ids = get_filtered_program_ids(
        program_name=program_name,
        genre=genre,
        program_content=program_content,
    )

    if filtered_ids is None:
        candidate_rows = PROGRAM_CACHE
    else:
        candidate_rows = [row for row in PROGRAM_CACHE if row["id"] in filtered_ids]

    results = []

    if text:
        query_embedding = client.embeddings.create(
            model=MODEL,
            input=text
        ).data[0].embedding

        for row in candidate_rows:
            score = cosine_similarity(query_embedding, row["embedding_vector"])
            results.append(build_result(row, score))

        results.sort(key=lambda r: r["score"], reverse=True)
    else:
        for row in candidate_rows:
            results.append(build_result(row))

        results.sort(key=lambda r: r["program_number"])

    top_results = results[:limit]

    if not top_results:
        return []

    program_ids = [r["id"] for r in top_results]
    tracks_by_program = fetch_tracks_for_programs(program_ids)

    for result in top_results:
        result["tracks"] = tracks_by_program.get(result["id"], [])
        del result["id"]

    return top_results


def more_like_this(
    program_number=None,
    limit=DEFAULT_LIMIT
):
    candidate_rows = PROGRAM_CACHE

    results = []

    target_row = next(
        (row for row in candidate_rows
         if row["program_number"] == program_number),
        None
    )

    if not target_row:
        return []

    this_embedding = target_row["embedding_vector"]

    for row in candidate_rows:
        if row["program_number"] == program_number:
            continue

        score = cosine_similarity(this_embedding, row["embedding_vector"])
        results.append(build_result(row, score))

    results.sort(key=lambda r: r["score"], reverse=True)

    top_results = results[:limit]

    if not top_results:
        return []

    program_ids = [r["id"] for r in top_results]
    tracks_by_program = fetch_tracks_for_programs(program_ids)

    for result in top_results:
        result["tracks"] = tracks_by_program.get(result["id"], [])
        del result["id"]

    return top_results


def genre_query():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT
          g.id,
          g.name
        FROM genres g
        ORDER BY g.name
    """)

    results = []

    for row in cursor.fetchall():
        results.append({
            "id": row["id"],
            "genre": row["name"],
        })

    cursor.close()
    conn.close()

    return results


@app.get("/genres")
def get_genres():
    return genre_query()


@app.get("/programs/{program_number}")
def get_program(
    program_number: int
):
    program = get_program_by_number(program_number)

    if not program:
        raise HTTPException(status_code=404, detail="Program not found")

    return program


@app.get("/similar/{program_number}")
def get_similar_programs(
    program_number: int,
    limit: int = DEFAULT_LIMIT
):
    return more_like_this(
        program_number=program_number,
        limit=limit
    )


@app.post("/search")
def search(query: SearchQuery):
    return run_search(
        text=query.text,
        program_name=query.program_name,
        genre=query.genre,
        program_content=query.program_content,
        limit=query.limit,
    )


@app.get("/preset-moods")
def get_preset_moods():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT slug, name, description
        FROM preset_moods
        WHERE active = 1
        ORDER BY name
    """)

    moods = cursor.fetchall()

    cursor.close()
    conn.close()

    return moods


@app.get("/preset-moods/{slug}/programs")
def get_preset_mood_programs(slug: str, limit: int = 20):
    limit = min(max(limit, 1), 100)

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT id, slug, name, description
        FROM preset_moods
        WHERE slug = %s
          AND active = 1
    """, (slug,))

    mood = cursor.fetchone()

    if not mood:
        cursor.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Preset mood not found")

    cursor.execute("""
        SELECT
            p.id,
            p.program_number,
            p.title,
            p.short_description,
            p.description,
            p.weather_report,
            p.program_date,
            p.producer,
            p.gallery_url,
            pmp.similarity_score,
            pmp.rank_order
        FROM preset_mood_programs pmp
        JOIN programs p ON p.id = pmp.program_id
        WHERE pmp.preset_mood_id = %s
        ORDER BY pmp.rank_order, p.program_number
        LIMIT %s
    """, (mood["id"], limit))

    program_rows = cursor.fetchall()

    cursor.close()
    conn.close()

    program_ids = [row["id"] for row in program_rows]
    tracks_by_program = fetch_tracks_for_programs(program_ids)

    programs = []

    for row in program_rows:
        program = build_result(row, row["similarity_score"])
        program["similarity_score"] = row["similarity_score"]
        program["rank_order"] = row["rank_order"]
        program["tracks"] = tracks_by_program.get(program["id"], [])
        del program["id"]
        programs.append(program)

    return {
        "mood": {
            "slug": mood["slug"],
            "name": mood["name"],
            "description": mood["description"]
        },
        "programs": programs
    }
