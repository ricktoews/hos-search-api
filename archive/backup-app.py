import json
import math
import os
import mysql.connector
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from typing import Optional

CONFIG_PATH = "/etc/hearts-of-space/mysql.json"
MODEL = "text-embedding-3-small"
DEFAULT_LIMIT = 10

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


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot / (norm_a * norm_b)


def normalize(value):
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def run_embedding_search(
    text=None,
    program_name=None,
    genre=None,
    program_content=None,
    limit=DEFAULT_LIMIT
):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        db_config = json.load(f)

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    text = normalize(text)
    program_name = normalize(program_name)
    genre = normalize(genre)
    program_content = normalize(program_content)

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
        params.extend([like, like, like, like, like])


    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    cursor.execute(f"""
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
        LEFT JOIN program_embeddings pe ON pe.program_id = p.id
        {where_sql}
    """, params)


    results = []

    if text:
        query_embedding = client.embeddings.create(
            model=MODEL,
            input=text
        ).data[0].embedding

    for row in cursor:
        if text:
            if not row["embedding"]:
                continue
            program_embedding = json.loads(row["embedding"])
            score = cosine_similarity(query_embedding, program_embedding)
        else:
            score = None

        results.append({
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
        })


    if text:
        results.sort(key=lambda r: r["score"], reverse=True)
    else:
        results.sort(key=lambda r: r["program_number"])

    top_results = results[:limit]

    if not top_results:
        cursor.close()
        conn.close()
        return []

    program_ids = [r["id"] for r in top_results]
    placeholders = ",".join(["%s"] * len(program_ids))

    cursor.execute(f"""
        SELECT
          pt.program_id,
          pt.start_position_in_stream,
          a.name AS artist,
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

    for result in top_results:
        result["tracks"] = tracks_by_program.get(result["id"], [])
        del result["id"]


    cursor.close()
    conn.close()

    return top_results


def genre_query():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        db_config = json.load(f)

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    cursor.execute(f"""
        SELECT
          g.id,
          g.name
        FROM genres g
        ORDER BY g.name
    """)

    results = []

    for row in cursor:
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

@app.post("/search")
def search(query: SearchQuery):
    return run_embedding_search(
        text=query.text,
        program_name=query.program_name,
        genre=query.genre,
        program_content=query.program_content,
        limit=query.limit
    )
