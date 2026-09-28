"""
mcp_server.py — exposes the image/blog-post matching engine to AI agents over MCP.

WHAT THIS IS
  An MCP (Model Context Protocol) server. Any MCP host — Claude Desktop,
  the MCP Inspector, an IDE agent — can connect to it and call the tools
  below. Same data and the same guard logic as review_api.py; MCP is just a
  second front door built for agents instead of humans.

SETUP  (written for the MCP Python SDK v2)
  pip install "mcp[cli]>=2,<3"
  mcp dev mcp_server.py          # opens the MCP Inspector to try the tools
  python mcp_server.py           # runs over stdio (what hosts launch)

TOOLS
  check_pair            run the guard on a hypothetical image/post pair (no DB needed)
  list_matches          list stored matches, filterable
  get_match             full detail for one match
  matches_for_image     every post scored against one image, best first
  review_match          record a human decision (the only tool that writes)

DESIGN NOTES
  - The DB path is resolved relative to this file, not the working
    directory. MCP hosts launch servers from wherever they like, so a
    relative "data/pipeline.db" would silently point at the wrong place.
  - Tools return a clear error dict instead of raising for expected
    failures (unknown id, bad filter), so the agent gets something it can
    reason about and recover from rather than a stack trace.
  - review_match is the only write. It never touches guard_decision /
    guard_reason — same audit rule as review_api.py — and it requires an
    explicit reviewer note so an agent can't rubber-stamp silently.
  - Each tool carries MCP annotations (read-only vs. writes), so a host
    can auto-allow the four read tools and ask before review_match.
  - Never print() to stdout here: over stdio, stdout IS the protocol
    channel, and stray output corrupts it. Log to stderr instead.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from guard import guard_decision as run_guard

DB_PATH = Path(__file__).resolve().parent / "data" / "pipeline.db"
VALID_CATEGORIES = ("fox", "wolf", "dog", "other", "uncertain")

mcp = MCPServer("image-relevance")

# Hints that tell the host which tools are safe to call freely and which change data.
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITES = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False)


def get_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run the pipeline first "
            "(run_batch.py, load_blog_posts.py, compute_matches.py)."
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@mcp.tool(annotations=READ_ONLY)
def check_pair(
    image_category: str,
    conf_category: float,
    conf_caption: float,
    expected_category: str,
    similarity: Optional[float] = None,
) -> dict:
    """Run the mismatch guard on one image/blog-post pair and explain the decision.

    Use this to test "would this pairing be approved?" without touching the
    database. Categories: fox, wolf, dog, other, uncertain. Confidences are
    0-1. similarity is the cosine similarity between the image caption and
    the post; it is only needed if the pair passes every earlier check, and
    the result will say so if it is missing.
    """
    if image_category not in VALID_CATEGORIES:
        return {"error": f"image_category must be one of {VALID_CATEGORIES}"}
    for name, v in (("conf_category", conf_category), ("conf_caption", conf_caption)):
        if not 0.0 <= v <= 1.0:
            return {"error": f"{name} must be between 0 and 1, got {v}"}
    try:
        decision, reason = run_guard(
            image_category, conf_category, conf_caption, expected_category, similarity
        )
    except ValueError as e:
        return {"error": str(e), "hint": "pass similarity to finish the check"}
    return {"decision": decision, "reason": reason}


@mcp.tool(annotations=READ_ONLY)
def list_matches(
    guard_decision: Optional[Literal["approved", "rejected"]] = None,
    category: Optional[str] = None,
    reviewed: Optional[bool] = None,
    limit: int = 20,
) -> list[dict] | dict:
    """List stored image/post matches, highest similarity first.

    Filters: guard_decision ('approved' or 'rejected'), category (image
    category, e.g. 'fox'), reviewed (true = human-reviewed only, false =
    still awaiting review). limit is capped at 100.
    """
    limit = max(1, min(limit, 100))
    query = """
        SELECT m.id, m.image_id, m.post_id, m.similarity, m.guard_decision,
               m.guard_reason, m.human_decision, i.category, i.caption,
               p.title AS post_title
        FROM matches m
        JOIN images i ON m.image_id = i.id
        JOIN blog_posts p ON m.post_id = p.id
        WHERE 1=1
    """
    params: list = []
    if guard_decision:
        query += " AND m.guard_decision = ?"
        params.append(guard_decision)
    if category:
        query += " AND i.category = ?"
        params.append(category)
    if reviewed is True:
        query += " AND m.human_decision IS NOT NULL"
    elif reviewed is False:
        query += " AND m.human_decision IS NULL"
    query += " ORDER BY m.similarity DESC LIMIT ?"
    params.append(limit)

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


@mcp.tool(annotations=READ_ONLY)
def get_match(match_id: str) -> dict:
    """Get full detail for one match: image tags and confidences, the post, and both the guard's and any human's decision."""
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT m.id, m.image_id, m.post_id, m.similarity, m.guard_decision,
                   m.guard_reason, m.human_decision, m.human_note, m.reviewed_at,
                   i.filepath, i.category, i.caption, i.conf_category, i.conf_caption,
                   p.title AS post_title, p.expected_category
            FROM matches m
            JOIN images i ON m.image_id = i.id
            JOIN blog_posts p ON m.post_id = p.id
            WHERE m.id = ?
            """,
            (match_id,),
        ).fetchone()
    if row is None:
        return {"error": f"No match with id {match_id}. Use list_matches to find valid ids."}
    return dict(row)


@mcp.tool(annotations=READ_ONLY)
def matches_for_image(image_id: str, include_rejected: bool = False) -> dict:
    """Every blog post scored against one image, best first. Approved only unless include_rejected is true."""
    with get_conn() as conn:
        image = conn.execute(
            "SELECT id, filepath, category, caption, conf_category, conf_caption "
            "FROM images WHERE id = ?",
            (image_id,),
        ).fetchone()
        if image is None:
            return {"error": f"No image with id {image_id}."}
        query = """
            SELECT m.id, m.post_id, m.similarity, m.guard_decision, m.guard_reason,
                   m.human_decision, p.title AS post_title
            FROM matches m JOIN blog_posts p ON m.post_id = p.id
            WHERE m.image_id = ?
        """
        if not include_rejected:
            query += " AND m.guard_decision = 'approved'"
        query += " ORDER BY m.similarity DESC"
        rows = conn.execute(query, (image_id,)).fetchall()
    return {"image": dict(image), "matches": [dict(r) for r in rows]}


@mcp.tool(annotations=WRITES)
def review_match(match_id: str, decision: Literal["approved", "rejected"], note: str) -> dict:
    """Record a human review decision on a match. This WRITES to the database.

    Only call this when a person has actually made the decision. The guard's
    original decision and reason are never changed; the review is stored
    alongside them. A non-empty note explaining the decision is required.
    """
    if not note.strip():
        return {"error": "note is required: explain why this decision was made"}
    with get_conn() as conn:
        if conn.execute("SELECT 1 FROM matches WHERE id = ?", (match_id,)).fetchone() is None:
            return {"error": f"No match with id {match_id}."}
        conn.execute(
            "UPDATE matches SET human_decision = ?, human_note = ?, reviewed_at = ? WHERE id = ?",
            (decision, note.strip(), datetime.now(timezone.utc).isoformat(), match_id),
        )
        row = conn.execute(
            "SELECT id, guard_decision, guard_reason, human_decision, human_note, reviewed_at "
            "FROM matches WHERE id = ?",
            (match_id,),
        ).fetchone()
    return dict(row)


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
