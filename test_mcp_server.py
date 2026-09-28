"""
test_mcp_server.py — deterministic tests for the MCP tool layer.

Runs against a throwaway SQLite database built in a temp folder, so no
real pipeline run, API key, or network is needed.  Run: pytest -q
"""
import sqlite3

import pytest

import mcp_server


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "pipeline.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE images(id TEXT PRIMARY KEY, filepath TEXT, category TEXT, caption TEXT,
                            conf_category REAL, conf_caption REAL, embedding TEXT, created_at TEXT);
        CREATE TABLE blog_posts(id TEXT PRIMARY KEY, title TEXT, body TEXT, embedding TEXT,
                                created_at TEXT, expected_category TEXT);
        CREATE TABLE matches(id TEXT PRIMARY KEY, image_id TEXT, post_id TEXT, similarity REAL,
                             guard_decision TEXT, guard_reason TEXT, created_at TEXT,
                             human_decision TEXT, human_note TEXT, reviewed_at TEXT);
        INSERT INTO images VALUES ('img1','data/raw/wolf/0001.jpg','wolf','A grey wolf in snow.',0.95,0.9,NULL,NULL);
        INSERT INTO blog_posts VALUES ('p1','Wolves in winter','',NULL,NULL,'wolf'),
                                      ('p2','Red fox dens','',NULL,NULL,'fox');
        INSERT INTO matches VALUES
          ('m1','img1','p1',0.41,'approved','similarity ok',NULL,NULL,NULL,NULL),
          ('m2','img1','p2',0.52,'rejected','category mismatch: image=wolf, post expects=fox',NULL,NULL,NULL,NULL);
    """)
    conn.commit()
    conn.close()
    monkeypatch.setattr(mcp_server, "DB_PATH", path)
    return path


# --- check_pair: pure guard logic, no database ---

def test_check_pair_rejects_cross_category_even_with_higher_similarity():
    # The wolf-vs-fox bug from EVIDENCE.md: must be rejected before similarity matters.
    r = mcp_server.check_pair("wolf", 0.9, 0.9, "fox", similarity=0.99)
    assert r["decision"] == "rejected" and "category mismatch" in r["reason"]

def test_check_pair_approves_above_threshold():
    assert mcp_server.check_pair("wolf", 0.9, 0.9, "wolf", similarity=0.30)["decision"] == "approved"

def test_check_pair_asks_for_similarity_when_needed():
    assert "error" in mcp_server.check_pair("wolf", 0.9, 0.9, "wolf")

def test_check_pair_validates_inputs():
    assert "error" in mcp_server.check_pair("cat", 0.9, 0.9, "wolf", 0.5)
    assert "error" in mcp_server.check_pair("wolf", 1.5, 0.9, "wolf", 0.5)


# --- database-backed tools ---

def test_list_matches_orders_and_filters(db):
    rows = mcp_server.list_matches()
    assert [r["id"] for r in rows] == ["m2", "m1"]          # highest similarity first
    assert [r["id"] for r in mcp_server.list_matches(guard_decision="approved")] == ["m1"]

def test_get_match_unknown_id_returns_error_not_exception(db):
    assert "error" in mcp_server.get_match("nope")

def test_matches_for_image_hides_rejected_by_default(db):
    assert [m["id"] for m in mcp_server.matches_for_image("img1")["matches"]] == ["m1"]
    assert len(mcp_server.matches_for_image("img1", include_rejected=True)["matches"]) == 2

def test_review_requires_note(db):
    assert "error" in mcp_server.review_match("m2", "approved", "   ")
    assert mcp_server.get_match("m2")["human_decision"] is None

def test_review_never_overwrites_guard_decision(db):
    r = mcp_server.review_match("m2", "approved", "Human override for testing")
    assert r["human_decision"] == "approved"
    assert r["guard_decision"] == "rejected"               # audit trail intact
    assert mcp_server.get_match("m2")["human_note"] == "Human override for testing"

def test_missing_database_gives_clear_message(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "DB_PATH", tmp_path / "missing.db")
    with pytest.raises(FileNotFoundError, match="Run the pipeline first"):
        mcp_server.list_matches()
