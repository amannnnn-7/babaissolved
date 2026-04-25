from fastapi.testclient import TestClient

from baba_rlvr.server.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_levels_listed():
    r = client.get("/levels")
    assert "tutorial_01" in r.json()


def test_full_episode_random():
    r = client.post("/reset", json={"level_id": "tutorial_01"})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    obs = r.json()["observation"]
    assert obs["step_count"] == 0
    # Walk right a few times.
    for _ in range(7):
        s = client.post(f"/step/{sid}", json={"action": "right"})
        assert s.status_code == 200
        if s.json()["done"]:
            break
    c = client.post(f"/close/{sid}")
    assert c.status_code == 200
    assert "WIN" in c.json()["milestones"]


def test_invalid_session():
    r = client.post("/step/not-a-session", json={"action": "up"})
    assert r.status_code == 404


def test_play_page_served():
    r = client.get("/play")
    assert r.status_code == 200
    assert "Baba Is You" in r.text


def test_play_frame_png():
    r = client.post("/reset", json={"level_id": "tutorial_01"})
    sid = r.json()["session_id"]
    r = client.get(f"/play/frame/{sid}.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_play_solve_returns_actions():
    r = client.get("/play/solve/tutorial_01")
    data = r.json()
    assert data["actions"] is not None
    assert len(data["actions"]) <= 10
