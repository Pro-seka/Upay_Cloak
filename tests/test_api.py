"""Integration test suite for UpayShield FastAPI backend.
Tests both /api/v1 contract endpoints and /api frontend compatibility routes.
"""
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["intelligence"] == "live"


def test_config():
    res = client.get("/api/v1/config")
    assert res.status_code == 200
    data = res.json()
    assert data["contract_version"] == "1.0"
    assert "en" in data["languages"]
    assert "bn" in data["languages"]


def test_frontend_overview():
    res = client.get("/api/overview")
    assert res.status_code == 200
    data = res.json()
    assert "kpis" in data
    assert "dist" in data
    assert "trend" in data
    assert "agents" in data
    assert data["kpis"]["alerts"] > 0


def test_frontend_cases():
    res = client.get("/api/cases")
    assert res.status_code == 200
    cases = res.json()
    assert len(cases) > 0
    c1 = cases[0]
    assert "case_id" in c1
    assert "why_risky" in c1
    assert "subgraph" in c1


def test_frontend_case_detail_and_ask():
    res = client.get("/api/cases/CASE-1001")
    assert res.status_code == 200
    c = res.json()
    assert c["case_id"] == "CASE-1001"
    assert c["alert_type"] == "account_takeover"

    # Test assistant ask
    ask_res = client.post("/api/cases/CASE-1001/ask", json={"q": "Why was this flagged?"})
    assert ask_res.status_code == 200
    ans = ask_res.json()
    assert "text" in ans
    assert len(ans["evidence_ids"]) > 0


def test_frontend_action_application():
    res = client.post("/api/cases/CASE-1001/action", json={"a": "hold"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("open", "investigating")
    assert len(data["audit"]) > 0


def test_graph_and_rings():
    r1 = client.get("/api/v1/graph/rings")
    assert r1.status_code == 200
    assert len(r1.json()) > 0

    r2 = client.get("/api/graph/019XX-XXX305")
    assert r2.status_code == 200
    g = r2.json()
    assert "nodes" in g
    assert "edges" in g


def test_agent_leaderboard():
    res = client.get("/api/v1/agents")
    assert res.status_code == 200
    agents = res.json()
    assert len(agents) > 0
    assert agents[0]["risk_score"] >= 0.5


def test_customer_scam_warning():
    req = {
        "user_id": "017XX-123456",
        "recipient_id": "018XX-999888",
        "amount": 35000.0,
        "language": "en",
    }
    res = client.post("/api/v1/warning/check", json=req)
    assert res.status_code == 200
    w = res.json()
    assert w["show_warning"] is True
    assert "Caution" in w["headline"]
    assert len(w["reasons"]) > 0
