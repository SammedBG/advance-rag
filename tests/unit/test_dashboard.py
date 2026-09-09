from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_dashboard_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Advanced RAG" in response.text
    assert "Enterprise AI Knowledge Hub" in response.text


def test_dashboard_direct_endpoint():
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "RAG & Agent Playground" in response.text


def test_static_css_and_js_served():
    css_resp = client.get("/static/style.css")
    assert css_resp.status_code == 200
    assert "--bg-base" in css_resp.text

    js_resp = client.get("/static/app.js")
    assert js_resp.status_code == 200
    assert "ADVANCED RAG PLATFORM" in js_resp.text
