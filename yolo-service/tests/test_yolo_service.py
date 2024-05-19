# tests/test_yolo_service.py
import pytest
from yolo_service import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_status(client):
    response = client.get('/status')
    assert response.status_code == 200
    assert response.json['status'] == 'running'

def test_video_feed(client):
    response = client.get('/video_feed')
    assert response.status_code == 200
    assert 'multipart/x-mixed-replace' in response.content_type
