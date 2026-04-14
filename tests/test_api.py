from fastapi.testclient import TestClient

from credexp.serving.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code in (200, 503)


def test_predict_valid_payload():
    payload = {
        "sk_id_curr": 123456,
        "features": {
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_2": 0.7,
        },
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in (200, 503)


def test_predict_rejects_wrong_type():
    payload = {
        "sk_id_curr": 123456,
        "features": "not_a_dict",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_extra_fields():
    payload = {
        "sk_id_curr": 123456,
        "features": {
            "EXT_SOURCE_1": 0.5,
        },
        "unexpected_field": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
