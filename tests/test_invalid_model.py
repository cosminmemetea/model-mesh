def test_invalid_model(client):
    response = client.post(
        "/predict",
        json={"text": "Test text.", "model": "unknown_model"}
    )
    assert response.status_code == 400
    # Assert that the correct error message is included
    assert "detail" in response.json()
