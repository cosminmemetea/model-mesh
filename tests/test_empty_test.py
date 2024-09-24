import pytest

@pytest.mark.parametrize("model", [
    "bert", "roberta", "distilbert"
])
def test_empty_text(client, model):
    response = client.post(
        "/predict",
        json={"text": "", "model": model}
    )
    assert response.status_code == 200
    # Extract the response to ensure the structure is correct
    response_data = response.json()
    result = response_data.get("result", {})  
    assert result is not None  # Ensure that the result is not empty
    assert "label" in result  # Ensure that the label field exists
    assert "score" in result  # Ensure that the score field exists
    assert response_data["model_used"] == model  # Ensure the model used is correct
