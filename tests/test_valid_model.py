import pytest

@pytest.mark.parametrize("model, text", [
    ("bert", "I love this!"),
    ("roberta", "This is terrible."),
    ("distilbert", "This is a neutral statement.")
])
def test_valid_models(client, model, text):
    response = client.post(
        "/predict",
        json={"text": text, "model": model}
    )
    assert response.status_code == 200
    # Print the full response to inspect its structure
    response_data = response.json()
    print(f"Full response data: {response_data}")  # Debugging output
    result = response_data.get("result", {})  
    assert result is not None
    assert "label" in result  # Ensure that the label field exists
    assert "score" in result  # Ensure that the score field exists
    assert isinstance(result["score"], float)  # Check that the score is a float
    assert response_data["model_used"] == model  # Ensure the correct model was used
