import pytest
from fastapi.testclient import TestClient
from api.fastapi_app import app

@pytest.fixture(scope="module")
def client():
    # Return a FastAPI test client for use in tests
    return TestClient(app)
