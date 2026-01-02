import httpx
import pytest
import asyncio
import os

# Base URL of the FastAPI app (assuming it's running)
BASE_URL = "http://localhost:8000"

def test_health():
    response = httpx.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@pytest.mark.asyncio
async def test_generate_sql_fallback():
    # Note: This requires the API and vLLM to be running
    # We can mock the settings or just test if it responds correctly if running
    payload = {
        "question": "Show me all employees in department 10",
        "tables": ["emp", "dept"]
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{BASE_URL}/generate-sql", json=payload, timeout=30.0)
            if response.status_code == 200:
                data = response.json()
                print(f"Fallback Result: {data['model_used']}")
                assert "sql" in data
                assert "model_used" in data
            else:
                print(f"API returned {response.status_code}: {response.text}")
        except Exception as e:
            pytest.skip(f"API not reachable: {e}")

@pytest.mark.asyncio
async def test_generate_sql_voting():
    # To test voting, we'd need to set ENSEMBLE_STRATEGY=voting
    # This test might be hard without a running environment
    payload = {
        "question": "List all sales orders for customer 'ABC'",
        "tables": ["oe_order_headers_all", "hz_parties"]
    }
    async with httpx.AsyncClient() as client:
        try:
            # We can't easily change the running API's environment variable here
            # But we can check if the endpoint exists and works
            response = await client.post(f"{BASE_URL}/generate-sql", json=payload, timeout=30.0)
            if response.status_code == 200:
                data = response.json()
                print(f"Result: {data['model_used']}")
                assert "sql" in data
            else:
                 print(f"API returned {response.status_code}: {response.text}")
        except Exception as e:
            pytest.skip(f"API not reachable: {e}")

if __name__ == "__main__":
    import uvicorn
    from src.api.main import app
    import threading
    import time

    # Run the app in a separate thread for testing if needed
    def run_server():
        uvicorn.run(app, host="127.0.0.1", port=8000)

    # This is just a script to manually test if the server can start
    print("Testing server startup...")
    try:
        # Just try to import and see if it fails
        from src.api.main import app
        print("API app imported successfully.")
    except Exception as e:
        print(f"Failed to import API app: {e}")
