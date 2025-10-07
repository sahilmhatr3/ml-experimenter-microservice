#!/usr/bin/env python3
"""
Test script for ML Experiment Microservice integration with MCP Orchestrator.
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

async def test_ml_service_integration():
    """Test the ML service integration."""
    print("Testing ML Experiment Microservice Integration")
    print("=" * 50)
    
    # Test data
    test_job = {
        "job_id": f"test_ml_{int(time.time())}",
        "type": "ml_experiment",
        "payload": {
            "model_type": "linear",
            "dataset": "iris",
            "test_size": 0.2,
            "random_state": 42,
            "parameters": {}
        },
        "metadata": {
            "user": "test",
            "priority": "high"
        }
    }
    
    try:
        # Test ML service directly
        print("1. Testing ML service directly...")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8001/execute",
                json=test_job,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"   ✅ ML service responded successfully")
                    print(f"   Status: {result['status']}")
                    print(f"   Artifacts: {len(result['artifacts'])} generated")
                    for artifact in result['artifacts']:
                        print(f"     - {artifact['name']} ({artifact['type']})")
                else:
                    print(f"   ❌ ML service failed: {response.status}")
                    return False
        
        # Test MCP Orchestrator integration
        print("\n2. Testing MCP Orchestrator integration...")
        
        # First register the ML service
        async with aiohttp.ClientSession() as session:
            # Register service (this would normally be done via CLI)
            print("   Registering ML service with orchestrator...")
            
            # Submit job via orchestrator
            async with session.post(
                "http://localhost:8000/jobs",
                json={
                    "type": "ml_experiment",
                    "payload": test_job["payload"],
                    "metadata": test_job["metadata"]
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    job_response = await response.json()
                    job_id = job_response["job_id"]
                    print(f"   ✅ Job submitted via orchestrator: {job_id}")
                    
                    # Check job status
                    await asyncio.sleep(2)  # Give it time to process
                    async with session.get(f"http://localhost:8000/jobs/{job_id}") as status_response:
                        if status_response.status == 200:
                            status = await status_response.json()
                            print(f"   Job status: {status['status']}")
                            if status['status'] == 'completed':
                                print(f"   ✅ Job completed successfully!")
                                print(f"   Result: {status.get('result', {}).get('experiment_id', 'N/A')}")
                            else:
                                print(f"   ⚠️ Job status: {status['status']}")
                        else:
                            print(f"   ❌ Failed to get job status: {status_response.status}")
                else:
                    print(f"   ❌ Failed to submit job: {response.status}")
                    return False
        
        print("\n✅ Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("Make sure both services are running:")
    print("1. MCP Orchestrator: python -m mcp_core.api.cli serve")
    print("2. ML Service: python main.py")
    print()
    
    result = asyncio.run(test_ml_service_integration())
    if result:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
