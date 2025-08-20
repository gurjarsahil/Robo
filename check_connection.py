#!/usr/bin/env python3
"""
Quick check for WebSocket connection
"""

import requests
import time

def check_connection():
    print("🔍 Checking simulator connection...")
    
    # Test reset - this will tell us if simulators are connected
    reset_response = requests.post("http://localhost:5000/reset")
    result = reset_response.json()
    print(f"Reset response: {result}")
    
    if "no simulators connected" in result.get('status', '').lower():
        print("❌ No simulators connected")
        print("Please ensure:")
        print("1. The browser is open")
        print("2. index.html is loaded") 
        print("3. WebSocket connection is established")
        return False
    else:
        print("✅ Simulator is connected!")
        return True

def test_movement():
    print("\n🚀 Testing basic movement...")
    
    # Try a simple move command
    move_response = requests.post("http://localhost:5000/move", json={'x': 5, 'z': 5})
    print(f"Move response: {move_response.status_code} - {move_response.json()}")
    
    if move_response.status_code == 200:
        print("✅ Movement command successful!")
        return True
    else:
        print("❌ Movement command failed")
        return False

if __name__ == "__main__":
    if check_connection():
        test_movement()
    else:
        print("\n💡 To fix this:")
        print("1. Make sure Microsoft Edge browser opened")
        print("2. Navigate to the simulator page if it didn't auto-open")
        print("3. Check browser console for WebSocket connection messages")
