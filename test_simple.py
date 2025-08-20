#!/usr/bin/env python3
"""
Simple test for autonomous navigation
"""

import requests
import time
import math


def test_server():
    """Test if server is working"""
    try:
        response = requests.get("http://localhost:5000/collisions")
        print(f"Server status: {response.status_code}")
        print(f"Response: {response.json()}")
        return True
    except Exception as e:
        print(f"Server error: {e}")
        return False


def simple_autonomous_navigation():
    """Simple autonomous navigation test"""
    print("🤖 Simple Autonomous Navigation Test")
    print("=" * 40)
    
    if not test_server():
        return
    
    # Reset simulator
    print("\n🔄 Resetting simulator...")
    reset_response = requests.post("http://localhost:5000/reset")
    print(f"Reset: {reset_response.json()}")
    time.sleep(1)
    
    # Set goal to NE corner
    print("\n🎯 Setting goal to NE corner...")
    goal_response = requests.post("http://localhost:5000/goal", json={'corner': 'NE'})
    print(f"Goal response: {goal_response.status_code}")
    print(f"Goal data: {goal_response.json()}")
    
    if goal_response.status_code != 200:
        print("❌ Goal setting failed, trying with coordinates...")
        goal_response = requests.post("http://localhost:5000/goal", json={'x': 45, 'z': -45})
        print(f"Goal with coords: {goal_response.json()}")
    
    # Start simple navigation
    print("\n🚀 Starting navigation...")
    current_pos = (0, 0)
    goal_pos = (45, -45)
    
    for attempt in range(10):
        print(f"\n--- Attempt {attempt + 1} ---")
        
        # Calculate distance to goal
        dx = goal_pos[0] - current_pos[0]
        dz = goal_pos[1] - current_pos[1]
        distance = math.sqrt(dx*dx + dz*dz)
        
        print(f"Current position: {current_pos}")
        print(f"Distance to goal: {distance:.1f}")
        
        if distance < 5:
            print("🎯 GOAL REACHED!")
            break
        
        # Take a step toward goal
        step_size = min(10, distance * 0.5)
        direction_x = dx / distance
        direction_z = dz / distance
        
        new_x = current_pos[0] + direction_x * step_size
        new_z = current_pos[1] + direction_z * step_size
        
        # Move to new position
        move_response = requests.post("http://localhost:5000/move", json={'x': new_x, 'z': new_z})
        print(f"Move to ({new_x:.1f}, {new_z:.1f}): {move_response.status_code}")
        
        if move_response.status_code == 200:
            current_pos = (new_x, new_z)
            
        # Wait for movement
        time.sleep(3)
        
        # Check collisions
        collision_response = requests.get("http://localhost:5000/collisions")
        collisions = collision_response.json().get('count', 0)
        print(f"Current collisions: {collisions}")
    
    print("\n📊 Navigation Complete!")
    final_collisions = requests.get("http://localhost:5000/collisions").json().get('count', 0)
    print(f"Final collision count: {final_collisions}")


if __name__ == "__main__":
    simple_autonomous_navigation()
