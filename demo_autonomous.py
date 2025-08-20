#!/usr/bin/env python3
"""
Autonomous Robot Demo
Quick demonstration of the autonomous navigation system.
"""

import time
import requests
import json


def check_server():
    """Check if server is running."""
    try:
        response = requests.get("http://localhost:5000/collisions")
        return True
    except:
        return False


def demo_autonomous_navigation():
    """Demonstrate autonomous navigation capabilities."""
    print("🤖 Autonomous Robot Navigation Demo")
    print("=" * 40)
    
    if not check_server():
        print("❌ Server not running!")
        print("Please start the server first:")
        print("  python server.py")
        return
    
    print("✅ Server is running")
    
    # Import the controller
    from run_autonomous_tests import SimpleRobotController
    
    controller = SimpleRobotController()
    
    print("\n🎯 Demonstrating autonomous navigation...")
    print("The robot will:")
    print("- Navigate to the NE corner autonomously")
    print("- Avoid obstacles intelligently")
    print("- Minimize collisions")
    print("- Report performance metrics")
    
    input("\nPress Enter to start the demo...")
    
    # Run single navigation test
    result = controller.intelligent_navigation('NE')
    
    print("\n📊 Demo Results:")
    print("-" * 30)
    
    if result.get('success', False):
        print("🎯 SUCCESS! Robot reached the goal!")
    else:
        print("❌ Navigation incomplete")
        if 'error' in result:
            print(f"Error: {result['error']}")
    
    print(f"Collisions: {result.get('collisions', 'unknown')}")
    print(f"Time taken: {result.get('time', 0):.1f} seconds")
    print(f"Navigation attempts: {result.get('attempts', 0)}")
    if 'final_position' in result:
        print(f"Final position: ({result['final_position'][0]:.1f}, {result['final_position'][1]:.1f})")
    
    if 'distance_to_goal' in result:
        print(f"Distance to goal: {result['distance_to_goal']:.1f}")
    
    print(f"\n🧠 Algorithm Performance:")
    collision_rate = result['collisions'] / result['attempts']
    print(f"Collision rate: {collision_rate:.2f} collisions per attempt")
    
    if result['success']:
        efficiency = result['attempts'] / 30 * 100  # 30 is max attempts
        print(f"Navigation efficiency: {100-efficiency:.1f}%")
    
    print("\n🚀 Full Test Suite Available:")
    print("Run 'python run_autonomous_tests.py' for complete Level 1, 2, and 3 tests")
    
    return result


if __name__ == "__main__":
    demo_autonomous_navigation()
