#!/usr/bin/env python3
"""
Simple Autonomous Robot Test Runner
This script runs the autonomous navigation tests without requiring heavy dependencies.
"""

import json
import time
import math
import requests
from typing import Dict, List


class SimpleRobotController:
    """Simplified robot controller for autonomous navigation testing."""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.current_position = (0.0, 0.0)  # (x, z)
        self.goal_position = None
        
        # Corner positions
        self.corners = {
            'NE': (45, -45),
            'NW': (-45, -45),
            'SE': (45, 45),
            'SW': (-45, 45)
        }
    
    def make_request(self, endpoint, method="POST", data=None):
        """Make HTTP request to robot API."""
        try:
            url = f"{self.base_url}{endpoint}"
            if method == "GET":
                response = requests.get(url)
            else:
                response = requests.post(url, json=data, headers={'Content-Type': 'application/json'})
            
            return response.json() if response.content else {}
        except Exception as e:
            print(f"Request error: {e}")
            return {'error': str(e)}
    
    def set_goal(self, corner):
        """Set goal position."""
        if corner in self.corners:
            self.goal_position = self.corners[corner]
            result = self.make_request("/goal", data={'corner': corner})
            print(f"Goal set to {corner} at {self.goal_position}")
            return 'error' not in result
        return False
    
    def reset_simulator(self):
        """Reset simulator."""
        self.make_request("/reset")
        self.current_position = (0.0, 0.0)
        time.sleep(1)
        print("Simulator reset")
    
    def get_collision_count(self):
        """Get collision count."""
        result = self.make_request("/collisions", method="GET")
        return result.get('count', 0)
    
    def move_to_position(self, x, z):
        """Move to position."""
        print(f"Moving to ({x:.1f}, {z:.1f})")
        result = self.make_request("/move", data={'x': x, 'z': z})
        if 'error' not in result:
            self.current_position = (x, z)
            return True
        return False
    
    def move_relative(self, turn, distance):
        """Move relative."""
        print(f"Moving relative: turn {turn:.1f}°, distance {distance:.1f}")
        result = self.make_request("/move_rel", data={'turn': turn, 'distance': distance})
        return 'error' not in result
    
    def intelligent_navigation(self, corner, max_attempts=30):
        """Intelligent navigation using step-by-step approach."""
        print(f"\n🚀 Starting intelligent navigation to {corner}")
        
        self.reset_simulator()
        if not self.set_goal(corner):
            return {'success': False, 'error': 'Invalid corner'}
        
        start_collisions = self.get_collision_count()
        start_time = time.time()
        
        # Strategy: Move in smaller steps, with obstacle avoidance
        for attempt in range(max_attempts):
            print(f"\n--- Attempt {attempt + 1} ---")
            
            # Calculate distance to goal
            dx = self.goal_position[0] - self.current_position[0]
            dz = self.goal_position[1] - self.current_position[1]
            distance_to_goal = math.sqrt(dx*dx + dz*dz)
            
            print(f"Current position: ({self.current_position[0]:.1f}, {self.current_position[1]:.1f})")
            print(f"Distance to goal: {distance_to_goal:.1f}")
            
            # Check if reached goal
            if distance_to_goal < 3.0:
                end_time = time.time()
                final_collisions = self.get_collision_count()
                print(f"\n🎯 GOAL REACHED! Final distance: {distance_to_goal:.1f}")
                return {
                    'success': True,
                    'collisions': final_collisions - start_collisions,
                    'attempts': attempt + 1,
                    'time': end_time - start_time,
                    'final_position': self.current_position
                }
            
            # Record collisions before move
            pre_move_collisions = self.get_collision_count()
            
            # Move strategy: Take smaller steps towards goal
            step_size = min(8.0, distance_to_goal * 0.6)  # 60% of remaining distance
            
            if distance_to_goal > 0:
                # Calculate target position
                direction_x = dx / distance_to_goal
                direction_z = dz / distance_to_goal
                target_x = self.current_position[0] + direction_x * step_size
                target_z = self.current_position[1] + direction_z * step_size
                
                # Try direct movement
                success = self.move_to_position(target_x, target_z)
                
                if success:
                    time.sleep(2)  # Wait for movement
                    
                    # Check for collisions after move
                    post_move_collisions = self.get_collision_count()
                    if post_move_collisions > pre_move_collisions:
                        print(f"💥 Collision detected! Trying evasive maneuvers...")
                        
                        # Evasive maneuver: try different angles
                        evasion_angles = [45, -45, 90, -90, 135, -135]
                        for angle in evasion_angles:
                            print(f"Trying evasion angle: {angle}°")
                            
                            # Small evasive move
                            self.move_relative(angle, 4)
                            time.sleep(1)
                            
                            # Check if this direction is clearer
                            test_collisions = self.get_collision_count()
                            if test_collisions == post_move_collisions:
                                # No new collision, continue in this direction
                                self.move_relative(0, step_size * 0.5)
                                time.sleep(1)
                                break
                    else:
                        print("✅ Move successful, no collisions")
                else:
                    print("❌ Move failed, trying alternative approach")
                    # Try relative movement instead
                    angle = math.degrees(math.atan2(dx, dz))
                    self.move_relative(angle, step_size)
                    time.sleep(2)
            
            # Small delay between attempts
            time.sleep(0.5)
        
        # Failed to reach goal
        end_time = time.time()
        final_collisions = self.get_collision_count()
        return {
            'success': False,
            'collisions': final_collisions - start_collisions,
            'attempts': max_attempts,
            'time': end_time - start_time,
            'final_position': self.current_position,
            'distance_to_goal': distance_to_goal
        }


class SimpleTestRunner:
    """Test runner for autonomous navigation."""
    
    def __init__(self):
        self.controller = SimpleRobotController()
        self.results = []
    
    def run_level_1_tests(self):
        """Run Level 1: Static obstacles navigation."""
        print("\n🚀 LEVEL 1: Static Obstacle Navigation Tests")
        print("=" * 50)
        
        corners = ['NE', 'NW', 'SE', 'SW']
        results = []
        
        for corner in corners:
            print(f"\n🎯 Testing navigation to corner {corner}")
            
            result = self.controller.intelligent_navigation(corner)
            result['corner'] = corner
            result['level'] = 1
            results.append(result)
            
            # Print immediate result
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            print(f"\nResult: {status}")
            print(f"Collisions: {result['collisions']}")
            print(f"Time: {result['time']:.1f}s")
            print(f"Attempts: {result['attempts']}")
            
            # Wait between tests
            time.sleep(2)
        
        self.results.extend(results)
        self.print_summary(results, "Level 1")
        return results
    
    def run_level_2_tests(self, speed=0.05):
        """Run Level 2: Moving obstacles navigation."""
        print(f"\n🚀 LEVEL 2: Moving Obstacle Navigation (Speed: {speed})")
        print("=" * 60)
        
        # Enable moving obstacles
        motion_config = {
            'enabled': True,
            'speed': speed,
            'bounds': {'minX': -45, 'maxX': 45, 'minZ': -45, 'maxZ': 45},
            'bounce': True
        }
        
        result = self.controller.make_request("/obstacles/motion", data=motion_config)
        print(f"Moving obstacles enabled: {result.get('status', 'unknown')}")
        
        corners = ['NE', 'NW', 'SE', 'SW']
        results = []
        
        for corner in corners:
            print(f"\n🎯 Testing navigation to {corner} with moving obstacles")
            
            result = self.controller.intelligent_navigation(corner)
            result['corner'] = corner
            result['level'] = 2
            result['obstacle_speed'] = speed
            results.append(result)
            
            # Print immediate result
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            print(f"\nResult: {status}")
            print(f"Collisions: {result['collisions']}")
            print(f"Time: {result['time']:.1f}s")
            
            time.sleep(2)
        
        # Disable moving obstacles
        self.controller.make_request("/obstacles/motion", data={'enabled': False})
        
        self.results.extend(results)
        self.print_summary(results, f"Level 2 (Speed: {speed})")
        return results
    
    def run_level_3_analysis(self):
        """Run Level 3: Performance analysis."""
        print("\n🚀 LEVEL 3: Performance Analysis")
        print("=" * 40)
        
        speeds = [0.02, 0.05, 0.08, 0.12]  # Reduced for faster testing
        speed_results = {}
        
        for speed in speeds:
            print(f"\n📊 Testing obstacle speed: {speed}")
            level_2_results = self.run_level_2_tests(speed)
            
            # Calculate average collisions
            total_collisions = sum(r['collisions'] for r in level_2_results)
            avg_collisions = total_collisions / len(level_2_results)
            speed_results[speed] = avg_collisions
            
            print(f"Average collisions at speed {speed}: {avg_collisions:.2f}")
        
        # Save speed analysis
        with open('speed_analysis.json', 'w') as f:
            json.dump(speed_results, f, indent=2)
        
        print("\n📈 Speed analysis saved to 'speed_analysis.json'")
        return speed_results
    
    def print_summary(self, results, level_name):
        """Print test summary."""
        print(f"\n📊 {level_name} Summary:")
        print("-" * 40)
        
        total_collisions = 0
        successful_runs = 0
        total_time = 0
        
        for result in results:
            if result['success']:
                successful_runs += 1
            total_collisions += result['collisions']
            total_time += result['time']
        
        success_rate = (successful_runs / len(results)) * 100
        avg_collisions = total_collisions / len(results)
        avg_time = total_time / len(results)
        
        print(f"Success Rate: {success_rate:.1f}% ({successful_runs}/{len(results)})")
        print(f"Average Collisions: {avg_collisions:.2f}")
        print(f"Average Time: {avg_time:.1f}s")
        print(f"Total Collisions: {total_collisions}")
    
    def save_all_results(self):
        """Save all results."""
        with open('autonomous_navigation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print("💾 All results saved to 'autonomous_navigation_results.json'")


def main():
    """Main test execution."""
    print("🤖 Autonomous Robot Navigation Test Suite")
    print("=" * 45)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:5000/collisions")
        print("✅ Server is running")
    except:
        print("❌ Server is not running! Please start server.py first.")
        print("Run: python server.py")
        return
    
    runner = SimpleTestRunner()
    
    try:
        # Run tests
        print("\nStarting autonomous navigation tests...")
        
        # Level 1: Static obstacles
        level_1_results = runner.run_level_1_tests()
        
        # Ask user if they want to continue to Level 2
        continue_tests = input("\nContinue to Level 2 tests? (y/n): ").lower().strip()
        
        if continue_tests == 'y':
            # Level 2: Moving obstacles
            level_2_results = runner.run_level_2_tests()
            
            # Ask about Level 3
            continue_level3 = input("\nContinue to Level 3 analysis? (y/n): ").lower().strip()
            
            if continue_level3 == 'y':
                # Level 3: Performance analysis
                level_3_results = runner.run_level_3_analysis()
        
        # Save all results
        runner.save_all_results()
        
        print("\n🏁 Testing completed!")
        print("Results saved to 'autonomous_navigation_results.json'")
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        runner.save_all_results()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
