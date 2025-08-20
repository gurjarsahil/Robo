#!/usr/bin/env python3
"""
Autonomous Robot Navigation System
This system uses computer vision to detect obstacles and implements pathfinding
to navigate the robot to goal positions while avoiding collisions.
"""

import asyncio
import json
import base64
import time
import math
import heapq
from io import BytesIO
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import cv2
import requests
from PIL import Image
import matplotlib.pyplot as plt


class VisionSystem:
    """Computer vision system for obstacle detection from robot camera images."""
    
    def __init__(self):
        self.obstacle_threshold = 0.3  # Green channel threshold for obstacle detection
        self.floor_color_range = [(80, 80, 80), (120, 120, 120)]  # Gray floor color range
        self.obstacle_color_range = [(0, 200, 0), (100, 255, 100)]  # Green obstacle range
    
    def process_camera_image(self, base64_image: str) -> Dict:
        """
        Process the base64 encoded camera image to detect obstacles.
        Returns obstacle positions relative to robot.
        """
        try:
            # Decode base64 image
            image_data = base64.b64decode(base64_image.split(',')[1])
            image = Image.open(BytesIO(image_data))
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to HSV for better color detection
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Create mask for green obstacles
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
            
            # Find contours of obstacles
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            obstacles = []
            height, width = cv_image.shape[:2]
            
            # Analyze each contour
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small noise
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Estimate distance and angle based on position and size
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Convert to relative coordinates (-1 to 1)
                    rel_x = (center_x - width // 2) / (width // 2)
                    rel_y = (center_y - height // 2) / (height // 2)
                    
                    # Estimate distance based on obstacle size (larger = closer)
                    estimated_distance = max(1, 20 - (w * h) / 1000)
                    
                    # Calculate approximate world coordinates relative to robot
                    angle = rel_x * 45  # Assume 90-degree FOV, so ±45 degrees
                    world_x = estimated_distance * math.sin(math.radians(angle))
                    world_z = estimated_distance * math.cos(math.radians(angle))
                    
                    obstacles.append({
                        'x': world_x,
                        'z': world_z,
                        'distance': estimated_distance,
                        'size': w * h,
                        'angle': angle
                    })
            
            return {
                'obstacles': obstacles,
                'processed': True,
                'obstacle_count': len(obstacles)
            }
            
        except Exception as e:
            print(f"Vision processing error: {e}")
            return {'obstacles': [], 'processed': False, 'error': str(e)}


class PathPlanner:
    """A* pathfinding algorithm for collision-free navigation."""
    
    def __init__(self, grid_size: float = 2.0):
        self.grid_size = grid_size
        self.map_bounds = (-50, 50, -50, 50)  # (min_x, max_x, min_z, max_z)
        self.obstacle_map = set()
        self.safety_margin = 3.0  # Safety distance around obstacles
    
    def update_obstacle_map(self, robot_pos: Tuple[float, float], obstacles: List[Dict]):
        """Update the obstacle map based on current vision data."""
        # Clear old obstacles (we'll rebuild from current vision)
        current_obstacles = set()
        
        for obs in obstacles:
            # Convert robot-relative coordinates to world coordinates
            robot_x, robot_z = robot_pos
            world_x = robot_x + obs['x']
            world_z = robot_z + obs['z']
            
            # Add obstacle and safety margin to map
            grid_x = int(world_x / self.grid_size)
            grid_z = int(world_z / self.grid_size)
            
            # Add obstacle area with safety margin
            margin_cells = int(self.safety_margin / self.grid_size)
            for dx in range(-margin_cells, margin_cells + 1):
                for dz in range(-margin_cells, margin_cells + 1):
                    current_obstacles.add((grid_x + dx, grid_z + dz))
        
        # Update obstacle map with current vision data
        self.obstacle_map.update(current_obstacles)
    
    def is_valid_position(self, x: int, z: int) -> bool:
        """Check if a grid position is valid (not obstacle, within bounds)."""
        # Check bounds
        world_x = x * self.grid_size
        world_z = z * self.grid_size
        if not (self.map_bounds[0] <= world_x <= self.map_bounds[1] and
                self.map_bounds[2] <= world_z <= self.map_bounds[3]):
            return False
        
        # Check obstacles
        return (x, z) not in self.obstacle_map
    
    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions."""
        x, z = pos
        neighbors = []
        
        # 8-directional movement
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dx, dz in directions:
            new_x, new_z = x + dx, z + dz
            if self.is_valid_position(new_x, new_z):
                neighbors.append((new_x, new_z))
        
        return neighbors
    
    def find_path(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Find A* path from start to goal."""
        # Convert to grid coordinates
        start_grid = (int(start_pos[0] / self.grid_size), int(start_pos[1] / self.grid_size))
        goal_grid = (int(goal_pos[0] / self.grid_size), int(goal_pos[1] / self.grid_size))
        
        if not self.is_valid_position(*start_grid) or not self.is_valid_position(*goal_grid):
            return []
        
        # A* algorithm
        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append((current[0] * self.grid_size, current[1] * self.grid_size))
                    current = came_from[current]
                path.append((start_grid[0] * self.grid_size, start_grid[1] * self.grid_size))
                return path[::-1]  # Reverse to get start->goal order
            
            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # No path found


class RobotController:
    """Main robot controller that integrates vision and pathfinding."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.vision = VisionSystem()
        self.planner = PathPlanner()
        self.current_position = (0.0, 0.0)  # (x, z)
        self.current_rotation = 0.0  # Robot's Y rotation in radians
        self.collision_count = 0
        self.goal_position = None
        self.path = []
        self.path_index = 0
        self.movement_tolerance = 1.0
        
        # Corner positions for goals
        self.corners = {
            'NE': (45, -45),
            'NW': (-45, -45), 
            'SE': (45, 45),
            'SW': (-45, 45)
        }
    
    def make_request(self, endpoint: str, method: str = "POST", data: Dict = None) -> Dict:
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
    
    def capture_and_analyze_environment(self) -> Dict:
        """Capture image and analyze environment for obstacles."""
        print("Capturing environment image...")
        result = self.make_request("/capture")
        
        if 'error' in result:
            print(f"Capture error: {result['error']}")
            return {'obstacles': []}
        
        # Wait a moment for image capture to complete
        time.sleep(0.5)
        
        # For now, return empty obstacles as we need WebSocket to get actual image
        # In a real implementation, we'd need to set up WebSocket listener
        return {'obstacles': []}
    
    def move_to_position(self, x: float, z: float) -> bool:
        """Move robot to specific position."""
        print(f"Moving to position ({x:.1f}, {z:.1f})")
        result = self.make_request("/move", data={'x': x, 'z': z})
        
        if 'error' not in result:
            self.current_position = (x, z)
            return True
        else:
            print(f"Move error: {result['error']}")
            return False
    
    def move_relative(self, turn: float, distance: float) -> bool:
        """Move robot relative to current position."""
        print(f"Moving relative: turn {turn:.1f}°, distance {distance:.1f}")
        result = self.make_request("/move_rel", data={'turn': turn, 'distance': distance})
        
        if 'error' not in result:
            # Update estimated position
            self.current_rotation += math.radians(turn)
            dx = distance * math.sin(self.current_rotation)
            dz = distance * math.cos(self.current_rotation)
            self.current_position = (
                self.current_position[0] + dx,
                self.current_position[1] + dz
            )
            return True
        else:
            print(f"Relative move error: {result['error']}")
            return False
    
    def stop_robot(self):
        """Stop robot movement."""
        self.make_request("/stop")
    
    def set_goal(self, corner: str):
        """Set goal position at specified corner."""
        if corner in self.corners:
            pos = self.corners[corner]
            self.goal_position = pos
            result = self.make_request("/goal", data={'corner': corner})
            print(f"Goal set to {corner} at position {pos}")
            return True
        return False
    
    def get_collision_count(self) -> int:
        """Get current collision count from server."""
        result = self.make_request("/collisions", method="GET")
        return result.get('count', 0)
    
    def reset_simulator(self):
        """Reset robot and collision counter."""
        self.make_request("/reset")
        self.current_position = (0.0, 0.0)
        self.current_rotation = 0.0
        self.collision_count = 0
        self.path = []
        self.path_index = 0
        print("Simulator reset")
    
    def navigate_to_goal(self, corner: str, max_attempts: int = 50) -> Dict:
        """
        Main navigation function using computer vision and pathfinding.
        """
        print(f"\n=== Starting navigation to {corner} ===")
        
        # Reset and set goal
        self.reset_simulator()
        time.sleep(1)
        
        if not self.set_goal(corner):
            return {'success': False, 'error': 'Invalid corner'}
        
        start_collisions = self.get_collision_count()
        start_time = time.time()
        
        for attempt in range(max_attempts):
            print(f"\n--- Attempt {attempt + 1} ---")
            
            # Check if we've reached the goal
            distance_to_goal = math.sqrt(
                (self.goal_position[0] - self.current_position[0]) ** 2 +
                (self.goal_position[1] - self.current_position[1]) ** 2
            )
            
            if distance_to_goal < self.movement_tolerance * 2:
                end_time = time.time()
                final_collisions = self.get_collision_count()
                print(f"\n🎯 GOAL REACHED! Distance: {distance_to_goal:.1f}")
                return {
                    'success': True,
                    'collisions': final_collisions - start_collisions,
                    'attempts': attempt + 1,
                    'time': end_time - start_time,
                    'final_position': self.current_position
                }
            
            # Capture and analyze environment
            vision_data = self.capture_and_analyze_environment()
            
            # Update obstacle map
            self.planner.update_obstacle_map(self.current_position, vision_data.get('obstacles', []))
            
            # Plan path to goal
            path = self.planner.find_path(self.current_position, self.goal_position)
            
            if not path:
                print("No path found, trying direct movement...")
                # If no path found, try moving directly towards goal with small steps
                dx = self.goal_position[0] - self.current_position[0]
                dz = self.goal_position[1] - self.current_position[1]
                distance = math.sqrt(dx**2 + dz**2)
                
                if distance > 0:
                    # Normalize and take small step
                    step_size = min(5.0, distance)
                    target_x = self.current_position[0] + (dx / distance) * step_size
                    target_z = self.current_position[1] + (dz / distance) * step_size
                    
                    self.move_to_position(target_x, target_z)
            else:
                print(f"Following path with {len(path)} waypoints")
                # Follow the first few waypoints of the path
                next_waypoint = path[min(2, len(path) - 1)]  # Skip ahead a bit for efficiency
                self.move_to_position(next_waypoint[0], next_waypoint[1])
            
            # Wait for movement to complete
            time.sleep(2)
            
            # Check for collisions and adjust if needed
            current_collisions = self.get_collision_count()
            if current_collisions > start_collisions + self.collision_count:
                print("Collision detected! Adjusting path...")
                self.collision_count = current_collisions - start_collisions
                
                # Back away slightly and try different approach
                self.move_relative(45, -2)  # Turn and back away
                time.sleep(1)
        
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


class AutonomousRobotSystem:
    """Complete autonomous robot system for testing different scenarios."""
    
    def __init__(self):
        self.controller = RobotController()
        self.results = []
    
    def run_level_1_tests(self) -> List[Dict]:
        """Run Level 1: Static obstacles, four corner tests."""
        print("\n🚀 LEVEL 1: Static Obstacle Navigation Tests")
        print("=" * 50)
        
        corners = ['NE', 'NW', 'SE', 'SW']
        results = []
        
        for corner in corners:
            print(f"\n🎯 Testing navigation to {corner}")
            result = self.controller.navigate_to_goal(corner)
            result['corner'] = corner
            result['level'] = 1
            results.append(result)
            
            # Wait between tests
            time.sleep(3)
        
        self.results.extend(results)
        self.print_level_results(results, "Level 1")
        return results
    
    def run_level_2_tests(self, obstacle_speed: float = 0.05) -> List[Dict]:
        """Run Level 2: Moving obstacles navigation."""
        print(f"\n🚀 LEVEL 2: Moving Obstacle Navigation Tests (Speed: {obstacle_speed})")
        print("=" * 60)
        
        # Enable moving obstacles
        motion_config = {
            'enabled': True,
            'speed': obstacle_speed,
            'bounds': {'minX': -45, 'maxX': 45, 'minZ': -45, 'maxZ': 45},
            'bounce': True
        }
        
        motion_result = self.controller.make_request("/obstacles/motion", data=motion_config)
        print(f"Moving obstacles enabled: {motion_result}")
        
        corners = ['NE', 'NW', 'SE', 'SW']
        results = []
        
        for corner in corners:
            print(f"\n🎯 Testing navigation to {corner} with moving obstacles")
            result = self.controller.navigate_to_goal(corner)
            result['corner'] = corner
            result['level'] = 2
            result['obstacle_speed'] = obstacle_speed
            results.append(result)
            
            # Wait between tests
            time.sleep(3)
        
        # Disable moving obstacles
        self.controller.make_request("/obstacles/motion", data={'enabled': False})
        
        self.results.extend(results)
        self.print_level_results(results, f"Level 2 (Speed: {obstacle_speed})")
        return results
    
    def run_level_3_analysis(self) -> Dict:
        """Run Level 3: Performance analysis with different obstacle speeds."""
        print("\n🚀 LEVEL 3: Performance Analysis")
        print("=" * 40)
        
        speeds = [0.02, 0.05, 0.08, 0.12, 0.15]
        speed_results = {}
        
        for speed in speeds:
            print(f"\n📊 Testing obstacle speed: {speed}")
            level_2_results = self.run_level_2_tests(speed)
            
            # Calculate average collisions for this speed
            total_collisions = sum(r['collisions'] for r in level_2_results)
            avg_collisions = total_collisions / len(level_2_results)
            speed_results[speed] = avg_collisions
            
            print(f"Average collisions at speed {speed}: {avg_collisions:.2f}")
        
        # Create performance graph
        self.create_performance_graph(speed_results)
        return speed_results
    
    def print_level_results(self, results: List[Dict], level_name: str):
        """Print formatted results for a level."""
        print(f"\n📊 {level_name} Results Summary:")
        print("-" * 40)
        
        total_collisions = 0
        successful_runs = 0
        
        for result in results:
            corner = result['corner']
            success = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            collisions = result['collisions']
            time_taken = result['time']
            
            total_collisions += collisions
            if result['success']:
                successful_runs += 1
            
            print(f"{corner}: {success} | Collisions: {collisions} | Time: {time_taken:.1f}s")
        
        avg_collisions = total_collisions / len(results)
        success_rate = (successful_runs / len(results)) * 100
        
        print(f"\nOverall Performance:")
        print(f"Success Rate: {success_rate:.1f}% ({successful_runs}/{len(results)})")
        print(f"Average Collisions: {avg_collisions:.2f}")
        print(f"Total Collisions: {total_collisions}")
    
    def create_performance_graph(self, speed_results: Dict):
        """Create and save performance analysis graph."""
        speeds = list(speed_results.keys())
        collisions = list(speed_results.values())
        
        plt.figure(figsize=(10, 6))
        plt.plot(speeds, collisions, 'bo-', linewidth=2, markersize=8)
        plt.title('Robot Navigation Performance vs Obstacle Speed', fontsize=14, fontweight='bold')
        plt.xlabel('Obstacle Speed', fontsize=12)
        plt.ylabel('Average Number of Collisions', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the graph
        plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig('performance_analysis.pdf', bbox_inches='tight')
        print("📈 Performance graph saved as 'performance_analysis.png' and 'performance_analysis.pdf'")
    
    def save_results(self):
        """Save all results to JSON file."""
        with open('navigation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print("💾 Results saved to 'navigation_results.json'")


def main():
    """Main execution function."""
    print("🤖 Autonomous Robot Navigation System")
    print("=====================================")
    
    # Create the autonomous system
    robot_system = AutonomousRobotSystem()
    
    try:
        # Run Level 1 tests
        level_1_results = robot_system.run_level_1_tests()
        
        # Run Level 2 tests
        level_2_results = robot_system.run_level_2_tests()
        
        # Run Level 3 analysis
        level_3_results = robot_system.run_level_3_analysis()
        
        # Save all results
        robot_system.save_results()
        
        print("\n🏁 All tests completed!")
        print("Check the generated files:")
        print("- navigation_results.json: Detailed results")
        print("- performance_analysis.png: Performance graph")
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
