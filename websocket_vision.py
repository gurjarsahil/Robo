#!/usr/bin/env python3
"""
WebSocket Vision Client
Handles real-time image capture and processing from the robot simulator.
"""

import asyncio
import websockets
import json
import base64
import threading
import queue
import time
from autonomous_robot import VisionSystem


class WebSocketVisionClient:
    """WebSocket client for real-time vision processing."""
    
    def __init__(self, ws_url="ws://localhost:8080"):
        self.ws_url = ws_url
        self.vision_system = VisionSystem()
        self.image_queue = queue.Queue()
        self.latest_vision_data = {'obstacles': []}
        self.connected = False
        self.websocket = None
        
    async def connect_and_listen(self):
        """Connect to WebSocket and listen for messages."""
        try:
            self.websocket = await websockets.connect(self.ws_url)
            self.connected = True
            print(f"✅ Connected to WebSocket: {self.ws_url}")
            
            # Send initial connection message
            await self.websocket.send(json.dumps({
                "type": "connection",
                "message": "Autonomous vision client connected"
            }))
            
            # Listen for messages
            async for message in self.websocket:
                await self.handle_message(message)
                
        except Exception as e:
            print(f"❌ WebSocket connection error: {e}")
            self.connected = False
    
    async def handle_message(self, message):
        """Process incoming WebSocket messages."""
        try:
            data = json.loads(message)
            message_type = data.get('type', '')
            
            if message_type == 'capture_image_response':
                # Process the captured image
                image_data = data.get('image', '')
                position = data.get('position', {})
                
                if image_data:
                    # Process image with vision system
                    vision_result = self.vision_system.process_camera_image(image_data)
                    vision_result['robot_position'] = position
                    vision_result['timestamp'] = data.get('timestamp', time.time())
                    
                    # Update latest vision data
                    self.latest_vision_data = vision_result
                    
                    print(f"🔍 Vision processed: {len(vision_result['obstacles'])} obstacles detected")
            
            elif message_type == 'collision':
                print(f"💥 Collision detected at {data.get('position', 'unknown')}")
            
            elif message_type == 'goal_reached':
                print(f"🎯 Goal reached at {data.get('position', 'unknown')}")
                
        except Exception as e:
            print(f"❌ Error processing WebSocket message: {e}")
    
    async def request_image_capture(self):
        """Request image capture from simulator."""
        if self.websocket and self.connected:
            try:
                await self.websocket.send(json.dumps({
                    "command": "capture_image"
                }))
                return True
            except Exception as e:
                print(f"❌ Error requesting image capture: {e}")
                return False
        return False
    
    def get_latest_vision_data(self):
        """Get the most recent vision processing results."""
        return self.latest_vision_data.copy()
    
    def start_background_thread(self):
        """Start WebSocket client in background thread."""
        def run_websocket():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.connect_and_listen())
        
        thread = threading.Thread(target=run_websocket, daemon=True)
        thread.start()
        return thread


class EnhancedRobotController:
    """Enhanced robot controller with WebSocket vision integration."""
    
    def __init__(self, base_url="http://localhost:5000", ws_url="ws://localhost:8080"):
        # Import the base controller
        from autonomous_robot import RobotController
        
        # Initialize base controller
        self.base_controller = RobotController(base_url)
        
        # Initialize WebSocket vision client
        self.ws_client = WebSocketVisionClient(ws_url)
        self.ws_thread = None
        
        # Start WebSocket connection
        self.start_websocket_client()
    
    def start_websocket_client(self):
        """Start WebSocket client for real-time vision."""
        self.ws_thread = self.ws_client.start_background_thread()
        time.sleep(2)  # Wait for connection
    
    def capture_and_analyze_environment(self):
        """Enhanced capture with real-time vision processing."""
        print("🔍 Capturing and analyzing environment...")
        
        # Request image capture via HTTP API
        result = self.base_controller.make_request("/capture")
        
        if 'error' in result:
            print(f"Capture request error: {result['error']}")
            return {'obstacles': []}
        
        # Wait for WebSocket to receive and process the image
        time.sleep(1.5)  # Allow time for image processing
        
        # Get processed vision data from WebSocket client
        vision_data = self.ws_client.get_latest_vision_data()
        
        if vision_data.get('processed', False):
            obstacles = vision_data.get('obstacles', [])
            print(f"✅ Vision analysis complete: {len(obstacles)} obstacles detected")
            
            # Debug: print obstacle information
            for i, obs in enumerate(obstacles):
                print(f"  Obstacle {i+1}: distance={obs['distance']:.1f}, angle={obs['angle']:.1f}°")
            
            return vision_data
        else:
            print("⚠️ No vision data available, using fallback obstacle detection")
            return {'obstacles': []}
    
    def navigate_to_goal(self, corner, max_attempts=50):
        """Navigate using enhanced vision system."""
        print(f"\n🚀 Enhanced navigation to {corner}")
        
        # Use base controller logic but with enhanced vision
        self.base_controller.reset_simulator()
        time.sleep(1)
        
        if not self.base_controller.set_goal(corner):
            return {'success': False, 'error': 'Invalid corner'}
        
        start_collisions = self.base_controller.get_collision_count()
        start_time = time.time()
        
        for attempt in range(max_attempts):
            print(f"\n--- Enhanced Attempt {attempt + 1} ---")
            
            # Check if we've reached the goal
            distance_to_goal = math.sqrt(
                (self.base_controller.goal_position[0] - self.base_controller.current_position[0]) ** 2 +
                (self.base_controller.goal_position[1] - self.base_controller.current_position[1]) ** 2
            )
            
            if distance_to_goal < self.base_controller.movement_tolerance * 2:
                end_time = time.time()
                final_collisions = self.base_controller.get_collision_count()
                print(f"\n🎯 GOAL REACHED! Distance: {distance_to_goal:.1f}")
                return {
                    'success': True,
                    'collisions': final_collisions - start_collisions,
                    'attempts': attempt + 1,
                    'time': end_time - start_time,
                    'final_position': self.base_controller.current_position
                }
            
            # Enhanced capture and analysis
            vision_data = self.capture_and_analyze_environment()
            
            # Update obstacle map with enhanced vision data
            self.base_controller.planner.update_obstacle_map(
                self.base_controller.current_position, 
                vision_data.get('obstacles', [])
            )
            
            # Plan path to goal
            path = self.base_controller.planner.find_path(
                self.base_controller.current_position, 
                self.base_controller.goal_position
            )
            
            if not path:
                print("No clear path found, using direct approach with obstacle avoidance...")
                
                # If obstacles are detected in vision, try to avoid them
                obstacles = vision_data.get('obstacles', [])
                if obstacles:
                    # Find the clearest direction
                    best_angle = 0
                    min_obstacle_density = float('inf')
                    
                    for test_angle in range(-90, 91, 30):  # Test angles from -90 to 90 degrees
                        obstacle_density = 0
                        for obs in obstacles:
                            angle_diff = abs(obs['angle'] - test_angle)
                            if angle_diff < 45:  # Within 45 degrees
                                obstacle_density += 1.0 / max(obs['distance'], 0.1)
                        
                        if obstacle_density < min_obstacle_density:
                            min_obstacle_density = obstacle_density
                            best_angle = test_angle
                    
                    # Move in the clearest direction
                    move_distance = min(8.0, distance_to_goal / 2)
                    self.base_controller.move_relative(best_angle, move_distance)
                    print(f"Moving {move_distance:.1f} units at angle {best_angle}° (clearest path)")
                else:
                    # No obstacles detected, move directly towards goal
                    dx = self.base_controller.goal_position[0] - self.base_controller.current_position[0]
                    dz = self.base_controller.goal_position[1] - self.base_controller.current_position[1]
                    distance = math.sqrt(dx**2 + dz**2)
                    
                    if distance > 0:
                        step_size = min(8.0, distance)
                        target_x = self.base_controller.current_position[0] + (dx / distance) * step_size
                        target_z = self.base_controller.current_position[1] + (dz / distance) * step_size
                        self.base_controller.move_to_position(target_x, target_z)
            else:
                print(f"Following planned path with {len(path)} waypoints")
                # Follow planned path
                next_waypoint_idx = min(1, len(path) - 1)
                next_waypoint = path[next_waypoint_idx]
                self.base_controller.move_to_position(next_waypoint[0], next_waypoint[1])
            
            # Wait for movement to complete
            time.sleep(2.5)
            
            # Check for collisions and adapt
            current_collisions = self.base_controller.get_collision_count()
            if current_collisions > start_collisions + self.base_controller.collision_count:
                print("💥 Collision detected! Implementing recovery maneuver...")
                self.base_controller.collision_count = current_collisions - start_collisions
                
                # Enhanced collision recovery
                self.base_controller.move_relative(90, -3)  # Turn 90° and back away
                time.sleep(1)
                self.base_controller.move_relative(-45, 2)  # Adjust heading
                time.sleep(1)
        
        # Navigation failed
        end_time = time.time()
        final_collisions = self.base_controller.get_collision_count()
        return {
            'success': False,
            'collisions': final_collisions - start_collisions,
            'attempts': max_attempts,
            'time': end_time - start_time,
            'final_position': self.base_controller.current_position,
            'distance_to_goal': distance_to_goal
        }


# Test the enhanced system
if __name__ == "__main__":
    import math
    
    print("🤖 Enhanced Autonomous Robot System with WebSocket Vision")
    print("=" * 60)
    
    try:
        # Create enhanced controller
        enhanced_controller = EnhancedRobotController()
        
        # Wait for WebSocket connection
        time.sleep(3)
        
        # Test single navigation
        print("\n🧪 Testing enhanced navigation to NE corner...")
        result = enhanced_controller.navigate_to_goal('NE')
        
        print(f"\n📊 Test Result:")
        print(f"Success: {result['success']}")
        print(f"Collisions: {result['collisions']}")
        print(f"Time: {result.get('time', 0):.1f}s")
        print(f"Attempts: {result['attempts']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
