#!/usr/bin/env python3

"""
WebSocket Topic Bridge - Enhanced for raw scan data relay
Uses rosbridge to relay topics from Husky to localhost
"""

import rospy
import json
import websocket
import threading
import time
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

class WebSocketBridge:
    def __init__(self):
        # Initialize ROS node on localhost
        rospy.init_node('websocket_bridge', anonymous=True)
        
        # Publishers on localhost network
        self.scan_pub = rospy.Publisher('/husky/scan_raw', LaserScan, queue_size=1)  # Changed to LaserScan
        self.odom_pub = rospy.Publisher('/husky/odom_raw', Odometry, queue_size=1)
        self.person_pub = rospy.Publisher('/person_detections', Float32MultiArray, queue_size=1)
        
        # WebSocket connection to Husky's rosbridge
        self.ws_url = "ws://192.168.131.224:9090"  # Husky's IP with rosbridge port
        self.ws = None
        
        # Setup WebSocket connection
        self.setup_websocket()
        rospy.loginfo("WebSocket bridge initialized for raw scan data")

    def setup_websocket(self):
        """Setup WebSocket connection to Husky's rosbridge"""
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # Start WebSocket in separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            rospy.loginfo(f"Connecting to rosbridge at {self.ws_url}")
            
        except Exception as e:
            rospy.logerr(f"Failed to setup WebSocket: {e}")

    def on_open(self, ws):
        """WebSocket connection opened"""
        rospy.loginfo("WebSocket connection established")
        
        # Subscribe to Husky topics via rosbridge
        topics_to_subscribe = [
            "/husky/scan_raw",      # Changed from /husky/lidar_raw
            "/husky/odom_raw",
            "/person_detections"
        ]
        
        for topic in topics_to_subscribe:
            subscribe_msg = {
                "op": "subscribe",
                "topic": topic
            }
            ws.send(json.dumps(subscribe_msg))
            rospy.loginfo(f"Subscribed to {topic} via WebSocket")

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            if data.get("op") == "publish":
                topic = data.get("topic")
                msg_data = data.get("msg")
                
                if topic == "/husky/scan_raw":
                    self.relay_scan_data(msg_data)  # New method for LaserScan
                elif topic == "/husky/odom_raw":
                    self.relay_odom_data(msg_data)
                elif topic == "/person_detections":
                    self.relay_person_data(msg_data)
                    
        except Exception as e:
            rospy.logwarn(f"Error processing WebSocket message: {e}")

    def relay_scan_data(self, msg_data):
        """Relay LaserScan data to localhost"""
        try:
            scan_msg = LaserScan()
            
            # Reconstruct LaserScan message from JSON
            header = msg_data.get("header", {})
            scan_msg.header.stamp = rospy.Time.now()  # Use current time
            scan_msg.header.frame_id = header.get("frame_id", "laser_link")
            
            # LaserScan parameters
            scan_msg.angle_min = msg_data.get("angle_min", -3.14159)
            scan_msg.angle_max = msg_data.get("angle_max", 3.14159)
            scan_msg.angle_increment = msg_data.get("angle_increment", 0.00436)
            scan_msg.time_increment = msg_data.get("time_increment", 0.0)
            scan_msg.scan_time = msg_data.get("scan_time", 0.1)
            scan_msg.range_min = msg_data.get("range_min", 0.1)
            scan_msg.range_max = msg_data.get("range_max", 20.0)
            
            # Range data
            scan_msg.ranges = msg_data.get("ranges", [])
            scan_msg.intensities = msg_data.get("intensities", [])
            
            # Publish to laptop
            self.scan_pub.publish(scan_msg)
            rospy.loginfo_throttle(2, f"Relayed scan with {len(scan_msg.ranges)} points via WebSocket")
            
        except Exception as e:
            rospy.logwarn(f"Error relaying scan data: {e}")

    def relay_odom_data(self, msg_data):
        """Relay odometry data to localhost"""
        try:
            odom_msg = Odometry()
            
            # Reconstruct odometry message from JSON
            pose_data = msg_data.get("pose", {}).get("pose", {})
            position = pose_data.get("position", {})
            orientation = pose_data.get("orientation", {})
            
            odom_msg.pose.pose.position.x = position.get("x", 0.0)
            odom_msg.pose.pose.position.y = position.get("y", 0.0)
            odom_msg.pose.pose.position.z = position.get("z", 0.0)
            
            odom_msg.pose.pose.orientation.x = orientation.get("x", 0.0)
            odom_msg.pose.pose.orientation.y = orientation.get("y", 0.0)
            odom_msg.pose.pose.orientation.z = orientation.get("z", 0.0)
            odom_msg.pose.pose.orientation.w = orientation.get("w", 1.0)
            
            # Set header
            header = msg_data.get("header", {})
            odom_msg.header.frame_id = header.get("frame_id", "odom")
            odom_msg.header.stamp = rospy.Time.now()
            
            self.odom_pub.publish(odom_msg)
            rospy.loginfo_throttle(2, "Relayed odometry data via WebSocket")
            
        except Exception as e:
            rospy.logwarn(f"Error relaying odometry data: {e}")

    def relay_person_data(self, msg_data):
        """Relay person detection data to localhost"""
        try:
            person_msg = Float32MultiArray()
            person_msg.data = msg_data.get("data", [])
            
            self.person_pub.publish(person_msg)
            rospy.loginfo_throttle(2, "Relayed person data via WebSocket")
            
        except Exception as e:
            rospy.logwarn(f"Error relaying person data: {e}")

    def on_error(self, ws, error):
        """WebSocket error handler"""
        rospy.logerr(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """WebSocket connection closed"""
        rospy.logwarn("WebSocket connection closed")

    def run(self):
        """Keep bridge running"""
        rospy.loginfo("WebSocket bridge running for raw scan data...")
        rospy.spin()

def main():
    try:
        bridge = WebSocketBridge()
        bridge.run()
    except Exception as e:
        rospy.logerr(f"Bridge error: {e}")

if __name__ == '__main__':
    main()
