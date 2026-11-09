#!/usr/bin/env python3

"""
Husky Sensor Publisher - Raw scan forwarding, front 180° only
Runs on Husky - crops Velodyne to ±90° (same FOV as simulation)
"""

import math
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from lidar_front import crop_scan_to_front

class HuskySensorPublisher:
    def __init__(self):
        rospy.init_node('husky_sensor_publisher', anonymous=True)
        
        self.scan_pub = rospy.Publisher('/husky/scan_raw', LaserScan, queue_size=1)
        self.odom_pub = rospy.Publisher('/husky/odom_raw', Odometry, queue_size=1)
        
        rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber('/odometry/filtered_odom', Odometry, self.odom_callback, queue_size=1)
        
        rospy.loginfo("Husky forwarding front-180° scan (sim FOV) and odom")

    def scan_callback(self, scan_msg):
        """Crop to front ±90° and forward as a proper LaserScan."""
        try:
            cropped = crop_scan_to_front(
                scan_msg.ranges,
                scan_msg.angle_min,
                scan_msg.angle_increment,
                intensities=scan_msg.intensities,
            )
            if cropped is None:
                rospy.logwarn_throttle(2.0, "No front-180° LiDAR returns; skipping scan")
                return

            out = LaserScan()
            out.header.stamp = rospy.Time.now()
            out.header.frame_id = scan_msg.header.frame_id
            out.angle_min = cropped['angle_min']
            out.angle_max = cropped['angle_max']
            out.angle_increment = cropped['angle_increment']
            out.time_increment = scan_msg.time_increment
            out.scan_time = scan_msg.scan_time
            out.range_min = scan_msg.range_min
            out.range_max = scan_msg.range_max
            out.ranges = cropped['ranges'].tolist()
            if cropped['intensities'] is not None:
                out.intensities = cropped['intensities'].tolist()

            self.scan_pub.publish(out)
            rospy.loginfo_throttle(
                2, f"Forwarded front-180° scan: {len(out.ranges)} points, "
                   f"angles [{math.degrees(out.angle_min):.0f}°, {math.degrees(out.angle_max):.0f}°]"
            )
        except Exception as e:
            rospy.logwarn(f"Error forwarding scan: {e}")

    def odom_callback(self, odom_msg):
        try:
            self.odom_pub.publish(odom_msg)
        except Exception as e:
            rospy.logwarn(f"Error forwarding odometry: {e}")

    def run(self):
        rospy.loginfo("Husky acting as sensor platform - front 180° LiDAR")
        rospy.spin()

if __name__ == '__main__':
    try:
        publisher = HuskySensorPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass
