#!/usr/bin/env python3

"""
ZED Camera 3D Object Detection - Enhanced with Tracking ID Support
Provides precise distance and position measurements with person tracking
"""

import cv2
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
import pyzed.sl as sl
import math
import time

class ZED3DProcessor:
    def __init__(self):
        # Initialize ROS
        rospy.init_node('zed_3d_processor', anonymous=True)
        
        # ROS publisher for person detections (enhanced with tracking IDs)
        self.person_pub = rospy.Publisher('/person_detections', Float32MultiArray, queue_size=1)
        
        # ZED Camera and 3D Object Detection setup
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.detection_params = sl.ObjectDetectionParameters()
        self.runtime_params = sl.ObjectDetectionRuntimeParameters()
        
        # Initialize ZED camera and object detection
        if not self.setup_zed_camera():
            rospy.logfatal("Failed to initialize ZED camera")
            return
        
        # Data storage
        self.image = sl.Mat()
        self.objects = sl.Objects()
        
        rospy.loginfo("ZED 3D Object Detection with Tracking initialized")
        rospy.loginfo("- Native 3D bounding boxes with metric coordinates")
        rospy.loginfo("- Person tracking with unique IDs")
        rospy.loginfo("- Compatible with GP trajectory prediction")

    def setup_zed_camera(self):
        """Setup ZED camera with 3D object detection and tracking"""
        try:
            # Camera initialization parameters - optimized for person detection
            self.init_params.camera_resolution = sl.RESOLUTION.HD720
            self.init_params.camera_fps = 30
            self.init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
            self.init_params.coordinate_units = sl.UNIT.METER
            self.init_params.depth_minimum_distance = 0.3
            self.init_params.depth_maximum_distance = 20.0
            
            # Open the camera
            status = self.zed.open(self.init_params)
            if status != sl.ERROR_CODE.SUCCESS:
                rospy.logerr(f"Failed to open ZED camera: {status}")
                return False
            
            # Enable positional tracking (required for object tracking)
            tracking_params = sl.PositionalTrackingParameters()
            tracking_params.enable_area_memory = False
            
            status = self.zed.enable_positional_tracking(tracking_params)
            if status != sl.ERROR_CODE.SUCCESS:
                rospy.logerr(f"Failed to enable positional tracking: {status}")
                return False
            
            rospy.loginfo("Positional tracking enabled successfully")
            
            # Enable 3D object detection with tracking
            self.detection_params.enable_tracking = True  # CRITICAL: Enable tracking for IDs
            self.detection_params.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_ACCURATE
            self.detection_params.enable_segmentation = False
            self.detection_params.max_range = 20.0
            
            status = self.zed.enable_object_detection(self.detection_params)
            if status != sl.ERROR_CODE.SUCCESS:
                rospy.logerr(f"Failed to enable object detection: {status}")
                return False
            
            # Runtime parameters
            self.runtime_params.detection_confidence_threshold = 50
            self.runtime_params.object_class_filter = [sl.OBJECT_CLASS.PERSON]
            
            rospy.loginfo("ZED camera with tracking initialized successfully")
            return True
            
        except Exception as e:
            rospy.logerr(f"Error setting up ZED camera: {e}")
            return False

    def process_frame(self):
        """Process frame with native ZED 3D object detection and tracking"""
        # Grab a new frame
        status = self.zed.grab()
        if status != sl.ERROR_CODE.SUCCESS:
            return False
        
        # Retrieve the left image for display
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        
        # Proper image conversion to ensure OpenCV compatibility
        image_data = self.image.get_data()
        
        if len(image_data.shape) == 3 and image_data.shape[2] == 4:
            image_cv = cv2.cvtColor(image_data, cv2.COLOR_RGBA2BGR)
        else:
            image_cv = image_data[:, :, :3]
        
        image_cv = np.ascontiguousarray(image_cv, dtype=np.uint8)
        
        # Detect 3D objects
        status = self.zed.retrieve_objects(self.objects, self.runtime_params)
        
        current_frame_detections = []
        
        if status == sl.ERROR_CODE.SUCCESS:
            # Process detected objects
            for obj in self.objects.object_list:
                if obj.label == sl.OBJECT_CLASS.PERSON and obj.confidence > 50:
                    # Get 3D position (in camera coordinates)
                    position = obj.position
                    camera_x = position[0]  # X in camera frame (right is positive)
                    camera_y = position[1]  # Y in camera frame (down is positive) 
                    camera_z = position[2]  # Z in camera frame (forward is positive)
                    
                    # Keep raw camera coordinates for publishing
                    # Camera frame: X=right, Y=down, Z=forward
                    # dt_v5.py will handle conversion to robot frame
                    
                    confidence = obj.confidence / 100.0  # Convert to 0-1 range
                    
                    # IMPORTANT: Get tracking ID
                    tracking_id = obj.id if hasattr(obj, 'id') and obj.id >= 0 else 0
                    
                    # Validate detection using camera coordinates
                    distance = math.sqrt(camera_x**2 + camera_z**2)
                    if 0.5 < distance < 15.0:
                        # Raw camera frame format: (camera_x, camera_z, conf, tracking_id)
                        # camera_x: positive = right of camera
                        # camera_z: positive = forward from camera
                        current_frame_detections.append((camera_x, camera_z, confidence, tracking_id))
                        
                        rospy.loginfo(f"Person ID {tracking_id}: distance={distance:.2f}m, "
                                    f"cam_pos=({camera_x:.2f}, {camera_z:.2f}), "
                                    f"confidence={confidence:.2f}")
                        
                        # Draw enhanced 3D visualization
                        self.draw_3d_detection(image_cv, obj, distance)
                    else:
                        rospy.logwarn(f"Person detection out of range: {distance:.2f}m")
        
        # Publish detections with tracking IDs
        self.publish_detections(current_frame_detections)
        
        # Display image with 3D detection overlay
        cv2.imshow('ZED 3D Object Detection', image_cv)
        
        return cv2.waitKey(1) & 0xFF != ord('q')

    def draw_3d_detection(self, image, obj, distance):
        """Draw enhanced 3D bounding box and information on the image"""
        bbox_2d = obj.bounding_box_2d
        
        if len(bbox_2d) >= 4:
            top_left = (int(bbox_2d[0][0]), int(bbox_2d[0][1]))
            bottom_right = (int(bbox_2d[2][0]), int(bbox_2d[2][1]))
            
            # Enhanced visualization with 3D information
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 3)
            
            position = obj.position
            tracking_id = obj.id if hasattr(obj, 'id') and obj.id >= 0 else 0
            
            # Main label with confidence, distance, and tracking ID
            main_label = f'Person ID:{tracking_id} {obj.confidence:.0f}% | {distance:.2f}m'
            cv2.putText(image, main_label, (top_left[0], top_left[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 3D coordinates
            pos_text = f'3D: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})'
            cv2.putText(image, pos_text, (top_left[0], bottom_right[1] + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw center point
            center_x = (top_left[0] + bottom_right[0]) // 2
            center_y = (top_left[1] + bottom_right[1]) // 2
            cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)

    def publish_detections(self, detections):
        """Publish 3D person detections with tracking IDs in raw camera frame"""
        # Message format: [num_persons, camera_x, camera_z, conf, id, ...]
        # camera_x: positive = right of camera (meters)
        # camera_z: positive = forward from camera (meters)
        msg_data = [len(detections)]
        
        for camera_x, camera_z, conf, tracking_id in detections:
            msg_data.extend([camera_x, camera_z, conf, tracking_id])
        
        msg = Float32MultiArray()
        msg.data = msg_data
        self.person_pub.publish(msg)
        
        if detections:
            rospy.loginfo_throttle(1, f"Published {len(detections)} tracked person detections")

    def get_camera_info(self):
        """Get ZED camera calibration information"""
        try:
            calibration_params = self.zed.get_camera_information().camera_configuration.calibration_parameters
            left_cam = calibration_params.left_cam
            stereo_transform = calibration_params.stereo_transform
            baseline = abs(stereo_transform.get_translation().get()[0])
            
            rospy.loginfo(f"ZED Camera Info: fx={left_cam.fx:.1f}, fy={left_cam.fy:.1f}")
            rospy.loginfo(f"ZED Baseline: {baseline:.3f}m")
            
            return left_cam.fx, left_cam.fy, baseline
            
        except Exception as e:
            rospy.logwarn(f"Error getting camera info: {e}")
            return None, None, None

    def run(self):
        """Main processing loop"""
        rospy.loginfo("Starting ZED 3D Object Detection with Tracking...")
        rospy.loginfo("Enhanced features:")
        rospy.loginfo("- Native 3D bounding boxes with ±2% accuracy")
        rospy.loginfo("- Person tracking with unique IDs across frames")
        rospy.loginfo("- Compatible with GP trajectory prediction")
        rospy.loginfo("Press 'q' in camera window to quit")
        
        self.get_camera_info()
        
        rate = rospy.Rate(15)  # 15 Hz for 3D processing
        
        while not rospy.is_shutdown():
            if not self.process_frame():
                break
            rate.sleep()
        
        self.cleanup()

    def cleanup(self):
        """Cleanup ZED resources"""
        try:
            self.zed.disable_object_detection()
            rospy.loginfo("Object detection disabled")
            
            self.zed.disable_positional_tracking()
            rospy.loginfo("Positional tracking disabled")
            
            self.zed.close()
            cv2.destroyAllWindows()
            rospy.loginfo("ZED 3D processor cleaned up successfully")
            
        except Exception as e:
            rospy.logwarn(f"Error during cleanup: {e}")

if __name__ == '__main__':
    try:
        processor = ZED3DProcessor()
        processor.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ZED 3D processor interrupted by user")
    except Exception as e:
        rospy.logerr(f"Error in ZED 3D processor: {e}")
