#!/usr/bin/env python3
"""
ROS node: subscribes to camera image, runs HSV detection + centroid tracking,
publishes the primary tracked object's pose in the robot base frame.
"""

import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge

from sawyer_vision_tracker.detector import Detector
from sawyer_vision_tracker.tracker import ObjectTracker
from sawyer_vision_tracker.coordinate_converter import CoordinateConverter


class VisionTrackerNode:

    def __init__(self):
        rospy.init_node("vision_tracker_node")

        # load params
        detection_cfg = rospy.get_param("detection", {})
        tracking_cfg = rospy.get_param("tracking", {})
        camera_cfg = rospy.get_param("camera", {})

        color_filter = rospy.get_param("~color_filter", "")
        self.max_objects = rospy.get_param("~max_objects", 1)

        # filter HSV ranges to a single color if requested
        if color_filter:
            all_ranges = detection_cfg.get("hsv_ranges", [])
            matched = [r for r in all_ranges if r["name"] == color_filter]
            if not matched:
                names = [r["name"] for r in all_ranges]
                rospy.logfatal(
                    f"Unknown color '{color_filter}'. Available: {names}"
                )
                rospy.signal_shutdown("Bad color filter")
                return
            detection_cfg["hsv_ranges"] = matched
            rospy.loginfo(f"Color filter active: detecting only '{color_filter}'")

        self.detector = Detector(detection_cfg)
        self.tracker = ObjectTracker(tracking_cfg)
        self.bridge = CvBridge()

        # coordinate converter
        intr = camera_cfg.get("intrinsics", {})
        self.converter = CoordinateConverter(
            fx=intr.get("fx", 554.38),
            fy=intr.get("fy", 554.38),
            cx=intr.get("cx", 320.5),
            cy=intr.get("cy", 240.5),
            camera_frame=camera_cfg.get("camera_frame", "front_cam_link"),
            base_frame=camera_cfg.get("base_frame", "base"),
        )
        self.fixed_z = camera_cfg.get("fixed_z", 0.7)

        # publishers
        self.pose_pub = rospy.Publisher(
            "/vision_tracker/target_pose", PoseStamped, queue_size=1
        )
        self.valid_pub = rospy.Publisher(
            "/vision_tracker/target_valid", Bool, queue_size=1
        )

        # subscribe to camera
        image_topic = camera_cfg.get("image_topic", "/camera/color/image_raw")
        rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.loginfo(f"Subscribed to {image_topic}")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr_throttle(5.0, f"cv_bridge error: {e}")
            return

        # detect
        detections = self.detector.detect(frame)

        # cap to N largest
        if self.max_objects and len(detections) > self.max_objects:
            detections.sort(key=lambda d: d.area, reverse=True)
            detections = detections[: self.max_objects]

        # track
        tracked = self.tracker.update(detections)

        # find primary object and convert to base frame
        primary = tracked[0] if tracked else None
        target_valid = False
        base_coords = None

        if primary is not None:
            cx, cy = int(primary.centroid[0]), int(primary.centroid[1])
            base_coords = self.converter.pixel_to_base(cx, cy, self.fixed_z)

            if base_coords is not None:
                target_valid = True
                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = "base"
                pose_msg.pose.position.x = base_coords[0]
                pose_msg.pose.position.y = base_coords[1]
                pose_msg.pose.position.z = base_coords[2]
                # orientation: gripper pointing down (same as sawyer_action move_to)
                pose_msg.pose.orientation.w = 1.0
                self.pose_pub.publish(pose_msg)

        self.valid_pub.publish(Bool(data=target_valid))

        # draw annotated frame
        self._draw(frame, tracked, primary, base_coords)

    def _draw(self, frame, tracked, primary, base_coords):
        """Draw bounding boxes, labels, and coordinate info on the frame."""
        font = cv2.FONT_HERSHEY_SIMPLEX

        for obj in tracked:
            color = obj.color_bgr
            x, y, w, h = obj.bbox
            cx, cy = int(obj.centroid[0]), int(obj.centroid[1])

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # crosshair
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), color, 1)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), color, 1)

            # label
            label = f"{obj.color_name} #{obj.id}"
            cv2.putText(frame, label, (x, y - 8), font, 0.5, color, 2)

            # trajectory
            pts = list(obj.trajectory)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                faded = tuple(int(c * alpha) for c in color)
                p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                p2 = (int(pts[i][0]), int(pts[i][1]))
                cv2.line(frame, p1, p2, faded, max(1, int(alpha * 3)))

        # base-frame coordinates for primary object
        if primary is not None and base_coords is not None:
            bx, by, bz = base_coords
            info = f"Base: ({bx:.3f}, {by:.3f}, {bz:.3f})"
            cv2.putText(frame, info, (10, frame.shape[0] - 15), font, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, info, (10, frame.shape[0] - 15), font, 0.55, (0, 200, 0), 1)

        # HUD
        n = len(tracked)
        cv2.putText(frame, f"Objects: {n}", (10, 25), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Objects: {n}", (10, 25), font, 0.6, (0, 0, 0), 1)

        cv2.imshow("Sawyer Vision Tracker", frame)
        cv2.waitKey(1)

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    node = VisionTrackerNode()
    node.run()
