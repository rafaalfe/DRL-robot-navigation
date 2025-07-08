#!/usr/bin/env python3

import rospy
import numpy as np
import math
import tf
from squaternion import Quaternion
from stable_baselines3 import TD3

# Import message types
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, PointCloud2
import sensor_msgs.point_cloud2 as pc2

class DrlLocalPlanner:
    def __init__(self):
        rospy.init_node('drl_local_planner_standalone', anonymous=True)

        # --- Parameters ---
        model_path = rospy.get_param("~model_path", "/home/rafaalfe/training_phase2/rovid_phase2_10cpu/best_model/best_model.zip")
        self.control_rate = rospy.get_param("~control_rate", 10)  # Hz

        # --- Load the Trained Model ---
        rospy.loginfo(f"Loading model from: {model_path}")
        self.model = TD3.load(model_path)
        rospy.loginfo("Model loaded successfully.")

        # --- Buat objek TransformListener ---
        self.tf_listener = tf.TransformListener()

        # --- Class Variables ---
        self.last_odom = None
        self.current_goal = None
        self.last_cmd_vel = Twist()
        self.odom_frame = "odom"

        # ... sisa variabel state sama ...
        self.perception_state_size = 1050
        self.rplidar_state_size = 240
        self.perception_range_max = 9.0
        self.rplidar_range_max = 6.0
        self.perception_state = np.ones(self.perception_state_size) * self.perception_range_max
        self.rplidar_state = np.ones(self.rplidar_state_size) * self.rplidar_range_max

        # --- Publishers and Subscribers ---
        self.vel_pub = rospy.Publisher('/rovid/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/rovid/odom', Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.rplidar_callback, queue_size=1)
        rospy.Subscriber('/depth_scan_cloud', PointCloud2, self.depth_data_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=1)
        
        rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.compute_velocity_command)
        rospy.loginfo("DRL Standalone Planner is running. Set a goal in RViz.")

    def goal_callback(self, goal_msg):
        """Callback ini menerima goal dan mentransformasikannya ke frame odom."""
        try:
            # Atur stempel waktu ke rospy.Time(0) untuk meminta transformasi TERBARU yang tersedia.
            goal_msg.header.stamp = rospy.Time(0)
            
            # Tunggu transformasi antara frame goal dan frame odom tersedia
            self.tf_listener.waitForTransform(self.odom_frame, goal_msg.header.frame_id, rospy.Time(0), rospy.Duration(4.0))
            
            # Lakukan transformasi goal ke frame odom
            transformed_goal = self.tf_listener.transformPose(self.odom_frame, goal_msg)
            
            # --- PERBAIKAN TYPO DI SINI ---
            self.current_goal = transformed_goal
            
            rospy.loginfo(f"Received new goal in '{goal_msg.header.frame_id}' frame. Transformed to '{self.odom_frame}' frame.")
            rospy.loginfo(f"New goal -> x: {self.current_goal.pose.position.x:.2f}, y: {self.current_goal.pose.position.y:.2f} (in odom frame)")

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Failed to transform goal: {e}")

    # ... (Fungsi callback sensor lainnya tetap sama) ...
    def rplidar_callback(self, scan_data):
        try:
            raw_ranges = np.array(scan_data.ranges)
            raw_ranges[np.isinf(raw_ranges)] = self.rplidar_range_max
            raw_ranges[np.isnan(raw_ranges)] = self.rplidar_range_max
            num_raw_points = len(raw_ranges)
            step = float(num_raw_points) / self.rplidar_state_size
            downsampled_ranges = [raw_ranges[int(i * step)] for i in range(self.rplidar_state_size)]
            self.rplidar_state = np.array(downsampled_ranges) / self.rplidar_range_max
        except Exception as e:
            rospy.logerr("Error processing RPLiDAR scan: %s", str(e))

    def depth_data_callback(self, cloud_msg):
        try:
            point_generator = pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z"))
            temp_points_normalized = []
            for point in point_generator:
                temp_points_normalized.extend([p / self.perception_range_max for p in point])
            processed_state = np.ones(self.perception_state_size, dtype=np.float32)
            num_values_to_copy = min(len(temp_points_normalized), self.perception_state_size)
            processed_state[:num_values_to_copy] = temp_points_normalized[:num_values_to_copy]
            self.perception_state = processed_state
        except Exception as e:
            rospy.logerr("Error processing DRL point cloud: %s", str(e))
            
    def odom_callback(self, od_data):
        self.last_odom = od_data
        self.odom_frame = od_data.header.frame_id

    def compute_velocity_command(self, event):
        if self.last_odom is None or self.current_goal is None:
            return

        # ... (Tidak ada perubahan di sini) ...
        odom_pose = self.last_odom.pose.pose
        odom_x, odom_y = odom_pose.position.x, odom_pose.position.y
        goal_x, goal_y = self.current_goal.pose.position.x, self.current_goal.pose.position.y
        
        distance_to_goal = np.linalg.norm([odom_x - goal_x, odom_y - goal_y])
        
        q = Quaternion(odom_pose.orientation.w, odom_pose.orientation.x, odom_pose.orientation.y, odom_pose.orientation.z)
        robot_angle = q.to_euler(degrees=False)[2]

        angle_to_goal = math.atan2(goal_y - odom_y, goal_x - odom_x)
        
        theta = angle_to_goal - robot_angle
        if theta > np.pi: theta -= 2 * np.pi
        if theta < -np.pi: theta += 2 * np.pi

        norm_distance = distance_to_goal / 10.0
        norm_theta = theta / np.pi
        
        last_linear_vel = self.last_cmd_vel.linear.x
        last_angular_vel = self.last_cmd_vel.angular.z
        
        robot_state = [norm_distance, norm_theta, last_linear_vel, last_angular_vel]
        
        observation = np.concatenate([self.perception_state, self.rplidar_state, robot_state]).astype(np.float32)

        action, _ = self.model.predict(observation, deterministic=True)
        
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0] * 0.35 + 0.15
        vel_cmd.angular.z = action[1] * 1.0
        
        self.vel_pub.publish(vel_cmd)
        self.last_cmd_vel = vel_cmd
        
        if distance_to_goal < 0.3:
             self.vel_pub.publish(Twist())
             self.current_goal = None
             rospy.loginfo("Goal reached! Waiting for new goal.")

if __name__ == '__main__':
    try:
        DrlLocalPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
