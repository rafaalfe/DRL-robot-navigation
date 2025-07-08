#!/usr/bin/env python3

import rospy
import numpy as np
import math
import tf
from squaternion import Quaternion
from stable_baselines3 import TD3

# Import message types
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan, PointCloud2
import sensor_msgs.point_cloud2 as pc2

class DrlPathFollower:
    def __init__(self):
        rospy.init_node('drl_path_follower_node', anonymous=True)

        # --- Parameters ---
        model_path = rospy.get_param("~model_path", "/home/rafaalfe/training_phase2/rovid_phase2_second_10cpu/best_model/best_model.zip")
        self.control_rate = rospy.get_param("~control_rate", 10)
        self.look_ahead_dist = rospy.get_param("~look_ahead_dist", 1.0 )
        self.final_goal_tolerance = 0.3

        # --- Load Model ---
        rospy.loginfo(f"Loading model from: {model_path}")
        self.model = TD3.load(model_path)
        rospy.loginfo("Model loaded successfully.")

        # --- Class Variables ---
        self.last_odom = None
        self.last_cmd_vel = Twist()
        self.odom_frame = "odom"
        self.robot_base_frame = "base_link"
        self.global_plan = []
        self.tf_listener = tf.TransformListener()

        # ... (variabel state sensor sama) ...
        self.perception_state_size = 1050
        self.rplidar_state_size = 240
        self.perception_range_max = 9.0
        self.rplidar_range_max = 6.0
        self.perception_state = np.ones(self.perception_state_size) * self.perception_range_max
        self.rplidar_state = np.ones(self.rplidar_state_size) * self.rplidar_range_max
        
        # --- Publishers and Subscribers ---
        self.lookahead_pub = rospy.Publisher('/lookahead_point', PoseStamped, queue_size=1)
        self.vel_pub = rospy.Publisher('/rovid/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/rovid/odom', Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.rplidar_callback, queue_size=1)
        rospy.Subscriber('/depth_scan_cloud', PointCloud2, self.depth_data_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/drl_global_plan', Path, self.path_callback, queue_size=1)
        
        # --- Reset state saat mulai dan saat mati untuk keamanan ---
        rospy.on_shutdown(self.shutdown_hook)
        self.reset_robot_state()

        rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.compute_velocity_command)
        rospy.loginfo("DRL Path Follower is running. Waiting for a global plan...")

    def reset_robot_state(self):
        """Mengirim perintah kecepatan nol dan mereset state aksi terakhir."""
        rospy.loginfo("Resetting robot state (velocity and last action).")
        stop_msg = Twist()
        # Publikasikan beberapa kali untuk memastikan pesan diterima
        rate = rospy.Rate(10)
        for _ in range(3):
            self.vel_pub.publish(stop_msg)
            rate.sleep()
        # Reset "bobot" atau state aksi terakhir ke nol
        self.last_cmd_vel = stop_msg

    def shutdown_hook(self):
        """Fungsi yang dipanggil saat node dimatikan untuk memastikan robot berhenti."""
        rospy.loginfo("DRL Path Follower is shutting down. Stopping the robot.")
        self.reset_robot_state()

    def path_callback(self, path_msg):
        """Menerima path baru dari global planner dan mereset state robot."""
        if not path_msg.poses:
            rospy.logwarn("Received an empty plan.")
            return
        
        # --- PERUBAHAN UTAMA: Reset state setiap kali ada goal baru ---
        rospy.loginfo("New global plan received. Resetting robot state before starting.")
        self.reset_robot_state()
        
        self.global_plan = path_msg.poses
        rospy.loginfo(f"Starting new plan with {len(self.global_plan)} waypoints.")

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
        self.robot_base_frame = od_data.child_frame_id

    def find_lookahead_point(self, robot_x, robot_y):
        """Mencari titik 'wortel' di jalur yang berjarak look_ahead_dist dari robot."""
        for i in range(len(self.global_plan) - 1, -1, -1):
            pt_x = self.global_plan[i].pose.position.x
            pt_y = self.global_plan[i].pose.position.y
            dist = math.hypot(robot_x - pt_x, robot_y - pt_y)
            
            if dist < self.look_ahead_dist:
                if i == len(self.global_plan) - 1:
                    return self.global_plan[-1]
                return self.global_plan[i+1]
        
        return self.global_plan[0]

    def compute_velocity_command(self, event):
        if self.last_odom is None or not self.global_plan:
            return

        odom_pose = self.last_odom.pose.pose
        odom_x, odom_y = odom_pose.position.x, odom_pose.position.y
        
        final_goal = self.global_plan[-1]
        dist_to_final_goal = math.hypot(odom_x - final_goal.pose.position.x, odom_y - final_goal.pose.position.y)
        if dist_to_final_goal < self.final_goal_tolerance:
            rospy.loginfo_once("Final goal reached!")
            self.reset_robot_state()
            self.global_plan = []
            return

        lookahead_point = self.find_lookahead_point(odom_x, odom_y)

        lookahead_pose_viz = PoseStamped()
        lookahead_pose_viz.header.frame_id = self.global_plan[0].header.frame_id
        lookahead_pose_viz.header.stamp = rospy.Time.now()
        lookahead_pose_viz.pose = lookahead_point.pose
        self.lookahead_pub.publish(lookahead_pose_viz)

        try:
            lookahead_point.header.stamp = rospy.Time(0)
            target_in_odom = self.tf_listener.transformPose(self.odom_frame, lookahead_point)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Failed to transform lookahead point: {e}")
            return

        goal_x, goal_y = target_in_odom.pose.position.x, target_in_odom.pose.position.y
        distance_to_target = math.hypot(odom_x - goal_x, odom_y - goal_y)
        
        q = Quaternion(odom_pose.orientation.w, odom_pose.orientation.x, odom_pose.orientation.y, odom_pose.orientation.z)
        robot_angle = q.to_euler(degrees=False)[2]
        angle_to_target = math.atan2(goal_y - odom_y, goal_x - odom_x)
        
        theta = angle_to_target - robot_angle
        if theta > np.pi: theta -= 2 * np.pi
        if theta < -np.pi: theta += 2 * np.pi

        norm_distance = distance_to_target / 10.0
        norm_theta = theta / np.pi
        
        robot_state = [norm_distance, norm_theta, self.last_cmd_vel.linear.x, self.last_cmd_vel.angular.z]
        observation = np.concatenate([self.perception_state, self.rplidar_state, robot_state]).astype(np.float32)

        action, _ = self.model.predict(observation, deterministic=True)
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0] * 0.35 + 0.15
        vel_cmd.angular.z = action[1] * 1.0
        self.vel_pub.publish(vel_cmd)
        self.last_cmd_vel = vel_cmd

if __name__ == '__main__':
    try:
        DrlPathFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
