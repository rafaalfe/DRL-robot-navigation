#!/usr/bin/env python3

import rospy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random
import time
import os

from geometry_msgs.msg import Twist, PoseStamped, Pose, Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan, PointCloud2
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState
from std_srvs.srv import Empty
from squaternion import Quaternion
import sensor_msgs.point_cloud2 as pc2

COLLISION_DIST = 0.42
REALSENSE_COLLISION_DIST = 0.40

class MultiRoomGazeboEnv(gym.Env):
    def __init__(self, port=11311, launch_delay=0):
        super(MultiRoomGazeboEnv, self).__init__()

        if launch_delay > 0:
            time.sleep(launch_delay)
        ros_port = str(port)
        os.environ["ROS_MASTER_URI"] = f"http://localhost:{ros_port}"
        try:
            rospy.init_node(f"rovid_env_client_{ros_port}", anonymous=True)
        except rospy.exceptions.ROSException as e:
            rospy.logerr(f"Gagal terhubung ke ROS Master di port {ros_port}. Error: {e}")
            raise e

        self.vel_pub = rospy.Publisher("/rovid/cmd_vel", Twist, queue_size=1)
        self.set_state_pub = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        
        self.odom_sub = rospy.Subscriber("/rovid/odom", Odometry, self._odom_callback)
        self.laser_sub = rospy.Subscriber("/scan", LaserScan, self._laser_callback)
        self.depth_sub = rospy.Subscriber("/depth_scan_cloud", PointCloud2, self._depth_callback, queue_size=1, buff_size=2**24)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1294,), dtype=np.float32)

        self.max_episode_steps = 1500
        self.look_ahead_dist = 0.8
        self.goal_tolerance = 0.5

        self.room_zones = [
            (-13.0, -9.0, -1.0, -1.0),    # Ruangan 1 (R1)
            (-2.0, -17.0, 2.0, -14.5),   # Ruangan 2 (R2)
            (-49.0, -9.0, -45.5, -6.3),  # Ruangan 3 (R3)
        ]
        self.corridor_zones = [
            (-50.3, -14.3, -15.3, -11.9), # Koridor 1 (K1)
            (-15.4, -14.5, -4.3, -11.0)  # Koridor 2 (K2)
        ]
        self.all_safe_zones = self.room_zones + self.corridor_zones

        self.static_obstacle_names = ["cardboard_box_1", "cardboard_box_2", "cardboard_box_3", "cardboard_box_5"]
        self.moving_obstacle_names = ["cardboard_box_0", "cardboard_box_4", "cardboard_box_6"]
        self.moving_obstacle_targets = {}

        self.current_step = 0
        self.last_odom = None
        self.global_plan = []
        self.last_distance_to_final_goal = 0.0
        
        # Variabel untuk data sensor mentah dan terproses
        self.rplidar_state_size = 240
        self.rplidar_range_max = 6.0
        self.raw_rplidar_scan = np.ones(self.rplidar_state_size) * self.rplidar_range_max
        self.processed_laser_scan = np.ones(self.rplidar_state_size) * 1.0

        self.perception_state_size = 1050
        self.perception_range_max = 9.0
        self.raw_perception_points = np.array([])
        self.processed_point_cloud = np.ones(self.perception_state_size) * 1.0
        
        self.obstacle_update_rate = 10
        rospy.Timer(rospy.Duration(1.0/self.obstacle_update_rate), self._update_moving_obstacles)
        
        rospy.loginfo(f"Environment untuk Arena LABC di port {ros_port} berhasil diinisialisasi.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        rospy.wait_for_service("/gazebo/reset_world")
        try: self.reset_proxy()
        except rospy.ServiceException: rospy.logerr("Reset simulation failed")
        self._generate_new_task()
        time.sleep(0.2)
        initial_observation = self._get_observation()
        return initial_observation, {}

    def step(self, action):
        self.current_step += 1
        linear_vel = action[0] * 0.35 + 0.15
        angular_vel = action[1] * 1.0
        vel_cmd = Twist(linear=Point(x=linear_vel), angular=Point(z=angular_vel))
        self.vel_pub.publish(vel_cmd)
        time.sleep(0.1)

        collision, min_overall_dist = self.observe_collision()
        observation = self._get_observation(linear_vel, angular_vel)
        reward, terminated, truncated = self._calculate_reward(collision, min_overall_dist)
        
        info = {"is_success": terminated and not truncated and not collision}
        return observation, reward, terminated, truncated, info

    def observe_collision(self):
        min_laser = self.rplidar_range_max
        if self.raw_rplidar_scan is not None and self.raw_rplidar_scan.size > 0:
            min_laser = np.min(self.raw_rplidar_scan)

        min_realsense = float('inf')
        if self.raw_perception_points is not None and self.raw_perception_points.size > 0:
            distances = np.linalg.norm(self.raw_perception_points, axis=1)
            if distances.size > 0:
                min_realsense = np.min(distances)

        min_overall_dist = min(min_laser, min_realsense)
        collision = (min_laser < COLLISION_DIST) or (min_realsense < REALSENSE_COLLISION_DIST)
        return collision, min_overall_dist

    def _get_observation(self, linear_vel=0.0, angular_vel=0.0):
        if not self.last_odom:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        odom_x, odom_y = self.last_odom.pose.pose.position.x, self.last_odom.pose.pose.position.y
        lookahead_point = self._find_lookahead_point(odom_x, odom_y)
        distance_to_target = math.hypot(lookahead_point.pose.position.x - odom_x, lookahead_point.pose.position.y - odom_y)
        
        q = Quaternion(self.last_odom.pose.pose.orientation.w, self.last_odom.pose.pose.orientation.x, self.last_odom.pose.pose.orientation.y, self.last_odom.pose.pose.orientation.z)
        robot_angle = q.to_euler(degrees=False)[2]
        angle_to_target = math.atan2(lookahead_point.pose.position.y - odom_y, lookahead_point.pose.position.x - odom_x)
        
        theta = angle_to_target - robot_angle
        if theta > np.pi: theta -= 2 * np.pi
        if theta < -np.pi: theta += 2 * np.pi
        
        norm_distance = distance_to_target / 10.0
        norm_theta = theta / np.pi
        
        robot_state = [norm_distance, norm_theta, linear_vel, angular_vel]
        
        # Gabungkan semua menjadi satu vektor observasi
        observation = np.concatenate([self.processed_point_cloud, self.processed_laser_scan, robot_state]).astype(np.float32)

        # --- SAFETY CHECK ---
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            rospy.logerr("NaN or Inf found in observation vector! Resetting to zeros.")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        return observation

    def _laser_callback(self, msg):
        # Menyimpan data mentah untuk deteksi tabrakan
        raw_ranges = np.array(msg.ranges)
        raw_ranges[np.isinf(raw_ranges)] = self.rplidar_range_max
        raw_ranges[np.isnan(raw_ranges)] = self.rplidar_range_max
        if len(raw_ranges) != self.rplidar_state_size:
            step = float(len(raw_ranges)) / self.rplidar_state_size
            self.raw_rplidar_scan = np.array([raw_ranges[int(i * step)] for i in range(self.rplidar_state_size)])
        else:
            self.raw_rplidar_scan = raw_ranges
        
        # Menyimpan data terproses untuk vektor observasi
        self.processed_laser_scan = self.raw_rplidar_scan / self.rplidar_range_max

    def _depth_callback(self, msg):
        try:
            point_generator = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))
            temp_points_raw = list(point_generator)
            self.raw_perception_points = np.array(temp_points_raw)
            
            # Memproses point cloud untuk vektor observasi
            temp_points_normalized = []
            for point in temp_points_raw:
                temp_points_normalized.extend([p / self.perception_range_max for p in point])
            
            processed_state = np.ones(self.perception_state_size, dtype=np.float32)
            num_to_copy = min(len(temp_points_normalized), self.perception_state_size)
            processed_state[:num_to_copy] = temp_points_normalized[:num_to_copy]
            self.processed_point_cloud = processed_state
        except Exception as e:
            rospy.logerr(f"Error processing DRL point cloud: {e}")
            self.raw_perception_points = np.array([])
            self.processed_point_cloud = np.ones(self.perception_state_size, dtype=np.float32)

    # ... (Sisa file tetap sama) ...
    def _generate_new_task(self):
        start_zone = random.choice(self.room_zones); goal_zone = random.choice([z for z in self.room_zones if z != start_zone]); intermediate_zone = random.choice(self.corridor_zones)
        start_x, start_y = self._get_random_point_in_zone(start_zone); intermediate_x, intermediate_y = self._get_random_point_in_zone(intermediate_zone); goal_x, goal_y = self._get_random_point_in_zone(goal_zone)
        state = ModelState(model_name="rovid", pose=Pose(position=Point(x=start_x, y=start_y, z=0))); self.set_state_pub.publish(state)
        self.global_plan = []; [self.global_plan.append(PoseStamped(pose=Pose(position=Point(x=p_x, y=p_y, z=0)))) for p_x, p_y in [(start_x, start_y), (intermediate_x, intermediate_y), (goal_x, goal_y)]]
        if self.last_odom: self.last_distance_to_final_goal = self._get_distance_to_final_goal()
        all_obstacles = self.static_obstacle_names + self.moving_obstacle_names
        for name in all_obstacles: self._place_random_obstacle(name, start_x, start_y, goal_x, goal_y)
        self.moving_obstacle_targets.clear()
        for name in self.moving_obstacle_names:
            obs_start_zone, obs_end_zone = random.sample(self.all_safe_zones, 2)
            obs_start_x, obs_start_y = self._get_random_point_in_zone(obs_start_zone); obs_end_x, obs_end_y = self._get_random_point_in_zone(obs_end_zone)
            obs_state = ModelState(model_name=name, pose=Pose(position=Point(x=obs_start_x, y=obs_start_y, z=0))); self.set_state_pub.publish(obs_state)
            self.moving_obstacle_targets[name] = {"start": (obs_start_x, obs_start_y), "end": (obs_end_x, obs_end_y), "speed": random.uniform(0.2, 0.4), "direction": 1}
    def _update_moving_obstacles(self, event):
        for name, target_info in self.moving_obstacle_targets.items():
            try:
                model_state = self.get_model_state(name, ""); current_pos = model_state.pose.position
                target_pos = target_info["end"] if target_info["direction"] == 1 else target_info["start"]
                dir_x, dir_y = target_pos[0] - current_pos.x, target_pos[1] - current_pos.y; dist = math.hypot(dir_x, dir_y)
                if dist < 0.2: target_info["direction"] *= -1; continue
                move_x, move_y = (dir_x / dist) * target_info["speed"] * (1.0/self.obstacle_update_rate), (dir_y / dist) * target_info["speed"] * (1.0/self.obstacle_update_rate)
                new_pose = model_state.pose; new_pose.position.x += move_x; new_pose.position.y += move_y
                self.set_state_pub.publish(ModelState(model_name=name, pose=new_pose))
            except Exception: pass
    def _place_random_obstacle(self, name, start_x, start_y, goal_x, goal_y):
        obs_zone = random.choice(self.all_safe_zones)
        obs_x, obs_y = self._get_random_point_in_zone(obs_zone)
        while math.hypot(obs_x - start_x, obs_y - start_y) < 1.5 or math.hypot(obs_x - goal_x, obs_y - goal_y) < 1.5:
            obs_zone = random.choice(self.all_safe_zones); obs_x, obs_y = self._get_random_point_in_zone(obs_zone)
        self.set_state_pub.publish(ModelState(model_name=name, pose=Pose(position=Point(x=obs_x, y=obs_y, z=0))))
    def _get_random_point_in_zone(self, zone): return random.uniform(zone[0], zone[2]), random.uniform(zone[1], zone[3])
    def _calculate_reward(self, collision, min_overall_dist):
        terminated = False; truncated = False
        dist_to_final = self._get_distance_to_final_goal()
        if dist_to_final < self.goal_tolerance: terminated = True; return 300.0, terminated, truncated
        if collision: terminated = True; return -300.0, terminated, truncated
        if self.current_step >= self.max_episode_steps: truncated = True
        reward_progress = (self.last_distance_to_final_goal - dist_to_final) * 150.0; self.last_distance_to_final_goal = dist_to_final
        cte = self._calculate_cross_track_error(); reward_cte = -1.5 * (cte**2)
        reward_obstacle = 0.0
        if min_overall_dist < 0.6: reward_obstacle = -2.0 * (1.0 - (min_overall_dist / 0.6))
        reward_time = -0.5
        total_reward = reward_progress + reward_cte + reward_obstacle + reward_time
        return total_reward, terminated, truncated
    def _find_lookahead_point(self, robot_x, robot_y):
        if not self.global_plan: return PoseStamped()
        for i in range(len(self.global_plan) - 1, -1, -1):
            pt_x, pt_y = self.global_plan[i].pose.position.x, self.global_plan[i].pose.position.y
            if math.hypot(robot_x - pt_x, robot_y - pt_y) < self.look_ahead_dist:
                return self.global_plan[i+1] if i < len(self.global_plan) - 1 else self.global_plan[-1]
        return self.global_plan[0]
    def _get_distance_to_final_goal(self):
        if not self.global_plan or not self.last_odom: return float('inf')
        final_goal = self.global_plan[-1].pose.position; odom_pos = self.last_odom.pose.pose.position
        return math.hypot(final_goal.x - odom_pos.x, final_goal.y - odom_pos.y)
    def _calculate_cross_track_error(self): return 0.0
    def _odom_callback(self, msg): self.last_odom = msg
