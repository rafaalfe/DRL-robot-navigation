import math
import os
import random
import subprocess
import time
from os import path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rospkg import RosPack
from sensor_msgs.msg import LaserScan, PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray

# --- Konstanta Global ---
GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.42  # Jarak deteksi tabrakan dari RPLiDAR
TIME_DELTA = 0.1  # Durasi setiap step
max_episode_steps = 500
REALSENSE_COLLISION_DIST = 0.40 # Jarak tabrakan yang lebih dekat untuk sensor depan (misal: 35 cm)


def check_pos(x, y):
    """
    Fungsi helper untuk memeriksa apakah sebuah posisi (goal/box) berada di dalam area rintangan statis.
    """
    goal_ok = True
    if -3.8 > x > -6.2 and 6.2 > y > 3.8: goal_ok = False
    if -1.3 > x > -2.7 and 4.7 > y > -0.2: goal_ok = False
    if -0.3 > x > -4.2 and 2.7 > y > 1.3: goal_ok = False
    if -0.8 > x > -4.2 and -2.3 > y > -4.2: goal_ok = False
    if -1.3 > x > -3.7 and -0.8 > y > -2.7: goal_ok = False
    if 4.2 > x > 0.8 and -1.8 > y > -3.2: goal_ok = False
    if 4 > x > 2.5 and 0.7 > y > -3.2: goal_ok = False
    if 6.2 > x > 3.8 and -3.3 > y > -4.2: goal_ok = False
    if 4.2 > x > 1.3 and 3.7 > y > 1.5: goal_ok = False
    if -3.0 > x > -7.2 and 0.5 > y > -1.5: goal_ok = False
    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5: goal_ok = False
    return goal_ok


class GazeboEnv(gym.Env):
    """
    ROS Gazebo Environment untuk training DRL, kompatibel dengan OpenAI Gym dan Stable Baselines3.
    Dirancang untuk training paralel dengan instance simulasi yang terisolasi.
    """

    def __init__(self, launchfile=None, port=11311, launch_delay=0): # launchfile tidak lagi dipakai
        super(GazeboEnv, self).__init__()

        # --- TAHAP 1: Tunggu giliran (jika ada jeda) ---
        if launch_delay > 0:
            print(f"Proses environment untuk port {port} menunggu {launch_delay} detik...")
            time.sleep(launch_delay)

        # --- TAHAP 2: Hubungkan ke ROS Master yang Sudah Ada ---
        ros_port = str(port)
        os.environ["ROS_MASTER_URI"] = f"http://localhost:{ros_port}"
        
        # Inisialisasi node ini ke roscore yang sudah berjalan di port tersebut
        try:
            rospy.init_node(f"rovid_env_client_{ros_port}", anonymous=True)
            print(f"Berhasil terhubung ke ROS Master di port {ros_port}.")
        except rospy.exceptions.ROSException as e:
            rospy.logerr(f"Gagal terhubung ke ROS Master di port {ros_port}. Pastikan Anda sudah menjalankan 'roscore' atau 'roslaunch' di terminal terpisah untuk port ini. Error: {e}")
            raise e


        # --- Inisialisasi Variabel State & Robot ---
        self.odom_x = 0
        self.odom_y = 0
        self.goal_x = 1.0
        self.goal_y = 0.0

        self.upper = 5.0 # Batas atas untuk random goal
        self.lower = -5.0 # Batas bawah untuk random goal
        self.last_odom = None

        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Inisialisasi State Persepsi
        self.perception_state_size = 1050  # 350 titik (5x70) * 3 koordinat (x,y,z)
        self.perception_range_max = 9.0    # Sesuai range L515
        self.perception_state = np.ones(self.perception_state_size) * self.perception_range_max

        self.rplidar_state_size = 240
        self.rplidar_range_max = 6.0       # Sesuai data /scan Anda
        self.rplidar_state = np.ones(self.rplidar_state_size) * self.rplidar_range_max
        self.raw_rplidar_scan = np.ones(self.rplidar_state_size) * self.rplidar_range_max
        # Di dalam __init__
        self.raw_perception_points = np.array([]) # Untuk menyimpan data mentah (x,y,z) dari RealSense
        
        # --- Publishers dan Subscribers ---
        self.vel_pub = rospy.Publisher("/rovid/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)

        self.odom_sub = rospy.Subscriber("/rovid/odom", Odometry, self.odom_callback, queue_size=1)
        self.drl_points_sub = rospy.Subscriber(
            "/depth_scan_cloud", PointCloud2, self.depth_data_callback, queue_size=1, buff_size=2**24)
        self.rplidar_sub = rospy.Subscriber("/scan", LaserScan, self.rplidar_callback, queue_size=1)

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "rovid"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0
        # --- Definisi Wajib untuk Stable Baselines3 ---
        # Action: [kecepatan_linear, kecepatan_sudut], nilai dinormalisasi antara -1 dan 1
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: gabungan semua data persepsi dan data robot
        state_dim = self.perception_state_size + self.rplidar_state_size + 4 # 1050 + 240 + 4 = 1294
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        print(f"Environment di port {ros_port} berhasil diinisialisasi.")

    def rplidar_callback(self, scan_data):
        """
        Callback ini menerima data LaserScan 360° dari RPLiDAR,
        melakukan downsampling ke ukuran yang tetap, menormalisasi,
        dan menyimpannya di self.rplidar_state.
        """
        try:
            raw_ranges = np.array(scan_data.ranges)
            
            # Ganti nilai 'inf' dan 'nan' dengan jarak maksimum sensor
            raw_ranges[np.isinf(raw_ranges)] = self.rplidar_range_max
            raw_ranges[np.isnan(raw_ranges)] = self.rplidar_range_max
            
            # --- Proses Downsampling ---
            # Ambil data mentah dan downsample ke ukuran yang kita inginkan (self.rplidar_state_size)
            # Ini adalah cara sederhana dan efisien untuk mengambil sampel secara merata
            num_raw_points = len(raw_ranges)
            step = float(num_raw_points) / self.rplidar_state_size
            
            downsampled_ranges = []
            for i in range(self.rplidar_state_size):
                index = int(i * step)
                downsampled_ranges.append(raw_ranges[index])
            
            self.raw_rplidar_scan = np.array(downsampled_ranges) # Simpan data mentah (dalam meter)

            # Normalisasi data ke rentang [0, 1]
            normalized_ranges = self.raw_rplidar_scan / self.rplidar_range_max
            
            # Simpan hasil akhir ke variabel state
            self.rplidar_state = normalized_ranges

        except Exception as e:
            rospy.logerr("Error processing RPLiDAR scan: %s", str(e))

    def depth_data_callback(self, cloud_msg):
        try:
            point_generator = pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z"))
            
            # --- PERUBAHAN DIMULAI DI SINI ---
            temp_points_raw = []
            temp_points_normalized = []
            
            for point in point_generator:
                # Simpan data mentah dalam meter
                temp_points_raw.append(point) 
                
                # Lakukan normalisasi seperti sebelumnya
                norm_x = point[0] / self.perception_range_max
                norm_y = point[1] / self.perception_range_max
                norm_z = point[2] / self.perception_range_max
                temp_points_normalized.extend([norm_x, norm_y, norm_z])
            
            # Simpan kedua versi data
            self.raw_perception_points = np.array(temp_points_raw)
            # --- PERUBAHAN SELESAI ---

            # Lanjutkan dengan kode yang ada untuk membuat state observasi
            processed_state = np.ones(self.perception_state_size, dtype=np.float32)
            num_values_to_copy = min(len(temp_points_normalized), self.perception_state_size)
            processed_state[:num_values_to_copy] = temp_points_normalized[:num_values_to_copy]
            self.perception_state = processed_state

        except Exception as e:
            rospy.logerr("Error processing DRL point cloud: %s", str(e))


    def odom_callback(self, od_data):
        self.last_odom = od_data

    # Perform an action and read a new state

    def step(self, action):
        self.current_step += 1

        # 1. Scaling aksi dari [-1, 1] ke kecepatan fisik
        linear_vel = action[0] * 0.35 + 0.15  # Map ke [0, 0.5] m/s
        angular_vel = action[1] * 1.0         # Map ke [-1.0, 1.0] rad/s
        
        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = angular_vel
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)
        
        # 2. Jalankan simulasi untuk satu langkah waktu
        rospy.wait_for_service("/gazebo/unpause_physics")
        try: self.unpause()
        except rospy.ServiceException: pass # Abaikan error jika service tidak tersedia sesaat
        time.sleep(TIME_DELTA)
        rospy.wait_for_service("/gazebo/pause_physics")
        try: self.pause()
        except rospy.ServiceException: pass

        # Guard clause jika data odometry tidak diterima
        if self.last_odom is None:
            rospy.logerr("Tidak menerima data odometry di step(), mengakhiri episode.")
            dummy_state = np.zeros(self.observation_space.shape, dtype=np.float32)
            # FIX: Kembalikan 5 item sesuai standar Gymnasium
            return dummy_state, -200.0, True, False, {"error": "no_odom"}

        # 3. Observasi environment setelah aksi
        done, collision, min_overall_dist = self.observe_collision()
        
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        
        # 4. Cek kondisi akhir episode (terminated atau truncated)
        target = False
        terminated = False
        truncated = False
        
        distance_to_goal = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        
        if distance_to_goal < GOAL_REACHED_DIST:
            print("!!! GOAL REACHED !!!")
            target = True
            terminated = True
        elif collision:
            print("--- COLLISION ---")
            terminated = True
        
        # Cek kondisi batas waktu (truncated)
        if self.current_step >= self.max_episode_steps:
            rospy.loginfo(f"Episode mencapai batas waktu {self.max_episode_steps} langkah.")
            truncated = True

        # 5. Hitung reward
        reward = self.get_reward(target, collision, linear_vel, angular_vel, min_overall_dist, distance_to_goal)
        
        # 6. Rakit state observasi untuk dikembalikan ke agen
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w, self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y, self.last_odom.pose.pose.orientation.z
        )
        angle = quaternion.to_euler(degrees=False)[2]

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        beta = math.atan2(skew_y, skew_x)
        theta = beta - angle
        if theta > np.pi: theta -= 2 * np.pi
        if theta < -np.pi: theta += 2 * np.pi

        perception_realsense = self.perception_state
        perception_rplidar = self.rplidar_state
        # Normalisasi data robot state
        norm_distance = distance_to_goal / 10.0
        norm_theta = theta / np.pi
        robot_state = [norm_distance, norm_theta, linear_vel, angular_vel]
        
        state = np.concatenate([perception_realsense, perception_rplidar, robot_state]).astype(np.float32)
        
        # Info dictionary, wajib untuk gym.Env
        info = {"is_success": target}
        
        # FIX: Hapus `truncated = False` yang salah di sini.
        # Kembalikan 5 item sesuai standar Gymnasium.
        return state, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        # Panggil super().reset() jika ingin menggunakan fitur seeding dari Gym
        super().reset(seed=seed)

        self.current_step = 0
        self.last_distance = None # Reset last_distance untuk kalkulasi reward

        rospy.wait_for_service("/gazebo/reset_world")
        try: self.reset_proxy()
        except rospy.ServiceException: print("Reset simulation failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        
        # Logika penempatan robot dan goal secara acak
        object_state = self.set_self_state
        x, y, position_ok = 0, 0, False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos(x, y)
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        # object_state.pose.orientation = quaternion.to_msg()
        self.set_state.publish(object_state)
        self.odom_x = x
        self.odom_y = y

        self.change_goal()
        self.random_box()
        # --- FIX: Panggil fungsi marker yang benar ---
        self.publish_markers([0.0, 0.0])

        # Unpause-pause untuk mendapatkan state awal
        rospy.wait_for_service("/gazebo/unpause_physics")
        try: self.unpause()
        except rospy.ServiceException: pass
        time.sleep(TIME_DELTA)
        rospy.wait_for_service("/gazebo/pause_physics")
        try: self.pause()
        except rospy.ServiceException: pass

        if self.last_odom is None:
            time.sleep(0.5)
            if self.last_odom is None:
                rospy.logerr("Tidak menerima data odometry saat reset.")
                dummy_state = np.zeros(self.observation_space.shape, dtype=np.float32)
                # FIX: Kembalikan 2 item (state, info) sesuai standar Gymnasium
                return dummy_state, {}
        
        # Rakit state awal yang akan dikembalikan
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(self.last_odom.pose.pose.orientation.w, self.last_odom.pose.pose.orientation.x, self.last_odom.pose.pose.orientation.y, self.last_odom.pose.pose.orientation.z)
        angle = quaternion.to_euler(degrees=False)[2]

        distance_to_goal = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        self.last_distance = distance_to_goal # Inisialisasi last_distance
        
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        beta = math.atan2(skew_y, skew_x)
        theta = beta - angle
        if theta > np.pi: theta -= 2 * np.pi
        if theta < -np.pi: theta += 2 * np.pi
        
        perception_realsense = self.perception_state
        perception_rplidar = self.rplidar_state
        norm_distance = distance_to_goal / 10.0
        norm_theta = theta / np.pi
        robot_state = [norm_distance, norm_theta, 0.0, 0.0]
        
        state = np.concatenate([perception_realsense, perception_rplidar, robot_state]).astype(np.float32)
        
        # Kembalikan 2 item (state, info) sesuai standar Gymnasium
        return state, {}


    
    def change_goal(self):
        # Place a new goal and check if its location is not on one of the obstacles
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            goal_ok = check_pos(self.goal_x, self.goal_y)

    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)


    def check_realsense_collision(points_array, collision_threshold):
        """
        Memeriksa tabrakan berdasarkan array titik 3D dan ambang batas jarak.
        Menolak nilai NaN dalam perhitungan jarak minimum.
        """
        if points_array.size == 0:
            return False, float('inf')

        # Menghitung jarak Euclidean untuk semua titik
        distances = np.linalg.norm(points_array, axis=1)
        print("Semua jarak Euclidean dari titik ke asal:")
        print(distances)

        # Menyaring jarak yang bukan NaN atau infinite
        valid_distances = distances[np.isfinite(distances)]

        # Jika tidak ada jarak valid, anggap tidak ada tabrakan
        if valid_distances.size == 0:
            return False, float('inf')

        # Mencari jarak minimum dari yang valid
        min_distance = np.min(valid_distances)

        # Memeriksa apakah jarak minimum lebih kecil dari ambang batas
        is_collision = min_distance < collision_threshold

        return is_collision, min_distance
    def observe_collision(self):
        # 1. Dapatkan jarak minimum dari LiDAR
        min_laser = self.rplidar_range_max
        if self.raw_rplidar_scan.size > 0:
            min_laser = np.min(self.raw_rplidar_scan)

        # 2. Dapatkan jarak minimum dari RealSense
        min_realsense = float('inf')
        if self.raw_perception_points.size > 0:
            # Menghitung semua jarak dari data RealSense yang valid
            distances = np.linalg.norm(self.raw_perception_points, axis=1)
            if distances.size > 0:
                min_realsense = np.min(distances)

        # 3. Tentukan jarak rintangan terdekat secara keseluruhan
        min_overall_dist = min(min_laser, min_realsense)

        # 4. Tentukan status tabrakan berdasarkan ambang batas masing-masing
        lidar_collision = min_laser < COLLISION_DIST  # e.g., 0.55m
        realsense_collision = min_realsense < REALSENSE_COLLISION_DIST # e.g., 0.35m
        
        collision = lidar_collision or realsense_collision
        
        # Kembalikan status tabrakan dan jarak terdekat gabungan
        if collision:
            return True, True, min_overall_dist
            
        return False, False, min_overall_dist

    def get_reward(self, target, collision, linear_vel, angular_vel, min_overall_dist, distance_to_goal):
        W_DIST = 150.0
        W_LASER = -2.0
        W_TIME = -0.5
        W_ACTION = -0.1

        if target:
            print("!!! GOAL REACHED !!!")
            return 300.0
        if collision:
            print("--- COLLISION ---")
            return -300.0

        prev_distance = getattr(self, 'last_distance', distance_to_goal)
        reward_dist = (prev_distance - distance_to_goal) * W_DIST
        self.last_distance = distance_to_goal

        reward_laser = W_LASER if min_overall_dist < 0.55 else 0.0
        reward_obstacle = 0.0
        if min_overall_dist < 0.55:
            reward_obstacle = -1.5 * (1.0 - (min_overall_dist / 0.55))
        reward_action = abs(angular_vel) * W_ACTION
        reward_time = W_TIME
        total_reward = reward_dist + reward_action + reward_time + reward_obstacle

        return total_reward
