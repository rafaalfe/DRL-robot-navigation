#!/usr/bin/env python3

import rospy
import tf
import numpy as np
import heapq
import cv2 # Kita akan menggunakan OpenCV untuk inflasi yang efisien

# Import message types
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point

class AStarGlobalPlanner:
    def __init__(self):
        rospy.init_node('astar_global_planner_node')

        # --- Parameters ---
        # Nilai di peta di atas ini dianggap rintangan
        self.obstacle_threshold = 100
        # Jarak aman dari rintangan (dalam meter)
        self.inflation_radius = rospy.get_param("~inflation_radius", 0.3) 

        # --- Class Variables ---
        self.map_data = None
        self.inflated_map_data = None # Peta yang sudah ditebalkan
        self.map_resolution = 0
        self.map_origin = None
        self.map_width = 0
        self.map_height = 0
        self.tf_listener = tf.TransformListener()
        self.map_frame = "map"
        self.robot_frame = "base_link"

        # --- Publishers and Subscribers ---
        self.path_pub = rospy.Publisher('/drl_global_plan', Path, queue_size=10)
        # Publikasikan peta yang sudah diinflasi untuk debugging
        self.inflated_map_pub = rospy.Publisher('/inflated_map', OccupancyGrid, queue_size=1) 
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)

        rospy.loginfo("A* Global Planner is running. Waiting for map and goal...")

    def inflate_map(self):
        """Menebalkan rintangan pada peta menggunakan OpenCV."""
        if self.map_data is None:
            return
        
        rospy.loginfo("Inflating map...")
        # Buat peta biner, 100 untuk rintangan, 0 untuk ruang bebas
        binary_map = np.zeros_like(self.map_data, dtype=np.uint8)
        binary_map[self.map_data > self.obstacle_threshold] = 100
        
        # Hitung radius inflasi dalam sel grid
        inflation_cells = int(self.inflation_radius / self.map_resolution)
        
        # Buat kernel untuk operasi morfologi (dilasi)
        kernel = np.ones((inflation_cells * 2 + 1, inflation_cells * 2 + 1), np.uint8)
        
        # Lakukan dilasi untuk "menebalkan" rintangan
        inflated_binary_map = cv2.dilate(binary_map, kernel, iterations=1)
        
        # Simpan peta yang sudah diinflasi
        self.inflated_map_data = np.copy(self.map_data)
        self.inflated_map_data[inflated_binary_map == 100] = 100 # Tandai area inflasi sebagai rintangan
        rospy.loginfo("Map inflation complete.")

        # Publikasikan peta yang sudah diinflasi untuk visualisasi
        inflated_map_msg = OccupancyGrid()
        inflated_map_msg.header = self.map_header
        inflated_map_msg.info = self.map_info
        inflated_map_msg.data = self.inflated_map_data.flatten().tolist()
        self.inflated_map_pub.publish(inflated_map_msg)

    def map_callback(self, msg):
        """Callback untuk menyimpan data peta dan memicu inflasi."""
        self.map_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_frame = msg.header.frame_id
        # Simpan info dan header untuk publikasi ulang
        self.map_info = msg.info
        self.map_header = msg.header
        rospy.loginfo_once("Map received successfully! Triggering inflation...")
        self.inflate_map() # Langsung proses peta saat diterima

    def world_to_grid(self, world_x, world_y):
        grid_x = int((world_x - self.map_origin.position.x) / self.map_resolution)
        grid_y = int((world_y - self.map_origin.position.y) / self.map_resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        world_x = (grid_x + 0.5) * self.map_resolution + self.map_origin.position.x
        world_y = (grid_y + 0.5) * self.map_resolution + self.map_origin.position.y
        return world_x, world_y
        
    def is_valid(self, x, y):
        """Cek apakah sebuah sel valid pada PETA YANG SUDAH DIINFLASI."""
        if x < 0 or x >= self.map_width or y < 0 or y >= self.map_height:
            return False
        # Gunakan peta yang sudah ditebalkan untuk pengecekan
        if self.inflated_map_data[y][x] > self.obstacle_threshold:
            return False
        return True

    def heuristic(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def a_star_planning(self, start_grid, goal_grid):
        rospy.loginfo(f"A* planning from {start_grid} to {goal_grid}")
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == goal_grid:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_grid)
                return path[::-1]

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not self.is_valid(neighbor[0], neighbor[1]):
                    continue
                tentative_g_score = g_score[current] + self.heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        rospy.logwarn("A* failed to find a path.")
        return None

    def goal_callback(self, msg):
        if self.inflated_map_data is None:
            rospy.logwarn("Cannot plan, map has not been inflated yet.")
            return

        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.map_frame, self.robot_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Failed to get robot pose: {e}")
            return

        start_world = (trans[0], trans[1])
        goal_world = (msg.pose.position.x, msg.pose.position.y)
        start_grid = self.world_to_grid(start_world[0], start_world[1])
        goal_grid = self.world_to_grid(goal_world[0], goal_world[1])

        if not self.is_valid(goal_grid[0], goal_grid[1]):
            rospy.logwarn("Goal is inside an obstacle or inflated area. Cannot plan.")
            return

        path_grid = self.a_star_planning(start_grid, goal_grid)
        if path_grid:
            path_msg = Path()
            path_msg.header.stamp = rospy.Time.now()
            path_msg.header.frame_id = self.map_frame
            for p_grid in path_grid:
                p_world = self.grid_to_world(p_grid[0], p_grid[1])
                pose = PoseStamped()
                pose.header.stamp = rospy.Time.now()
                pose.header.frame_id = self.map_frame
                pose.pose.position.x = p_world[0]
                pose.pose.position.y = p_world[1]
                path_msg.poses.append(pose)
            self.path_pub.publish(path_msg)
            rospy.loginfo("Global plan published successfully.")

if __name__ == '__main__':
    try:
        AStarGlobalPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
