#include <pluginlib/class_list_macros.h>
#include "pclprocess/drl_local_planner_plugin.h" // Sesuaikan dengan path header Anda

// Daftarkan plugin ini ke ROS
PLUGINLIB_EXPORT_CLASS(drl_local_planner::DrlLocalPlanner, nav_core::BaseLocalPlanner)

namespace drl_local_planner {

  DrlLocalPlanner::DrlLocalPlanner() : 
    initialized_(false), 
    tf_(nullptr), 
    costmap_ros_(nullptr),
    device_(torch::kCPU),
    new_laser_data_(false) {}

  DrlLocalPlanner::~DrlLocalPlanner() {}

  void DrlLocalPlanner::initialize(std::string name, tf2_ros::Buffer* tf, costmap_2d::Costmap2DROS* costmap_ros) {
    if (!initialized_) {
      ros::NodeHandle private_nh("~/" + name);
      ros::NodeHandle nh; // NodeHandle publik untuk subscriber

      tf_ = tf;
      costmap_ros_ = costmap_ros;

      // --- Ambil Parameter dari Server Parameter ROS ---
      private_nh.param("look_ahead_dist", look_ahead_dist_, 1.0);
      private_nh.param("goal_tolerance", goal_tolerance_, 0.3);
      private_nh.param("laser_topic", laser_topic_, std::string("/scan"));
      
      // Ukuran ini KRUSIAL dan harus sesuai dengan training environment
      private_nh.param("observation_size", observation_size_, 1294); 
      private_nh.param("laser_obs_size", laser_obs_size_, 1292);

      // Faktor penskalaan ini juga KRUSIAL
      private_nh.param("linear_scale", linear_scale_, 0.35);
      private_nh.param("linear_offset", linear_offset_, 0.15);
      private_nh.param("angular_scale", angular_scale_, 1.0);
      
      ROS_INFO("DRL Planner Parameters: look_ahead_dist=%.2f, laser_topic=%s, obs_size=%d",
               look_ahead_dist_, laser_topic_.c_str(), observation_size_);

      // Cek apakah GPU tersedia dan set perangkat
      if (torch::cuda::is_available()) {
        device_ = torch::kCUDA;
        ROS_INFO("CUDA is available, using GPU for Torch model.");
      } else {
        ROS_INFO("CUDA not available, using CPU for Torch model.");
      }

      // Muat model TorchScript
      std::string model_path;
      private_nh.param("model_path", model_path, std::string("drl_actor_model.pt"));
      
      try {
        module_ = torch::jit::load(model_path, device_); // Langsung muat ke device
        module_.eval(); // Set model ke mode evaluasi (penting!)
        ROS_INFO("Successfully loaded TorchScript model from %s", model_path.c_str());
      }
      catch (const c10::Error& e) {
        ROS_FATAL("Failed to load TorchScript model: %s. Make sure the model file is in the correct path.", e.what());
        return;
      }

      // Inisialisasi subscriber untuk data laser
      laser_sub_ = nh.subscribe<sensor_msgs::LaserScan>(laser_topic_, 1, &DrlLocalPlanner::laserCallback, this);

      initialized_ = true;
      ROS_INFO("DRL Local Planner C++ Plugin Initialized Successfully.");
    } else {
      ROS_WARN("This planner has already been initialized... doing nothing");
    }
  }

  void DrlLocalPlanner::laserCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
    last_laser_scan_ = msg;
    if (!new_laser_data_) {
      new_laser_data_ = true;
    }
  }

  bool DrlLocalPlanner::setPlan(const std::vector<geometry_msgs::PoseStamped>& plan) {
    if (!initialized_) {
      ROS_ERROR("This planner has not been initialized, please call initialize() before using this planner");
      return false;
    }
    global_plan_ = plan;
    return true;
  }

  bool DrlLocalPlanner::computeVelocityCommands(geometry_msgs::Twist& cmd_vel) {
    if (!initialized_) {
      ROS_ERROR("This planner has not been initialized.");
      return false;
    }

    if (global_plan_.empty()) {
      ROS_WARN("Global plan is empty, stopping the robot.");
      return false;
    }
    
    if (!new_laser_data_ || !last_laser_scan_) {
        ROS_WARN_THROTTLE(1.0, "Waiting for laser scan data on topic %s...", laser_topic_.c_str());
        return false;
    }
    
    // Cek umur data laser untuk keamanan
    if ((ros::Time::now() - last_laser_scan_->header.stamp).toSec() > 0.5) {
        ROS_WARN("Laser scan data is stale, stopping the robot.");
        cmd_vel.linear.x = 0.0;
        cmd_vel.angular.z = 0.0;
        return true; // Berhasil berhenti
    }

    // --- LOGIKA UTAMA ---

    // 1. DAPATKAN STATE ROBOT & TENTUKAN TUJUAN LOKAL
    geometry_msgs::PoseStamped robot_pose;
    if (!getRobotPose(robot_pose)) {
      ROS_ERROR("Failed to get robot pose from costmap.");
      return false;
    }

    // Cari waypoint terdekat dari robot di rencana global
    int closest_waypoint_idx = -1;
    double min_dist_sq = std::numeric_limits<double>::max();
    for (size_t i = 0; i < global_plan_.size(); ++i) {
        double dx = robot_pose.pose.position.x - global_plan_[i].pose.position.x;
        double dy = robot_pose.pose.position.y - global_plan_[i].pose.position.y;
        double dist_sq = dx*dx + dy*dy;
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            closest_waypoint_idx = i;
        }
    }

    // Cari tujuan lokal (look-ahead point) dari waypoint terdekat
    int local_goal_idx = closest_waypoint_idx;
    while(local_goal_idx < global_plan_.size() - 1) {
        double dx = global_plan_[local_goal_idx].pose.position.x - global_plan_[closest_waypoint_idx].pose.position.x;
        double dy = global_plan_[local_goal_idx].pose.position.y - global_plan_[closest_waypoint_idx].pose.position.y;
        if (std::hypot(dx, dy) >= look_ahead_dist_) {
            break;
        }
        local_goal_idx++;
    }
    
    geometry_msgs::PoseStamped local_goal = global_plan_[local_goal_idx];

    // Transformasi tujuan lokal ke frame robot (misal: "base_link")
    geometry_msgs::PoseStamped goal_in_robot_frame;
    try {
        local_goal.header.stamp = ros::Time(0); // Gunakan waktu terbaru
        tf_->transform(local_goal, goal_in_robot_frame, costmap_ros_->getBaseFrameID());
    } catch (tf2::TransformException &ex) {
        ROS_WARN("Could not transform local goal to robot frame: %s", ex.what());
        return false;
    }

    // 2. BANGUN VEKTOR OBSERVASI
    std::vector<float> observation_vec;
    observation_vec.reserve(observation_size_);

    // 2.1. Tambahkan informasi tujuan (jarak & sudut)
    double distance_to_local_goal = std::hypot(goal_in_robot_frame.pose.position.x, goal_in_robot_frame.pose.position.y);
    double angle_to_local_goal = std::atan2(goal_in_robot_frame.pose.position.y, goal_in_robot_frame.pose.position.x);

    // Normalisasi (ASUMSI: sama dengan di environment training)
    observation_vec.push_back(static_cast<float>(distance_to_local_goal / 4.0)); // Contoh: normalisasi dengan jarak maks 4m
    observation_vec.push_back(static_cast<float>(angle_to_local_goal / M_PI));   // Contoh: normalisasi dengan PI

    // 2.2. Tambahkan data sensor (laser)
    if (last_laser_scan_->ranges.size() != laser_obs_size_) {
        ROS_ERROR_THROTTLE(2.0, "Laser scan size (%zu) does not match expected observation size (%d)! Cannot create valid observation.", last_laser_scan_->ranges.size(), laser_obs_size_);
        return false;
    }

    for (float range : last_laser_scan_->ranges) {
        // Proses 'inf' dan 'nan' persis seperti di environment training
        if (std::isinf(range) || std::isnan(range)) {
            range = last_laser_scan_->range_max;
        }
        // Normalisasi (ASUMSI: sama dengan di environment training)
        observation_vec.push_back(range / last_laser_scan_->range_max);
    }
    
    // Verifikasi final
    if (observation_vec.size() != observation_size_) {
        ROS_FATAL("FATAL: Final observation vector size (%zu) mismatch! Expected %d.", observation_vec.size(), observation_size_);
        return false;
    }

    // 3. LAKUKAN INFERENSI
    at::Tensor input_tensor = torch::from_blob(observation_vec.data(), {1, observation_size_}, torch::kFloat32).to(device_);
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);
    
    at::Tensor output_tensor;
    try {
      output_tensor = module_.forward(inputs).toTensor();
    } catch (const c10::Error& e) {
      ROS_ERROR("Error during model inference: %s", e.what());
      return false;
    }

    // 4. KONVERSI OUTPUT KE PERINTAH KECEPATAN
    float linear_vel_action = output_tensor[0][0].item<float>();
    float angular_vel_action = output_tensor[0][1].item<float>();
    
    // Terapkan penskalaan dan offset yang sama dengan environment training
    cmd_vel.linear.x = linear_vel_action * linear_scale_ + linear_offset_;
    cmd_vel.angular.z = angular_vel_action * angular_scale_;

    return true;
  }

  bool DrlLocalPlanner::isGoalReached() {
    if (!initialized_) {
      ROS_ERROR("This planner has not been initialized.");
      return false;
    }
    
    if (global_plan_.empty()) {
        return false; // Tidak ada rencana, tidak ada tujuan
    }

    geometry_msgs::PoseStamped robot_pose;
    if (!getRobotPose(robot_pose)) {
        ROS_ERROR("Failed to get robot pose for goal check.");
        return false;
    }

    const geometry_msgs::PoseStamped& final_goal = global_plan_.back();
    double dist_to_goal = std::hypot(final_goal.pose.position.x - robot_pose.pose.position.x, 
                                     final_goal.pose.position.y - robot_pose.pose.position.y);
    
    if (dist_to_goal < goal_tolerance_) {
        ROS_INFO("Goal reached!");
        return true;
    }

    return false;
  }

  bool DrlLocalPlanner::getRobotPose(geometry_msgs::PoseStamped& global_pose) const {
    return costmap_ros_->getRobotPose(global_pose);
  }

}; // namespace drl_local_planner
