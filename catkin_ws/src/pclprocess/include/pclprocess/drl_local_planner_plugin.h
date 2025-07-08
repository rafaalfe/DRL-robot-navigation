#ifndef DRL_LOCAL_PLANNER_PLUGIN_H_
#define DRL_LOCAL_PLANNER_PLUGIN_H_

// Header Wajib ROS
#include <ros/ros.h>
#include <nav_core/base_local_planner.h>
#include <tf2_ros/buffer.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/LaserScan.h>

// Header Torch (LibTorch)
#include <torch/script.h>
#include <torch/torch.h>

// Header Standar C++
#include <string>
#include <vector>
#include <cmath>

namespace drl_local_planner {

  /**
   * @class DrlLocalPlanner
   * @brief Plugin local planner untuk ROS yang menggunakan model Deep Reinforcement Learning (DRL)
   * yang diekspor sebagai TorchScript.
   */
  class DrlLocalPlanner : public nav_core::BaseLocalPlanner {
  public:
    /**
     * @brief Konstruktor default
     */
    DrlLocalPlanner();

    /**
     * @brief Destruktor
     */
    ~DrlLocalPlanner();

    /**
     * @brief Menginisialisasi plugin planner.
     * @param name Nama dari planner ini.
     * @param tf Pointer ke buffer transform tf2.
     * @param costmap_ros Pointer ke wrapper costmap yang digunakan oleh planner.
     */
    void initialize(std::string name, tf2_ros::Buffer* tf, costmap_2d::Costmap2DROS* costmap_ros);

    /**
     * @brief Mengatur rencana global baru untuk diikuti.
     * @param plan Rencana yang akan diikuti.
     * @return True jika rencana berhasil diatur.
     */
    bool setPlan(const std::vector<geometry_msgs::PoseStamped>& plan);

    /**
     * @brief Fungsi utama yang menghitung perintah kecepatan.
     * @param cmd_vel Perintah kecepatan (Twist) yang akan diisi.
     * @return True jika perintah kecepatan yang valid berhasil dihitung.
     */
    bool computeVelocityCommands(geometry_msgs::Twist& cmd_vel);

    /**
     * @brief Mengecek apakah robot telah mencapai tujuan akhir.
     * @return True jika tujuan telah tercapai.
     */
    bool isGoalReached();

  private:
    /**
     * @brief Callback untuk data LaserScan.
     * @param msg Pesan LaserScan yang diterima.
     */
    void laserCallback(const sensor_msgs::LaserScan::ConstPtr& msg);

    /**
     * @brief Mendapatkan pose robot saat ini dari costmap.
     * @param global_pose Pose robot yang akan diisi.
     * @return True jika pose berhasil didapatkan.
     */
    bool getRobotPose(geometry_msgs::PoseStamped& global_pose) const;

    // --- Variabel Member ---

    bool initialized_;                      // Flag status inisialisasi
    tf2_ros::Buffer* tf_;                   // Pointer ke buffer TF2
    costmap_2d::Costmap2DROS* costmap_ros_; // Pointer ke costmap
    std::vector<geometry_msgs::PoseStamped> global_plan_; // Rencana global saat ini

    // Torch / LibTorch
    torch::jit::script::Module module_;     // Modul model TorchScript
    torch::Device device_;                  // Perangkat untuk inferensi (CPU/GPU)

    // ROS
    ros::Subscriber laser_sub_;             // Subscriber untuk data Lidar/Laser
    sensor_msgs::LaserScan::ConstPtr last_laser_scan_; // Pesan laser terakhir yang diterima
    bool new_laser_data_;                   // Flag untuk menandakan data laser baru

    // Parameter yang dapat dikonfigurasi
    double look_ahead_dist_;                // Jarak pandang ke depan untuk tujuan lokal
    double goal_tolerance_;                 // Toleransi jarak untuk isGoalReached
    std::string laser_topic_;               // Nama topik laser
    int observation_size_;                  // Ukuran total vektor observasi
    int laser_obs_size_;                    // Ukuran data laser dalam observasi
    
    // Faktor penskalaan aksi (HARUS SAMA DENGAN LINGKUNGAN TRAINING)
    double linear_scale_;
    double linear_offset_;
    double angular_scale_;
  };
}; // namespace drl_local_planner

#endif // DRL_LOCAL_PLANNER_PLUGIN_H_
