#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>

/**
 * @brief Kelas untuk mengonversi gambar kedalaman menjadi pesan PointCloud2 3D yang terstruktur (Ordered).
 *
 * Node ini mengambil lima "irisan" horizontal dari gambar kedalaman,
 * menggunakan parameter kalibrasi kamera untuk melakukan deproyeksi menjadi titik-titik 3D (X, Y, Z),
 * dan menghasilkan pesan PointCloud2 yang terstruktur (height=5, width=num_rays).
 */
class DepthToOrderedPointCloud
{
private:
    ros::NodeHandle nh_;
    ros::Subscriber depth_sub_;
    ros::Subscriber info_sub_;
    ros::Publisher pointcloud_pub_;
    
    // Parameter kalibrasi kamera
    double fx_, fy_;
    double cx_, cy_;
    
    // Flag status
    bool intrinsics_received_;
    
    // Parameter output PointCloud
    double range_min_;
    double range_max_;
    double scan_frequency_;
    int num_rays_;
    double angle_min_;
    double angle_max_; 
    
    // Parameter untuk pemrosesan gambar
    // MODIFIKASI: Menambah dua irisan baru
    int slice_height_top_;
    int slice_height_upper_;
    int slice_height_middle_;
    int slice_height_lower_;
    int slice_height_bottom_;
    int slice_thickness_;
    std::string frame_id_;

    int image_width_;
    int image_height_;

public:
    DepthToOrderedPointCloud() : nh_("~"), intrinsics_received_(false)
    {
        // Mengambil parameter dari ROS Parameter Server
        nh_.param("range_min", range_min_, 0.2);
        nh_.param("range_max", range_max_, 9.0);
        nh_.param("slice_thickness", slice_thickness_, 5);
        nh_.param("scan_frequency", scan_frequency_, 30.0);
        nh_.param("num_rays", num_rays_, 70);
        nh_.param("frame_id", frame_id_, std::string("camera_depth_optical_frame"));
        
        nh_.param("angle_min", angle_min_, -1.39626); // Default -80 derajat
        nh_.param("angle_max", angle_max_, 1.39626);  // Default +80 derajat
        
        // MODIFIKASI: Membaca parameter untuk 5 ketinggian
        nh_.param("slice_height_top", slice_height_top_, 50);
        nh_.param("slice_height_upper", slice_height_upper_, -1);
        nh_.param("slice_height_middle", slice_height_middle_, -1);
        nh_.param("slice_height_lower", slice_height_lower_, -1);
        nh_.param("slice_height_bottom", slice_height_bottom_, -1);

        // Setup ROS Subscribers and Publisher
        depth_sub_ = nh_.subscribe("/camera/depth/image_rect_raw", 1, &DepthToOrderedPointCloud::depthCallback, this);
        info_sub_ = nh_.subscribe("/camera/depth/camera_info", 1, &DepthToOrderedPointCloud::infoCallback, this);
        pointcloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/depth_scan_cloud", 10);
        
        ROS_INFO("Node 'depth_to_ordered_pointcloud' initialized. Waiting for camera calibration info...");
        ROS_INFO("PointCloud density set to 5x%d points.", num_rays_);
    }
    
    void infoCallback(const sensor_msgs::CameraInfo::ConstPtr& info_msg)
    {
        if (intrinsics_received_) {
            info_sub_.shutdown();
            return;
        }
        
        fx_ = info_msg->K[0];
        fy_ = info_msg->K[4];
        cx_ = info_msg->K[2];
        cy_ = info_msg->K[5];
        image_width_ = info_msg->width;
        image_height_ = info_msg->height;
        
        if (fx_ <= 0 || fy_ <= 0 || image_width_ <= 0 || image_height_ <= 0) {
            ROS_ERROR("Invalid camera parameters received!");
            return;
        }
        
        intrinsics_received_ = true;
        
        // MODIFIKASI: Mengatur nilai default untuk 5 ketinggian
        if (slice_height_middle_ < 0) slice_height_middle_ = image_height_ / 2;
        if (slice_height_bottom_ < 0) slice_height_bottom_ = image_height_ - 50;
        if (slice_height_upper_ < 0) slice_height_upper_ = slice_height_top_ + (slice_height_middle_ - slice_height_top_) / 2;
        if (slice_height_lower_ < 0) slice_height_lower_ = slice_height_middle_ + (slice_height_bottom_ - slice_height_middle_) / 2;


        ROS_INFO("Camera calibration received: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f, size=%dx%d", 
                 fx_, fy_, cx_, cy_, image_width_, image_height_);
        ROS_INFO("Scan heights set to: Top=%d, Upper=%d, Middle=%d, Lower=%d, Bottom=%d", 
                 slice_height_top_, slice_height_upper_, slice_height_middle_, slice_height_lower_, slice_height_bottom_);
    }
    
    void depthCallback(const sensor_msgs::Image::ConstPtr& depth_msg)
    {
        if (!intrinsics_received_) {
            ROS_WARN_THROTTLE(2, "Waiting for camera calibration info...");
            return;
        }

        try {
            cv_bridge::CvImageConstPtr cv_ptr;
            if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
                cv_ptr = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
            } else if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
                cv_ptr = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
            } else {
                ROS_ERROR("Unsupported depth image encoding: %s", depth_msg->encoding.c_str());
                return;
            }
            
            const cv::Mat& depth_image = cv_ptr->image;
            
            // --- Pembuatan Ordered PointCloud2 ---
            sensor_msgs::PointCloud2 cloud_msg;
            cloud_msg.header = depth_msg->header;
            cloud_msg.header.frame_id = frame_id_;
            cloud_msg.height = 5; // MODIFIKASI: Lima baris untuk lima pindaian
            cloud_msg.width  = num_rays_;
            cloud_msg.is_dense = false; 
            cloud_msg.is_bigendian = false;
            
            sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
            modifier.setPointCloud2FieldsByString(1, "xyz");
            
            sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
            sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
            sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");

            // MODIFIKASI: Vector dengan 5 ketinggian irisan
            std::vector<int> slice_rows = {slice_height_top_, slice_height_upper_, slice_height_middle_, slice_height_lower_, slice_height_bottom_};
            double angle_increment = (angle_max_ - angle_min_) / (num_rays_ - 1);

            // MODIFIKASI: Iterasi untuk setiap 5 baris pindaian
            for (int slice_idx = 0; slice_idx < 5; ++slice_idx) {
                int center_row = slice_rows[slice_idx];
                int slice_start = std::max(0, center_row - slice_thickness_ / 2);
                int slice_end = std::min(image_height_ - 1, center_row + (slice_thickness_ - 1) / 2);

                for (int ray_idx = 0; ray_idx < num_rays_; ++ray_idx) {
                    double target_angle = angle_min_ + ray_idx * angle_increment;
                    double target_col = fx_ * tan(target_angle) + cx_;

                    if (target_col < 0 || target_col >= image_width_) {
                        *(iter_x + slice_idx * num_rays_ + ray_idx) = std::numeric_limits<float>::quiet_NaN();
                        *(iter_y + slice_idx * num_rays_ + ray_idx) = std::numeric_limits<float>::quiet_NaN();
                        *(iter_z + slice_idx * num_rays_ + ray_idx) = std::numeric_limits<float>::quiet_NaN();
                        continue;
                    }
                    
                    std::vector<double> valid_depths;
                    for (int row = slice_start; row <= slice_end; ++row) {
                        double depth_value = getPixelDepth(depth_image, row, static_cast<int>(target_col));
                        if (depth_value > 0) valid_depths.push_back(depth_value);
                    }
                    
                    if (!valid_depths.empty()) {
                        std::sort(valid_depths.begin(), valid_depths.end());
                        double depth_z_raw = valid_depths[valid_depths.size() / 2];

                        double depth_z_meters = (depth_image.type() == CV_16UC1) ? depth_z_raw / 4000.0 : depth_z_raw;
                        
                        if (depth_z_meters >= range_min_ && depth_z_meters <= range_max_) {
                            float x = (static_cast<float>(target_col) - cx_) * depth_z_meters / fx_;
                            float y = (static_cast<float>(center_row) - cy_) * depth_z_meters / fy_;
                            float z = depth_z_meters;

                            *(iter_x + slice_idx * num_rays_ + ray_idx) = x;
                            *(iter_y + slice_idx * num_rays_ + ray_idx) = y;
                            *(iter_z + slice_idx * num_rays_ + ray_idx) = z;
                        } else {
                            *(iter_x + slice_idx * num_rays_ + ray_idx) = std::numeric_limits<float>::quiet_NaN();
                            *(iter_y + slice_idx * num_rays_ + ray_idx) = std::numeric_limits<float>::quiet_NaN();
                            *(iter_z + slice_idx * num_rays_ + ray_idx) = std::numeric_limits<float>::quiet_NaN();
                        }
                    } else {
                        *(iter_x + slice_idx * num_rays_ + ray_idx) = std::numeric_limits<float>::quiet_NaN();
                        *(iter_y + slice_idx * num_rays_ + ray_idx) = std::numeric_limits<float>::quiet_NaN();
                        *(iter_z + slice_idx * num_rays_ + ray_idx) = std::numeric_limits<float>::quiet_NaN();
                    }
                }
            }
            pointcloud_pub_.publish(cloud_msg);
        }
        catch (const cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }

private:
    double getPixelDepth(const cv::Mat& depth_image, int row, int col) {
        if (row < 0 || row >= image_height_ || col < 0 || col >= image_width_) return 0.0;
        if (depth_image.type() == CV_16UC1) {
            return static_cast<double>(depth_image.at<uint16_t>(row, col));
        } else { // CV_32FC1
            return static_cast<double>(depth_image.at<float>(row, col));
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "depth_to_ordered_pointcloud_node");
    
    try {
        DepthToOrderedPointCloud converter;
        ros::spin();
    }
    catch (const std::exception& e) {
        ROS_FATAL("Exception in main: %s", e.what());
        return -1;
    }
    
    return 0;
}
