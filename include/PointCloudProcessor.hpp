#ifndef POINTCLOUDPROCESSOR_HPP
#define POINTCLOUDPROCESSOR_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <chrono>
#include <cstdlib>
#include <vector>
#include "AStarPlanner.hpp"

class PointCloudProcessor : public rclcpp::Node {
public:
    PointCloudProcessor();

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    double fitLineAndGetAngle(const std::vector<std::pair<int, int>>& path);

    // Subscribers for the pointcloud and odometry
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    // Variable to store the odometry transformation
    tf2::Transform odom_to_start_;
    tf2::Transform rotate_yaw_;

    // Current robot pose
    double current_x_, current_y_, current_z_, current_yaw_;

    // Global occupancy grid
    cv::Mat global_occupancy_grid_;
    int global_grid_rows_, global_grid_cols_;
    int x_origin_, y_origin_;

    // Parameters for occupancy grid
    double scaling_factor_;
    int occupancy_grid_rows_, occupancy_grid_cols_;

    // Confidence matrix
    int confidence_matrix[200][200];

    // Last confidence matrix
    int last_confidence[200][200];

    char state = 'none';

    double angle_path = 0;
    int frame_counter = 0;

    std::chrono::steady_clock::time_point start_time_;
};

#endif // POINTCLOUDPROCESSOR_HPP
