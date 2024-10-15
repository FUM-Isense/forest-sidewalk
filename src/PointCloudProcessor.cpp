#include "PointCloudProcessor.hpp"
#include <pcl/filters/passthrough.h>
#include <pcl_ros/transforms.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <opencv2/opencv.hpp>

PointCloudProcessor::PointCloudProcessor() 
        : Node("pointcloud_processor"), 
          scaling_factor_(20.0), 
          occupancy_grid_rows_(100), 
          occupancy_grid_cols_(80),
          global_grid_rows_(200), 
          global_grid_cols_(200),
          current_x_(0), current_y_(0), current_z_(0), current_yaw_(0),
          global_occupancy_grid_(cv::Mat::zeros(global_grid_rows_, global_grid_cols_, CV_8UC1)) {
    // Subscribe to the pointcloud and odometry topics
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odometry/filtered", 10,
        std::bind(&PointCloudProcessor::odomCallback, this, std::placeholders::_1)
    );

    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/odom_last_frame", rclcpp::QoS(10).best_effort(),
        std::bind(&PointCloudProcessor::pointCloudCallback, this, std::placeholders::_1)
    );

    // Set the entire 81st column to 1
    global_occupancy_grid_.col(70).setTo(1);
    global_occupancy_grid_.col(130).setTo(1);

    // Initializing the confidence matrix
    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < 200; j++) {
            confidence_matrix[i][j] = 0;  // Initialize the array with zeros
            last_confidence[i][j] = 0;  // Initialize the array with zeros
        }
    }
}

void PointCloudProcessor::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    start_time_ = std::chrono::steady_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *pcl_cloud);

    // Apply the inverse of the odometry transformation (odom_to_start_)
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl_ros::transformPointCloud(*pcl_cloud, *transformed_cloud, odom_to_start_);

    // Detect the ground plane using RANSAC
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.15);  // Adjust based on precision required
    seg.setInputCloud(transformed_cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()){  // If c (z component of the normal) is near 1, assume it's horizontal
        RCLCPP_WARN(this->get_logger(), "No ground plane detected.");
        return;
    }

    Eigen::Vector3f normal_vector(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    Eigen::Vector3f target_normal(0.0, 0.0, 1.0);  // Normal of the XY plane (Z axis)

    // Compute the rotation axis (cross product of the two normals)
    Eigen::Vector3f rotation_axis = normal_vector.cross(target_normal);
    float temp_angle = std::acos(normal_vector.dot(target_normal) / (normal_vector.norm() * target_normal.norm()));

    rotation_axis.normalize();
    tf2::Quaternion temp_quaternion;
    temp_quaternion.setRotation(tf2::Vector3(rotation_axis.x(), rotation_axis.y(), rotation_axis.z()), temp_angle);

    // Create the transform (rotation only, no translation)
    tf2::Transform temp_transform;
    temp_transform.setRotation(temp_quaternion);

    // Apply the rotation to the point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl_ros::transformPointCloud(*transformed_cloud, *aligned_cloud, temp_transform);
    // Extract the outliers (points not part of the ground plane)
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(aligned_cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);  // Keep the outliers (non-ground points)
    pcl::PointCloud<pcl::PointXYZ>::Ptr outlier_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    extract.filter(*outlier_cloud);

    // Filter the point cloud to only include points where x < 3.0
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass_x;
    pass_x.setInputCloud(outlier_cloud);
    pass_x.setFilterFieldName("x");
    pass_x.setFilterLimits(-std::numeric_limits<float>::max(), 2.0 + current_x_);  // Keep points where x < 3.0
    pass_x.filter(*filtered_cloud);
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl_ros::transformPointCloud(*filtered_cloud, *rotated_cloud, odom_to_start_.inverse());

    // Initialize point count grid (1000x1000)
    cv::Mat point_count_grid = cv::Mat::zeros(1000, 1000, CV_32SC1);

    // Iterate over the point cloud and populate the point count grid
    for (const auto& point : rotated_cloud->points) {
        // Convert point coordinates to grid indices
        int x = static_cast<int>(point.x * 100);  // Scaling and shifting point (x)
        int y = static_cast<int>(point.y * 100);  // Scaling and shifting point (y)
        
        // Ensure the indices are within grid bounds
        if (x >= 0 && x < 1000 && y >= -500 && y < 500) {
            point_count_grid.at<int>(999 - x, 999 - (y + 500))++;  // Increment the point count in the grid cell
        }
    }

    cv::Mat occupancy_grid = cv::Mat::zeros(200, 200, CV_8UC1);
    int step = 5;
    int occupide_threshold = 1;
    // Iterate over the point count grid and apply the threshold
    for (int row = 0; row < 1000; row += step) {
        for (int col = 0; col < 1000; col += step) {
            int cell_counter = 0;
            // Loop through each 5x5 patch
            for (int i = 0; i < step; i++) {
                for (int j = 0; j < step; j++) {
                    cell_counter += point_count_grid.at<int>(row + i, col + j);
                }
            }
            // Check if the patch is dense enough and mark it
            if (cell_counter > occupide_threshold) {
                occupancy_grid.at<uint8_t>(row / step, col / step) = 1;
            }
        }
    }

    cv::Mat transformed_local_grid = cv::Mat::zeros(global_grid_rows_, global_grid_cols_, CV_8UC1);;
    // Transform local occupancy grid to global occupancy grid
    for (int row = 0; row < occupancy_grid.rows; ++row) {
        for (int col = 0; col < occupancy_grid.cols; ++col) {
            if (occupancy_grid.at<uint8_t>(row, col) == 1) {
                // Compute local coordinates (in meters)
                double x_local = (occupancy_grid_rows_ / 2 - row) / scaling_factor_;
                double y_local = (col - occupancy_grid_cols_ / 2) / scaling_factor_;

                // Rotate and translate to get global coordinates
                double x_global = current_x_ + cos(current_yaw_) * x_local - sin(current_yaw_) * y_local;
                double y_global = - current_y_ + sin(current_yaw_) * x_local + cos(current_yaw_) * y_local;

                // Map to global grid indices
                int global_row = global_grid_rows_ / 2 - static_cast<int>(x_global * scaling_factor_);
                int global_col = static_cast<int>(y_global * scaling_factor_) + global_grid_cols_ / 2;

                // Ensure indices are within bounds
                if (global_row >= -50 && global_row < global_grid_rows_ - 50 &&
                    global_col >= 0 && global_col < global_grid_cols_) {
                    transformed_local_grid.at<uint8_t>(global_row + 50, global_col) = 1;
                }
            }
        }
    }

    int conf_threshold = 5;
    for (int i = 0; i < global_grid_rows_; i++) {
        for (int j = 0; j < global_grid_cols_; j++) {
            if (last_confidence[i][j] > 0) { // Comparing 1
                if (confidence_matrix[i][j] > conf_threshold) { // Confidental point
                    continue;
                }
                else if (confidence_matrix[i][j] == conf_threshold) { // New point
                    global_occupancy_grid_.at<uint8_t>(i, j) = 1;
                    confidence_matrix[i][j]++;
                }
                else if (confidence_matrix[i][j] >= 0) { // Possible Point
                    confidence_matrix[i][j]++;
                }
            }
            else if (last_confidence[i][j] == 0) { // Comparing 0s
                if (confidence_matrix[i][j] >= conf_threshold) { // Confidental point
                    continue;
                }
                else if (confidence_matrix[i][j] >= 0) { // Noise Point
                    confidence_matrix[i][j]= 0;
                }
            }
        }
    }

    // Update the last confidence matrix
    for (int i = 0; i < global_grid_rows_; i++) {
        for (int j = 0; j < global_grid_cols_; j++) {
            if (occupancy_grid.at<uint8_t>(i, j) == 1) {
                last_confidence[i][j] = 1;
            }
            else {
                last_confidence[i][j] = 0;
            }
        }
    }
    // Visualize the global map with the path
    cv::Mat displayLocalGrid;
    occupancy_grid.convertTo(displayLocalGrid, CV_8UC1, 255);  // Convert occupancy grid to 8-bit for display

    cv::imshow("Local Map", displayLocalGrid);  // Show the grid with path
    if (cv::waitKey(1) == 27) return;  // Exit on ESC key

    // // Visualize the global map with the path
    // cv::Mat displaypointGrid;
    // point_count_grid.convertTo(displaypointGrid, CV_8UC1, 255);  // Convert occupancy grid to 8-bit for display

    // cv::imshow("Local Map 2", displaypointGrid);  // Show the grid with path
    // if (cv::waitKey(1) == 27) return;  // Exit on ESC key

    // Call A* planner
    AStarPlanner planner(global_occupancy_grid_);
    std::pair<int, int> start = {std::min((199 - current_x_ * 20), 199.0), (100 - current_y_ * 20)};
    std::pair<int, int> goal = {100, 100}; // Example goal

    if ((current_x_ * 20 >= 78) || (std::sqrt(std::pow(start.first - goal.first, 2) + std::pow(start.second - goal.second, 2)) < 5.0)){
        system("espeak \"Task Passed!\"");
        exit(0);
    }

    std::vector<std::pair<int, int>> path = planner.plan(start, goal);


    // Visualize the global map with the path
    cv::Mat displayGlobalGrid;
    global_occupancy_grid_.convertTo(displayGlobalGrid, CV_8UC1, 255);  // Convert occupancy grid to 8-bit for display

    cv::Mat rgbGrid;
    cv::cvtColor(displayGlobalGrid, rgbGrid, cv::COLOR_GRAY2BGR);

    for (int i = 0; i < 10; ++i) {
        const auto& p = path[i];
        rgbGrid.at<cv::Vec3b>(p.first, p.second) = cv::Vec3b(0, 0, 255);  // Red for the path
    }

    if (frame_counter == 15) {
        angle_path = fitLineAndGetAngle(path);
        frame_counter = 0;
    }

    double angle_pilot = current_yaw_ * (180.0 / M_PI) + 90;

    double x = angle_path - angle_pilot;
    
    bool audio_feedback = true;  // Set this to true to enable audio feedback
    
    if (x < -20.0) {
        if (state != 'r') {
            RCLCPP_INFO(this->get_logger(), "Right");
            if (audio_feedback) {
                system("espeak \"right\"");
            }
            state = 'r';
        }
    } else if (x > 20.0) {
        if (state != 'l') {
            RCLCPP_INFO(this->get_logger(), "Left");
            if (audio_feedback) {
                system("espeak \"left\"");
            }
            state = 'l';
        }
    } else{
        if (state != 'f') {
            if (std::abs(x) < 10.0){
                RCLCPP_INFO(this->get_logger(), "Forward");
                if (audio_feedback) {
                    system("espeak \"Forward\"");
                }
                state = 'f';
            }
        }
    }
    frame_counter++;

    rgbGrid.at<cv::Vec3b>(std::min((199 - current_x_ * 20), 199.0), (100 - current_y_ * 20)) = cv::Vec3b(0, 255, 0); // current position

    // Resize the grid from 200x200 to 800x800 for better visualization
    cv::Mat enlargedGrid;
    cv::resize(rgbGrid, enlargedGrid, cv::Size(800, 800), 0, 0, cv::INTER_NEAREST);  // Resize using nearest neighbor interpolation
    
    // Display the enlarged RGB matrix with path visualization
    cv::imshow("Enlarged Confidence Matrix in RGB", enlargedGrid);
    if (cv::waitKey(1) == 27) return;  // Exit on ESC key    
}


void PointCloudProcessor::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    current_x_ = msg->pose.pose.position.x;
    current_y_ = msg->pose.pose.position.y;
    current_z_ = msg->pose.pose.position.z;

    // Extract the quaternion from odometry
    tf2::Quaternion q(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w
    );

    // Convert quaternion to roll, pitch, and yaw
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    current_yaw_ = yaw;

    // Create a transformation that only uses the yaw value
    tf2::Transform odom_transform_yaw;
    tf2::Quaternion yaw_quat;
    yaw_quat.setRPY(0, 0, yaw);
    odom_transform_yaw.setRotation(yaw_quat);
    odom_transform_yaw.setOrigin(tf2::Vector3(0, 0, 0));  // Zero out the translation

    // Store the inverse of the yaw-only transformation for point cloud alignment
    odom_to_start_ = odom_transform_yaw.inverse();
}

double PointCloudProcessor::fitLineAndGetAngle(const std::vector<std::pair<int, int>>& path) {
    std::vector<cv::Point2f> points;
    for (int i = 0; i < 10 && i < path.size(); ++i) {
        points.push_back(cv::Point2f(static_cast<float>(path[i].first), static_cast<float>(path[i].second)));
    }

    cv::Mat pointsMat(points.size(), 1, CV_32FC2, points.data());
    cv::Vec4f line;
    cv::fitLine(pointsMat, line, cv::DIST_L2, 0, 0.01, 0.01);

    float vx = line[0];
    float vy = line[1];
    float x0 = line[2];
    float y0 = line[3];

    float slope = vy / vx;
    int x1 = 500;
    int y1 = static_cast<int>(y0 + ((x1 - x0) * slope));

    double delta_x = x1 - x0;
    double delta_y = y1 - y0;
    double angle_radians = std::atan2(delta_y, delta_x);
    double angle_degrees = angle_radians * (180.0 / CV_PI);

    return angle_degrees + 90;
}
