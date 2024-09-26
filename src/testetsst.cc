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
#include <queue>
#include <vector>
#include <cmath>

// Struct for storing grid coordinates and the cost
struct Node {
    int x, y;
    float cost, heuristic;
    
    Node(int x_, int y_, float cost_, float heuristic_) 
        : x(x_), y(y_), cost(cost_), heuristic(heuristic_) {}
    
    // Comparator to prioritize nodes with lower total cost (cost + heuristic)
    bool operator<(const Node& other) const {
        return (cost + heuristic) > (other.cost + other.heuristic);
    }
};

// Function to compute the heuristic (Manhattan distance)
float heuristic(int x1, int y1, int x2, int y2) {
    return std::abs(x1 - x2) + std::abs(y1 - y2);  // Manhattan distance
}

// A* pathfinding function
std::vector<std::pair<int, int>> a_star(const cv::Mat& map, int start_x, int start_y, int goal_x, int goal_y) {
    std::priority_queue<Node> open_set;
    std::vector<std::vector<bool>> closed_set(map.rows, std::vector<bool>(map.cols, false));
    std::vector<std::vector<std::pair<int, int>>> came_from(map.rows, std::vector<std::pair<int, int>>(map.cols, {-1, -1}));

    open_set.push(Node(start_x, start_y, 0.0f, heuristic(start_x, start_y, goal_x, goal_y)));

    std::vector<std::pair<int, int>> directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};  // 4 directions

    while (!open_set.empty()) {
        Node current = open_set.top();
        open_set.pop();

        // Check if the goal is reached
        if (current.x == goal_x && current.y == goal_y) {
            std::vector<std::pair<int, int>> path;
            while (came_from[current.x][current.y].first != -1) {
                path.emplace_back(current.x, current.y);
                std::tie(current.x, current.y) = came_from[current.x][current.y];
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        // Mark the current node as closed
        closed_set[current.x][current.y] = true;

        // Explore neighbors
        for (const auto& dir : directions) {
            int neighbor_x = current.x + dir.first;
            int neighbor_y = current.y + dir.second;

            // Check if the neighbor is within bounds and not an obstacle or visited
            if (neighbor_x >= 0 && neighbor_x < map.rows && neighbor_y >= 0 && neighbor_y < map.cols &&
                map.at<uint8_t>(neighbor_x, neighbor_y) == 0 && !closed_set[neighbor_x][neighbor_y]) {
                
                float new_cost = current.cost + 1.0f;  // Cost for moving to neighbor
                open_set.push(Node(neighbor_x, neighbor_y, new_cost, heuristic(neighbor_x, neighbor_y, goal_x, goal_y)));
                came_from[neighbor_x][neighbor_y] = {current.x, current.y};
            }
        }
    }

    return {};  // Return empty path if no path is found
}

class PointCloudProcessor : public rclcpp::Node {
public:
    PointCloudProcessor() : Node("pointcloud_processor"), global_map(1000, 1000, CV_8UC1, cv::Scalar(0)) {
        // Subscribe to the pointcloud and odometry topics
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/odom_last_frame", rclcpp::QoS(10).best_effort(),
            std::bind(&PointCloudProcessor::pointCloudCallback, this, std::placeholders::_1)
        );

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&PointCloudProcessor::odomCallback, this, std::placeholders::_1)
        );

        // Initializing the confidence matrix
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 100; j++) {
                confidence_matrix[i][j] = 0;  // Initialize the array with zeros
                last_confidence[i][j] = 0;  // Initialize the array with zeros
            }
        }
        
    }
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
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
        seg.setDistanceThreshold(0.1);  // Adjust based on precision required
        seg.setInputCloud(transformed_cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) {
            RCLCPP_WARN(this->get_logger(), "No ground plane detected.");
            return;
        }

        // Extract the outliers (points not part of the ground plane)
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(transformed_cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);  // Keep the outliers (non-ground points)
        pcl::PointCloud<pcl::PointXYZ>::Ptr outlier_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        extract.filter(*outlier_cloud);

        // Filter the point cloud to only include points where x < 3.0
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PassThrough<pcl::PointXYZ> pass_x;
        pass_x.setInputCloud(outlier_cloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(-std::numeric_limits<float>::max(), 2.0);  // Keep points where x < 3.0
        pass_x.filter(*filtered_cloud);


        // Initialize point count grid (500x400)
        cv::Mat point_count_grid = cv::Mat::zeros(500, 400, CV_32SC1);

        // Iterate over the point cloud and populate the point count grid
        for (const auto& point : filtered_cloud->points) {
            // Convert point coordinates to grid indices
            int x = static_cast<int>(point.x * 100);  // Scaling and shifting point (x)
            int y = static_cast<int>(point.y * 100);  // Scaling and shifting point (y)
            
            
            // Ensure the indices are within grid bounds
            if (x >= 0 && x < 500 && y >= -200 && y < 200) {
                point_count_grid.at<int>(500 - x, 400 - (y + 200))++;  // Increment the point count in the grid cell
            }
        }

        cv::Mat occupancy_grid = cv::Mat::zeros(500, 400, CV_8UC1);
        int step = 10;
        bool occupide_cell = false;
        // Iterate over the point count grid and apply the threshold
        for (int row = 300; row < 500; row += step) {
            for (int col = 0; col < 400; col += step) {
                // Loop through each 5x5 patch
                for (int i = 0; i < step; i++) {
                    for (int j = 0; j < step; j++) {
                        if (point_count_grid.at<int>(row + i, col + j) >= 1){ 
                            occupide_cell = true;
                            break;
                        }
                    }
                    if (occupide_cell) break;
                }
                // Check if the patch is dense enough and mark it
                if (occupide_cell) {
                    for (int i = 0; i < step; i++) {
                        for (int j = 0; j < step; j++) {
                            if ((row + i < 500) && (col + j < 400)) {
                                occupancy_grid.at<uint8_t>(row + i, col + j) = 1;
                            }
                        }
                    }
                    occupide_cell = false;
                }
            }
        }

        // Visualize the occupancy grid using OpenCV
        cv::Mat displayLocalGrid;
        point_count_grid.convertTo(displayLocalGrid, CV_8UC1, 255);  // Convert occupancy grid to 8-bit for display

        cv::imshow("Local Map", displayLocalGrid);  // Show the grid
        if (cv::waitKey(1) == 27) return;  // Exit on ESC key

        updateGlobalMap(occupancy_grid);

        // Visualize the occupancy grid using OpenCV
        cv::Mat displayGlobalGrid;
        global_map.convertTo(displayGlobalGrid, CV_8UC1, 255);  // Convert occupancy grid to 8-bit for display

        cv::imshow("Global Map", displayGlobalGrid);  // Show the grid
        if (cv::waitKey(1) == 27) return;  // Exit on ESC key


        // // Call A* pathfinding
        // int start_x = static_cast<int>(999);
        // if ((1000 - (current_odom_x * 100)) < 1000) int start_x = static_cast<int>(999 - (current_odom_x * 100));
        // int start_y = static_cast<int>(500 - (current_odom_y * 100));
        // int goal_x = 500;  // Goal point (500, 500)
        // int goal_y = 500;

        // // Run A* to get the path
        // std::vector<std::pair<int, int>> path = a_star(global_map, start_x, start_y, goal_x, goal_y);

        // // Draw the path on the global map
        // for (const auto& [x, y] : path) {
        //     global_map.at<uint8_t>(x, y) = 150;  // Use a different value (e.g., 150) for the path
        // }

        // // Visualize the global map with the path
        // cv::Mat displayGlobalGrid;
        // global_map.convertTo(displayGlobalGrid, CV_8UC1, 255);  // Convert occupancy grid to 8-bit for display
        // cv::imshow("Global Map with Path", displayGlobalGrid);  // Show the global map with the path

        // if (cv::waitKey(1) == 27) return;  // Exit on ESC key
    }
    // Callback to store odometry data
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        // Get the odometry data for Global map (position and orientation)
        current_odom_x = msg->pose.pose.position.x;
        current_odom_y = msg->pose.pose.position.y;
        current_odom_theta = msg->pose.pose.orientation.z; 


        // Extract the quaternion and position from odometry
        tf2::Quaternion q(
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z,
            msg->pose.pose.orientation.w
        );
        
        tf2::Vector3 position(
            msg->pose.pose.position.x,
            msg->pose.pose.position.y,
            msg->pose.pose.position.z
        );

        // Create a transform from odometry
        tf2::Transform odom_transform;
        odom_transform.setRotation(q);
        odom_transform.setOrigin(position);

        // Store the inverse of the odometry transform
        odom_to_start_ = odom_transform.inverse();
    }



    void updateGlobalMap(const cv::Mat& local_grid) {
        // Translate the local grid based on the current odometry
        cv::Mat transform = cv::getRotationMatrix2D(cv::Point2f(local_grid.cols / 2, local_grid.rows / 2), 
                                                    current_odom_theta * 180.0 / M_PI, 1.0);
        // Add translation based on odometry
        transform.at<double>(0, 2) += (current_odom_x * 100 + 300);
        transform.at<double>(1, 2) += (current_odom_y * 100 + 500);

        // Create a temporary global grid to hold the transformed local grid
        cv::Mat transformed_local_grid;
        // cv::warpAffine(local_grid, transformed_local_grid, transform, global_map.size(), cv::INTER_NEAREST, cv::BORDER_TRANSPARENT);
        cv::warpAffine(local_grid, transformed_local_grid, transform, global_map.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

        
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 100; j++) {
                if (last_confidence[i][j] > 0) { // Comparing 1
                    if (confidence_matrix[i][j] > 10) { // Confidental point
                        continue;
                    }
                    else if (confidence_matrix[i][j] == 10) { // New point
                        insertPoint(i*10, j*10);
                        confidence_matrix[i][j]++;
                    }
                    else if (confidence_matrix[i][j] >= 0) { // Possible Point
                        confidence_matrix[i][j]++;
                    }
                    // RCLCPP_INFO(this->get_logger(), "2");
                }
                else if (last_confidence[i][j] == 0) { // Comparing 0s
                    if (confidence_matrix[i][j] >= 10) { // Confidental point
                        continue;
                    }
                    else if (confidence_matrix[i][j] >= 0) { // Noise Point
                        confidence_matrix[i][j]= 0;
                    }
                // RCLCPP_INFO(this->get_logger(), "3");
                }
            }
        }

        // Update the last confidence matrix
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 100; j++) {
                if (transformed_local_grid.at<uint8_t>(i*10, j*10) == 1) {
                    last_confidence[i][j] = 1;
                }
                else {
                    last_confidence[i][j] = 0;
                }
            }
        }

    }

    void insertPoint(int row, int col){
        for (int i = row; i < row+10; i++) {
            for (int j = col; j < col+10; j++) {
                global_map.at<uint8_t>(i, j) = 1;
            }
        }
    }

private:

    // Subscribers for the pointcloud and odometry
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;


    // Variable to store the odometry transformation
    tf2::Transform odom_to_start_;


    // Variables to store odometry data
    double current_odom_x = 0.0;
    double current_odom_y = 0.0;
    double current_odom_theta = 0.0;

    // Global occupancy grid
    cv::Mat global_map;

    // Convidence matrix
    int confidence_matrix[100][100];

    // Last confidence matrix
    int last_confidence[100][100];

};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudProcessor>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
