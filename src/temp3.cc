#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <cv_bridge/cv_bridge.h>
#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

class DepthImageProcessor : public rclcpp::Node {
public:
    DepthImageProcessor() : Node("depth_image_processor"), global_map(1000, 1000, CV_8UC1, cv::Scalar(0)) {
        // Setup ROS 2 subscription for the depth image
        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/camera/depth/image_rect_raw", 10,
            std::bind(&DepthImageProcessor::depthCallback, this, std::placeholders::_1)
        );

        // Setup ROS 2 subscription for the odometry data
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&DepthImageProcessor::odomCallback, this, std::placeholders::_1)
        );

        // Create publisher for the global OccupancyGrid message
        occupancy_grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/occupancy_grid", 10);

        vis.CreateVisualizerWindow("Open3D", 640, 480);

        // Initializing the confidence matrix
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 100; j++) {
                confidence_matrix[i][j] = 0;  // Initialize the array with zeros
            }
        }

        // Initializing the last confidence matrix
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 100; j++) {
                last_confidence[i][j] = 0;  // Initialize the array with zeros
            }
        }
    }

    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            // Convert the ROS 2 image to OpenCV Mat using cv_bridge
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
            cv::Mat depth_image = cv_ptr->image;

            // Apply threshold filter to show only depth values less than 5 meters (5000 mm)
            double max_distance = 5000.0;
            cv::threshold(depth_image, depth_image, max_distance, 0, cv::THRESH_TOZERO_INV);

            // Ensure the depth image is of the correct type for Open3D
            auto depth_image_o3d = std::make_shared<open3d::geometry::Image>();
            depth_image_o3d->Prepare(depth_image.cols, depth_image.rows, 1, 2);
            memcpy(depth_image_o3d->data_.data(), depth_image.data, depth_image.total() * depth_image.elemSize());

            // Convert depth image to point cloud
            // open3d::camera::PinholeCameraIntrinsic intrinsics(640, 480, 380.570, 380.570, 321.218, 237.158); // D435i
            // open3d::camera::PinholeCameraIntrinsic intrinsics(640, 480, 389.861, 389.861, 318.562, 239.314); // D456
            open3d::camera::PinholeCameraIntrinsic intrinsics(1280, 720, 649.768, 649.768, 637.603, 358.857); // D456 HD

            auto pcd = open3d::geometry::PointCloud::CreateFromDepthImage(*depth_image_o3d, intrinsics);

            Eigen::Matrix4d flip_transform = Eigen::Matrix4d::Identity();
            flip_transform(1, 1) = -1;
            flip_transform(2, 2) = -1;
            pcd->Transform(flip_transform);

            if (pcd->points_.empty()) return;

            // Segment plane
            Eigen::Vector4d plane_model;
            std::vector<size_t> inliers;
            std::tie(plane_model, inliers) = pcd->SegmentPlane(0.05, 3, 20);

            // Get the plane normal vector
            Eigen::Vector3d plane_normal = plane_model.head<3>();

            // Calculate rotation to align the plane normal with the Z-axis (up)
            Eigen::Vector3d target_normal(0, 1, 0);
            Eigen::Vector3d v = plane_normal.cross(target_normal);
            double s = v.norm();
            double c = plane_normal.dot(target_normal);

            if (s != 0) {
                Eigen::Matrix3d vx = Eigen::Matrix3d::Zero();
                vx(0, 1) = -v(2);
                vx(0, 2) = v(1);
                vx(1, 0) = v(2);
                vx(1, 2) = -v(0);
                vx(2, 0) = -v(1);
                vx(2, 1) = v(0);

                Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity() + vx + (vx * vx) * ((1 - c) / (s * s));
                pcd->Rotate(rotation_matrix, Eigen::Vector3d(0, 0, 0));
            }

            // Extract inlier and outlier point clouds
            auto inlier_cloud = pcd->SelectByIndex(inliers);
            auto outlier_cloud = pcd->SelectByIndex(inliers, true);

            double ground_reference_y = 0.0;
            if (!inlier_cloud->points_.empty()) {
                std::vector<double> y_values;
                y_values.reserve(inlier_cloud->points_.size());
                for (const auto& point : inlier_cloud->points_) {
                    y_values.push_back(point(1));
                }

                std::sort(y_values.begin(), y_values.end());
                size_t mid_index = y_values.size() / 2;
                if (y_values.size() % 2 == 0) {
                    ground_reference_y = (y_values[mid_index - 1] + y_values[mid_index]) / 2.0;
                } else {
                    ground_reference_y = y_values[mid_index];
                }
            }

            // Filter to remove points below the ground reference minus margin
            std::vector<Eigen::Vector3d> filtered_outlier_points;
            for (const auto& point : outlier_cloud->points_) {
                if (point(1) > (ground_reference_y + 0.2) &&
                    point(1) < -0.1 &&
                    point(2) > -2.0 &&
                    point(2) < -0.5) {
                    filtered_outlier_points.push_back(point);
                }
            }

            // Create new point cloud for filtered points
            auto filtered_cloud = std::make_shared<open3d::geometry::PointCloud>();
            filtered_cloud->points_ = filtered_outlier_points;
            filtered_cloud->PaintUniformColor(Eigen::Vector3d(0, 1, 0));  // Green
            inlier_cloud->PaintUniformColor(Eigen::Vector3d(1, 0, 0));  // Red

            // Update the visualizer
            vis.ClearGeometries();
            vis.AddGeometry(inlier_cloud);
            vis.AddGeometry(filtered_cloud);
            vis.PollEvents();
            vis.UpdateRender();

            // Perform clustering and create occupancy grid
            cv::Mat occupancy_grid = cv::Mat::zeros(500, 400, CV_8UC1);
            for (const auto& point : filtered_outlier_points) {
                int x = static_cast<int>((2 + point(0)) * 100);
                int z = static_cast<int>(500 + point(2) * 100);
                if (z >= 0 && z < 500 && x >= 0 && x < 400) {
                    occupancy_grid.at<uint8_t>(z, x) = 1;
                }
            }

            cv::Mat cells = cv::Mat::zeros(500, 400, CV_8UC1);
            int step = 10;
            int threshold = 10;
            for (int patch_row = 0; patch_row < 500; patch_row += step) {
                for (int patch_col = 0; patch_col < 400; patch_col += step) {
                    if (cv::sum(occupancy_grid(cv::Rect(patch_col, patch_row, step, step)))[0] > threshold) {
                        cells(cv::Rect(patch_col, patch_row, step, step)).setTo(1);
                    }
                }
            }
            
            // Rotate the cells matrix by 90 degrees to the left (counterclockwise)
            cv::transpose(cells, cells);
            cv::flip(cells, cells, 1);  // Flip the transposed matrix vertically

            // Flip the resulting matrix along the X-axis
            cv::flip(cells, cells, 0);  // Flip horizontally to mirror in the X-axis
            
            // Now use odometry data to transform and merge into global grid
            updateGlobalMap(cells);

            // Publish the global occupancy grid
            publishGlobalOccupancyGrid();

        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        // Get the odometry data (position and orientation)
        current_odom_x = msg->pose.pose.position.x;
        current_odom_y = msg->pose.pose.position.y;
        current_odom_theta = msg->pose.pose.orientation.z; 
    }

    void updateGlobalMap(const cv::Mat& local_grid) {
        // Translate the local grid based on the current odometry
        cv::Mat transform = cv::getRotationMatrix2D(cv::Point2f(local_grid.cols / 2, local_grid.rows / 2), 
                                                    current_odom_theta * 180.0 / M_PI, 1.0);
        
        // Add translation based on odometry
        transform.at<double>(0, 2) += (current_odom_x * 100);  // Convert meters to grid cells
        transform.at<double>(1, 2) += (current_odom_y * 100 + 300);

        // Create a temporary global grid to hold the transformed local grid
        cv::Mat transformed_local_grid;
        // cv::warpAffine(local_grid, transformed_local_grid, transform, global_map.size(), cv::INTER_NEAREST, cv::BORDER_TRANSPARENT);
        cv::warpAffine(local_grid, transformed_local_grid, transform, global_map.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

        // // Merge the transformed local grid into the global map
        // for (int i = 0; i < transformed_local_grid.rows; ++i) {
        //     for (int j = 0; j < transformed_local_grid.cols; ++j) {
        //         // Only update cells where the local grid has new information (e.g., obstacles or free space)
        //         if (transformed_local_grid.at<uint8_t>(i, j) == 1) {
        //             global_map.at<uint8_t>(i, j) = 1;  // Mark as occupied
        //         }
        //     }
        // }
        
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 100; j++) {
                if (last_confidence[i][j] == 1) { // Comparing 1
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

    void publishGlobalOccupancyGrid() {
        // Prepare the OccupancyGrid message
        nav_msgs::msg::OccupancyGrid occupancy_grid_msg;
        occupancy_grid_msg.header.stamp = this->now();
        occupancy_grid_msg.header.frame_id = "odom";
        occupancy_grid_msg.info.resolution = 0.01;  // Grid cell resolution in meters
        occupancy_grid_msg.info.width = global_map.cols;
        occupancy_grid_msg.info.height = global_map.rows;
        occupancy_grid_msg.info.origin.position.x = 0.0;
        occupancy_grid_msg.info.origin.position.y = -5.0;
        occupancy_grid_msg.info.origin.position.z = 0.0;
        occupancy_grid_msg.info.origin.orientation.w = 1.0;

        // Fill the data into the occupancy grid message
        occupancy_grid_msg.data.resize(global_map.rows * global_map.cols);
        for (int i = 0; i < global_map.rows; ++i) {
            for (int j = 0; j < global_map.cols; ++j) {
                occupancy_grid_msg.data[i * global_map.cols + j] = (global_map.at<uint8_t>(i, j) == 1) ? 100 : 0;
            }
        }

        // Publish the OccupancyGrid message
        occupancy_grid_pub_->publish(occupancy_grid_msg);
    }

    void stop() {
        vis.DestroyVisualizerWindow();
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_grid_pub_;
    open3d::visualization::Visualizer vis;

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

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DepthImageProcessor>();
    rclcpp::spin(node);
    node->stop();
    rclcpp::shutdown();
    return 0;
}
