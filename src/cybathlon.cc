#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <cv_bridge/cv_bridge.h>
#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <thread>
#include <mutex>
#include <vector>
    
class DepthImageProcessor : public rclcpp::Node {
public:
    DepthImageProcessor() : Node("depth_image_processor") {
        // Create depth image subscription in the main thread
        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/camera/depth/image_rect_raw", 10,
            std::bind(&DepthImageProcessor::depthCallback, this, std::placeholders::_1)
        );
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&DepthImageProcessor::odomCallback, this, std::placeholders::_1)
        );

        // Initialize the global map (adjust size as necessary)
        global_map = cv::Mat::zeros(500, 400, CV_8UC1);  // Global map initialization

        vis.CreateVisualizerWindow("Open3D", 640, 480);
    }

    // Callback to handle odom data
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        current_pose_ = msg->pose.pose;  // Store current pose
        RCLCPP_INFO(this->get_logger(), "Odometry received: x=%.2f, y=%.2f",
                    current_pose_.position.x * 100, current_pose_.position.y * 100);
    }

    // Callback to handle depth images
    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            // Convert ROS 2 depth image to OpenCV Mat
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
            cv::Mat depth_image = cv_ptr->image;

            // Threshold to keep depth values under 5 meters
            double max_distance = 5000.0;
            cv::threshold(depth_image, depth_image, max_distance, 0, cv::THRESH_TOZERO_INV);

            // Convert depth image to Open3D format
            auto depth_image_o3d = std::make_shared<open3d::geometry::Image>();
            depth_image_o3d->Prepare(depth_image.cols, depth_image.rows, 1, 2);
            memcpy(depth_image_o3d->data_.data(), depth_image.data, depth_image.total() * depth_image.elemSize());

            // Convert depth image to point cloud
            open3d::camera::PinholeCameraIntrinsic intrinsics(640, 480, 380.570, 380.570, 321.218, 237.158);
            auto pcd = open3d::geometry::PointCloud::CreateFromDepthImage(*depth_image_o3d, intrinsics);

            // Apply a flip to correct the orientation
            Eigen::Matrix4d flip_transform = Eigen::Matrix4d::Identity();
            flip_transform(1, 1) = -1;
            flip_transform(2, 2) = -1;
            pcd->Transform(flip_transform);

            if (pcd->points_.empty()) return;

            RCLCPP_INFO(this->get_logger(), "Processing depth image");

            // Convert point cloud to local occupancy grid (2D)
            cv::Mat occupancy_grid = createLocalOccupancyGrid(pcd);
            cv::Mat displayLocalGrid;
            occupancy_grid.convertTo(displayLocalGrid, CV_8UC1, 255);

            // Transform local grid to global map using odometry
            cv::Mat transformed_grid;
            transformed_grid = transformLocalToGlobal(occupancy_grid, current_pose_);

            // Ensure both maps have the same size and type
            if (transformed_grid.size() != global_map.size()) {
                cv::resize(transformed_grid, transformed_grid, global_map.size());
            }

            transformed_grid.convertTo(transformed_grid, global_map.type());  // Ensure both are of the same type

            // Update global map
            updateGlobalMap(transformed_grid);

            cv::Mat displayGlobalGrid;
            global_map.convertTo(displayGlobalGrid, CV_8UC1, 255);

            // Display the global map
            cv::imshow("Global 2D Map", displayGlobalGrid);
            if (cv::waitKey(1) == 27) return;  // Exit on ESC key

        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }
    void stop() {
        vis.DestroyVisualizerWindow();
    }
private:
    // Method to create local occupancy grid from point cloud
    cv::Mat createLocalOccupancyGrid(const std::shared_ptr<open3d::geometry::PointCloud>& pcd) {
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

        // Perform clustering
        cv::Mat occupancy_grid = cv::Mat::zeros(500, 400, CV_8UC1);
        for (const auto& point : filtered_outlier_points) {
            int x = static_cast<int>((2 + point(0)) * 100);
            int z = static_cast<int>(-point(2) * 100);
            if (z >= 0 && z < 500 && x >= 0 && x < 400) {
                occupancy_grid.at<uint8_t>(z, x) = 1;
            }
        }

        cv::Mat cells = cv::Mat::zeros(500, 400, CV_8UC1);
        // int step = 10;
        // int threshold = 10;
        // for (int patch_row = 0; patch_row < 180; patch_row += step) {
        //     for (int patch_col = 0; patch_col < 380; patch_col += step) {
        //         if (cv::sum(occupancy_grid(cv::Rect(patch_col, patch_row, step, step)))[0] > threshold) {
        //             cells(cv::Rect(patch_col, patch_row, step, step)).setTo(1);
        //         }
        //     }
        // }


        return cells;
    }

    // Method to transform local occupancy grid to global map
    cv::Mat transformLocalToGlobal(const cv::Mat& local_map, const geometry_msgs::msg::Pose& pose) {
        double x = pose.position.x;
        double y = pose.position.y;

        // Convert quaternion to yaw
        double qw = pose.orientation.w;
        double qx = pose.orientation.x;
        double qy = pose.orientation.y;
        double qz = pose.orientation.z;
        double yaw = std::atan2(2.0 * (qy * qw + qx * qz), 1.0 - 2.0 * (qy * qy + qz * qz));
        
        // Invert the Y-axis to match the 2D grid's coordinate system
        y = -y;

        // Create a transformation matrix (rotation + translation)
        cv::Mat transformation = (cv::Mat_<double>(2, 3) <<
            std::cos(yaw), -std::sin(yaw), x * 100 + global_map.cols / 2,  // Convert meters to pixels
            std::sin(yaw), std::cos(yaw), y * 100 + global_map.rows / 2);

        cv::Mat transformed_map;
        cv::warpAffine(local_map, transformed_map, transformation, global_map.size(), cv::INTER_NEAREST);

        // set pose in the global map
        if (y >= -1.9 && y < 1.9 && x >= 0 && x < 5){
            transformed_map(cv::Rect((y + 2) * 100, (4.9 - x)*100, 10, 10)).setTo(1);
        }

        return transformed_map;
    }

    // Method to update the global map with the transformed local map
    void updateGlobalMap(const cv::Mat& transformed_map) {
        cv::bitwise_or(global_map, transformed_map, global_map);  // Combine the maps
    }


    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    geometry_msgs::msg::Pose current_pose_;  // Stores current odometry pose

    cv::Mat global_map;  // Global occupancy grid map

    open3d::visualization::Visualizer vis;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<DepthImageProcessor>();
    RCLCPP_INFO(node->get_logger(), "DepthImageProcessor node initialized");

    // Spin the depth image subscription in the main thread
    rclcpp::spin(node);
    node->stop();
    rclcpp::shutdown();
    return 0;
}
