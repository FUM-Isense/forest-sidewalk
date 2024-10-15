#ifndef ASTARPLANNER_HPP
#define ASTARPLANNER_HPP

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>
#include <queue>
#include <cmath>
#include <limits>

// Define a hashing function for std::pair<int, int> for unordered_map
struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        return std::hash<T1>()(p.first) ^ std::hash<T2>()(p.second);
    }
};

// A* Planner class
class AStarPlanner {
public:
    AStarPlanner(const cv::Mat& global_map);

    std::vector<std::pair<int, int>> plan(const std::pair<int, int>& start, const std::pair<int, int>& goal);

private:
    struct Node {
        double cost;
        std::pair<int, int> cell;
        Node(double c, std::pair<int, int> p) : cost(c), cell(p) {}
    };

    struct NodeComparator {
        bool operator()(const Node& a, const Node& b) {
            return a.cost > b.cost; // Min-heap behavior
        }
    };

    const std::vector<std::pair<int, int>> neighbors = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    int rows_, cols_, safe_distance_;
    cv::Mat global_map_;
    std::vector<std::vector<int>> distances_to_obstacles_;

    double heuristic(const std::pair<int, int>& cell, const std::pair<int, int>& goal);
    std::vector<std::pair<int, int>> reconstructPath(const std::unordered_map<std::pair<int, int>, std::pair<int, int>, PairHash>& came_from,
                                                     std::pair<int, int> current);
    bool isValid(const std::pair<int, int>& cell) const;
    std::vector<std::vector<int>> computeDistancesToObstacles(const cv::Mat& grid);
};

#endif // ASTARPLANNER_HPP
