// Copyright 2020 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * Copyright 2015-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ml_controller/ml_controller_node.hpp"

#include "ml_controller/ml_controller_viz.hpp"
#include "ml_controller/util/planning_utils.hpp"
#include "ml_controller/util/tf_utils.hpp"

#include <vehicle_info_util/vehicle_info_util.hpp>

#include <algorithm>
#include <memory>
#include <utility>

namespace
{
double calcLookaheadDistance(
  const double velocity, const double lookahead_distance_ratio, const double min_lookahead_distance)
{
  const double lookahead_distance = lookahead_distance_ratio * std::abs(velocity);
  return std::max(lookahead_distance, min_lookahead_distance);
}

}  // namespace

namespace ml_controller
{
MlControllerNode::MlControllerNode(const rclcpp::NodeOptions & node_options)
: Node("ml_controller", node_options), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
{
  

  // Vehicle Parameters
  // const auto vehicle_info = vehicle_info_util::VehicleInfoUtil(*this).getVehicleInfo();
  // param_.wheel_base = vehicle_info.wheel_base_m;

  // Node Parameters
  //param_.ctrl_period = this->declare_parameter<double>("control_period");

  // Algorithm Parameters
  //param_.lookahead_distance_ratio = this->declare_parameter<double>("lookahead_distance_ratio");
  //param_.min_lookahead_distance = this->declare_parameter<double>("min_lookahead_distance");
  //param_.reverse_min_lookahead_distance =
  //  this->declare_parameter<double>("reverse_min_lookahead_distance");
  //TensorRT Parameters
    const auto model_path = declare_parameter("model_path", "test.onnx");
    const auto precision = declare_parameter("precision", "fp32");
    ml_controller_ = std::make_unique<MlController>(model_path, precision);

  // Subscribers
  using std::placeholders::_1;
  sub_trajectory_ = this->create_subscription<autoware_auto_planning_msgs::msg::Trajectory>(
    "input/reference_trajectory", 1, std::bind(&MlControllerNode::onTrajectory, this, _1));
  sub_current_odometry_ = this->create_subscription<nav_msgs::msg::Odometry>(
    "input/current_odometry", 1, std::bind(&MlControllerNode::onCurrentOdometry, this, _1));

  // Publishers
  pub_ctrl_cmd_ = this->create_publisher<autoware_auto_control_msgs::msg::AckermannLateralCommand>(
    "output/control_raw", 1);

  // Debug Publishers
  pub_debug_marker_ =
    this->create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/markers", 0);

  // Timer
  {
    const auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(param_.ctrl_period));
    timer_ = rclcpp::create_timer(
      this, get_clock(), period_ns, std::bind(&MlControllerNode::onTimer, this));
  }

  //  Wait for first current pose
  tf_utils::waitForTransform(tf_buffer_, "map", "base_link");
}

bool MlControllerNode::isDataReady()
{
  if (!current_odometry_) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "waiting for current_odometry...");
    return false;
  }

  if (!trajectory_) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "waiting for trajectory...");
    return false;
  }

  if (!current_pose_) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "waiting for current_pose...");
    return false;
  }

  return true;
}

void MlControllerNode::onCurrentOdometry(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
  current_odometry_ = msg;
}

void MlControllerNode::onTrajectory(
  const autoware_auto_planning_msgs::msg::Trajectory::ConstSharedPtr msg)
{
  trajectory_ = msg;
}

void MlControllerNode::onTimer()
{
  current_pose_ = self_pose_listener_.getCurrentPose();

  if (!isDataReady()) {
    return;
  }

  const auto result = calcControlSignals();

  if (result.first) {
    //TO DO PUBLISH RESULTS WHEREVER THEY NEED TO GO
    publishCommands(result.second);
    publishDebugMarker();
  } else {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "failed to solve ml_controller");
  }
}

void MlControllerNode::publishCommands(const Object controlSignals)
{
  autoware_auto_control_msgs::msg::AckermannLateralCommand cmd;
  cmd.stamp = get_clock()->now();
  cmd.steering_tire_angle =controlSignals.steering_tire_angle;
  cmd.steering_tire_rotation_rate=controlSignals.steering_tire_rotation_rate;
  pub_ctrl_cmd_->publish(cmd);
}

void MlControllerNode::publishDebugMarker() const
{
  visualization_msgs::msg::MarkerArray marker_array;

  marker_array.markers.push_back(createNextTargetMarker(debug_data_.next_target));
  marker_array.markers.push_back(
    createTrajectoryCircleMarker(debug_data_.next_target, current_pose_->pose));

  pub_debug_marker_->publish(marker_array);
}

std::pair<bool,Object> MlControllerNode::calcControlSignals()
{
  // Ignore invalid trajectory
  if (trajectory_->points.size() < 3) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "received path size is < 3, ignored");
    return {};
  }

  // Calculate target point for velocity/acceleration
  const auto target_point = calcTargetPoint();
  if (!target_point) {
    return {};
  }

  const double target_vel = target_point->longitudinal_velocity_mps;

  // Calculate lookahead distance
  const bool is_reverse = (target_vel < 0);
  const double min_lookahead_distance =
    is_reverse ? param_.reverse_min_lookahead_distance : param_.min_lookahead_distance;
  const double lookahead_distance = calcLookaheadDistance(
    current_odometry_->twist.twist.linear.x, param_.lookahead_distance_ratio,
    min_lookahead_distance);

  // Set PurePursuit data
  ml_controller_->setCurrentPose(current_pose_->pose);
  ml_controller_->setWaypoints(planning_utils::extractPoses(*trajectory_));
  ml_controller_->setLookaheadDistance(lookahead_distance);
  
  // Run PurePursuit
  const auto ml_controller_result = ml_controller_->run();
 
  // Set debug data
  debug_data_.next_target = ml_controller_->getLocationOfNextTarget();

  return ml_controller_result;
}

boost::optional<autoware_auto_planning_msgs::msg::TrajectoryPoint>
MlControllerNode::calcTargetPoint() const
{
  const auto closest_idx_result = planning_utils::findClosestIdxWithDistAngThr(
    planning_utils::extractPoses(*trajectory_), current_pose_->pose, 3.0, M_PI_4);

  if (!closest_idx_result.first) {
    RCLCPP_ERROR(get_logger(), "cannot find closest waypoint");
    return {};
  }

  return trajectory_->points.at(closest_idx_result.second);
}
}  // namespace ml_controller


#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ml_controller::MlControllerNode)
