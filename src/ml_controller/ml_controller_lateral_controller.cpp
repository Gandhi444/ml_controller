// Copyright 2020-2022 Tier IV, Inc., Leo Drive Teknoloji A.Ş.
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

#include "ml_controller/ml_controller_lateral_controller.hpp"

#include "ml_controller/ml_controller_viz.hpp"
#include "ml_controller/util/planning_utils.hpp"
#include "ml_controller/util/tf_utils.hpp"

#include <vehicle_info_util/vehicle_info_util.hpp>

#include <algorithm>
#include <memory>
#include <utility>



namespace ml_controller
{
MlLateralController::MlLateralController(rclcpp::Node & node)
: clock_(node.get_clock()),
  logger_(node.get_logger().get_child("lateral_controller")),
  tf_buffer_(clock_),
  tf_listener_(tf_buffer_)
{
  

  // Vehicle Parameters
  const auto vehicle_info = vehicle_info_util::VehicleInfoUtil(node).getVehicleInfo();
  param_.wheel_base = vehicle_info.wheel_base_m;
  param_.max_steering_angle = vehicle_info.max_steer_angle_rad;
  // Algorithm Parameters
  param_.converged_steer_rad_ = node.declare_parameter<double>("converged_steer_rad");
  param_.resampling_ds = node.declare_parameter<double>("resampling_ds");
  param_.model_path = node.declare_parameter<std::string>("model_path", "test.onnx");
  param_.precision = node.declare_parameter<std::string>("precision", "fp32");
  param_.lookahead_distance = node.declare_parameter<double>("lookahead_distance", 0.0);
  param_.closest_thr_dist = node.declare_parameter<double>("closest_thr_dist", 3.0);
  param_.closest_thr_ang = node.declare_parameter<double>("closest_thr_ang", 0.785);
  param_.trajectory_input_points_ = node.declare_parameter<int32_t>("trajectory_input_points_", 10);
  ml_controller_ = std::make_unique<MlController>(param_.model_path, param_.precision,param_.trajectory_input_points_,param_.max_steering_angle);
  // Debug Publishers
  pub_debug_marker_ =
    node.create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/markers", 0);
  pub_debug_values_ = node.create_publisher<tier4_debug_msgs::msg::Float32MultiArrayStamped>(
    "~/debug/ld_outputs", rclcpp::QoS{1});

  // Publish predicted trajectory
  pub_predicted_trajectory_ = node.create_publisher<autoware_auto_planning_msgs::msg::Trajectory>(
    "~/output/predicted_trajectory", 1);
}


bool MlLateralController::isReady([[maybe_unused]] const InputData & input_data)
{
  return true;
}

void MlLateralController::setResampledTrajectory()
{
  // Interpolate with constant interval distance.
  std::vector<double> out_arclength;
  const auto input_tp_array = motion_utils::convertToTrajectoryPointArray(trajectory_);
  const auto traj_length = motion_utils::calcArcLength(input_tp_array);
  for (double s = 0; s < traj_length; s += param_.resampling_ds) {
    out_arclength.push_back(s);
  }
  trajectory_resampled_ =
    std::make_shared<autoware_auto_planning_msgs::msg::Trajectory>(motion_utils::resampleTrajectory(
      motion_utils::convertToTrajectory(input_tp_array), out_arclength));
  trajectory_resampled_->points.back() = trajectory_.points.back();
  trajectory_resampled_->header = trajectory_.header;
  output_tp_array_ = motion_utils::convertToTrajectoryPointArray(*trajectory_resampled_);
}


LateralOutput MlLateralController::run(const InputData & input_data)
{
  current_pose_ = input_data.current_odometry.pose.pose;
  trajectory_ = input_data.current_trajectory;
  current_odometry_ = input_data.current_odometry;
  current_steering_ = input_data.current_steering;


  ml_controller_->setCurrentPose(current_pose_);
  ml_controller_->setWaypoints(planning_utils::extractPoses(trajectory_));
  setResampledTrajectory();

  const auto ml_controller_result=ml_controller_->run();
  AckermannLateralCommand cmd_msg;
  cmd_msg.stamp=clock_->now();
  cmd_msg.steering_tire_angle=ml_controller_result.second.steering_tire_angle;
  cmd_msg.steering_tire_rotation_rate=0;//ml_controller_result.second.steering_tire_rotation_rate;
  LateralOutput output;
  output.control_cmd = cmd_msg;
  output.sync_data.is_steer_converged = calcIsSteerConverged(cmd_msg);

  return output;
}

bool MlLateralController::calcIsSteerConverged(const AckermannLateralCommand & cmd)
{
  return std::abs(cmd.steering_tire_angle - current_steering_.steering_tire_angle) <
         static_cast<float>(param_.converged_steer_rad_);
}



}  // namespace ml_controller