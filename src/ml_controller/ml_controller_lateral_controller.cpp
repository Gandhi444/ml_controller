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

namespace
{
enum TYPE {
  VEL_LD = 0,
  CURVATURE_LD = 1,
  LATERAL_ERROR_LD = 2,
  TOTAL_LD = 3,
  CURVATURE = 4,
  LATERAL_ERROR = 5,
  VELOCITY = 6,
  SIZE  // this is the number of enum elements
};
}  // namespace

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
  // param_.ld_velocity_ratio = node.declare_parameter<double>("ld_velocity_ratio");
  // param_.ld_lateral_error_ratio = node.declare_parameter<double>("ld_lateral_error_ratio");
  // param_.ld_curvature_ratio = node.declare_parameter<double>("ld_curvature_ratio");
  // param_.long_ld_lateral_error_threshold =
  //   node.declare_parameter<double>("long_ld_lateral_error_threshold");
  // param_.min_lookahead_distance = node.declare_parameter<double>("min_lookahead_distance");
  // param_.max_lookahead_distance = node.declare_parameter<double>("max_lookahead_distance");
  // param_.reverse_min_lookahead_distance =
  //   node.declare_parameter<double>("reverse_min_lookahead_distance");
  param_.converged_steer_rad_ = node.declare_parameter<double>("converged_steer_rad");
  // param_.prediction_ds = node.declare_parameter<double>("prediction_ds");
  // param_.prediction_distance_length = node.declare_parameter<double>("prediction_distance_length");
  // param_.resampling_ds = node.declare_parameter<double>("resampling_ds");
  // param_.curvature_calculation_distance =
  //   node.declare_parameter<double>("curvature_calculation_distance");
  // param_.enable_path_smoothing = node.declare_parameter<bool>("enable_path_smoothing");
  // param_.path_filter_moving_ave_num = node.declare_parameter<int64_t>("path_filter_moving_ave_num");
  param_.model_path = node.declare_parameter<std::string>("model_path", "test.onnx");
  param_.precision = node.declare_parameter<std::string>("precision", "fp32");
  ml_controller_ = std::make_unique<MlController>(param_.model_path, param_.precision);
  // Debug Publishers
  pub_debug_marker_ =
    node.create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/markers", 0);
  pub_debug_values_ = node.create_publisher<tier4_debug_msgs::msg::Float32MultiArrayStamped>(
    "~/debug/ld_outputs", rclcpp::QoS{1});

  // Publish predicted trajectory
  pub_predicted_trajectory_ = node.create_publisher<autoware_auto_planning_msgs::msg::Trajectory>(
    "~/output/predicted_trajectory", 1);
}


TrajectoryPoint MlLateralController::calcNextPose(
  const double ds, TrajectoryPoint & point, AckermannLateralCommand cmd) const
{
  geometry_msgs::msg::Transform transform;
  transform.translation = tier4_autoware_utils::createTranslation(ds, 0.0, 0.0);
  transform.rotation =
    planning_utils::getQuaternionFromYaw(((tan(cmd.steering_tire_angle) * ds) / param_.wheel_base));
  TrajectoryPoint output_p;

  tf2::Transform tf_pose;
  tf2::Transform tf_offset;
  tf2::fromMsg(transform, tf_offset);
  tf2::fromMsg(point.pose, tf_pose);
  tf2::toMsg(tf_pose * tf_offset, output_p.pose);
  return output_p;
}





bool MlLateralController::isReady([[maybe_unused]] const InputData & input_data)
{
  return true;
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
  cmd_msg.steering_tire_rotation_rate=ml_controller_result.second.steering_tire_rotation_rate;
  LateralOutput output;
  output.control_cmd = cmd_msg;
  output.sync_data.is_steer_converged = calcIsSteerConverged(cmd_msg);

  // calculate predicted trajectory with iterative calculation
  const auto predicted_trajectory = generatePredictedTrajectory();
  if (!predicted_trajectory) {
    RCLCPP_ERROR(logger_, "Failed to generate predicted trajectory.");
  } else {
    pub_predicted_trajectory_->publish(*predicted_trajectory);
  }

  return output;
}

bool MlLateralController::calcIsSteerConverged(const AckermannLateralCommand & cmd)
{
  return std::abs(cmd.steering_tire_angle - current_steering_.steering_tire_angle) <
         static_cast<float>(param_.converged_steer_rad_);
}


AckermannLateralCommand MlLateralController::generateCtrlCmdMsg(
  const double target_curvature)
{
  const double tmp_steering =
    planning_utils::convertCurvatureToSteeringAngle(param_.wheel_base, target_curvature);
  AckermannLateralCommand cmd;
  cmd.stamp = clock_->now();
  cmd.steering_tire_angle = static_cast<float>(
    std::min(std::max(tmp_steering, -param_.max_steering_angle), param_.max_steering_angle));

  // pub_ctrl_cmd_->publish(cmd);
  return cmd;
}

void MlLateralController::publishDebugMarker() const
{
  visualization_msgs::msg::MarkerArray marker_array;

  marker_array.markers.push_back(createNextTargetMarker(debug_data_.next_target));
  marker_array.markers.push_back(
    createTrajectoryCircleMarker(debug_data_.next_target, current_odometry_.pose.pose));
}

}  // namespace ml_controller