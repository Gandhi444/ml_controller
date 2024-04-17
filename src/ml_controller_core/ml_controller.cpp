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

#include "ml_controller/ml_controller.hpp"

#include "ml_controller/util/planning_utils.hpp"

#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace ml_controller
{
using cuda_utils::CudaUniquePtr;
using cuda_utils::CudaUniquePtrHost;
using cuda_utils::makeCudaStream;
using cuda_utils::StreamUniquePtr;

MlController::MlController(
    const std::string & model_path, const std::string & precision, const int num_class = 8,
    const float score_threshold = 0.3, const float nms_threshold = 0.7,
    const tensorrt_common::BuildConfig build_config = tensorrt_common::BuildConfig(),
    const bool use_gpu_preprocess = false, std::string calibration_image_list_file = std::string(),
    const double norm_factor = 1.0, [[maybe_unused]] const std::string & cache_dir = "",
    const tensorrt_common::BatchConfig & batch_config = {1, 1, 1},
    const size_t max_workspace_size = (1 << 30),int inputLength,int outputLength)
{
  lookahead_distance_=0.0;
  closest_thr_dist_=3.0; 
  closest_thr_ang_=(M_PI / 4);
  norm_factor_ = norm_factor;
  trt_common_ = std::make_unique<tensorrt_common::TrtCommon>(
      model_path, precision, nullptr, batch_config, max_workspace_size, build_config);
  trt_common_->setup();
  const auto input_dims = trt_common_->getBindingDimensions(0);

  const auto out_scores_dims = trt_common_->getBindingDimensions(3);
  input_Length_=inputLength;
  output_Length_=outputLength;
  input_d_ = cuda_utils::make_unique<float[]>(input_Length_);
  output_d_ = cuda_utils::make_unique<float[]>(input_Length_);
}  
bool MlController::isDataReady()
{
  if (!curr_wps_ptr_) {
    return false;
  }
  if (!curr_pose_ptr_) {
    return false;
  }
  return true;
}

std::pair<bool, double> MlController::run()
{
  if (!isDataReady()) {
    return std::make_pair(false, std::numeric_limits<double>::quiet_NaN());
  }

  auto closest_pair = planning_utils::findClosestIdxWithDistAngThr(
    *curr_wps_ptr_, *curr_pose_ptr_, closest_thr_dist_, closest_thr_ang_);

  if (!closest_pair.first) {
    RCLCPP_WARN(
      logger, "cannot find, curr_bool: %d, closest_idx: %d", closest_pair.first,
      closest_pair.second);
    return std::make_pair(false, std::numeric_limits<double>::quiet_NaN());
  }

  int32_t next_wp_idx = findNextPointIdx(closest_pair.second);

  if (next_wp_idx == -1) {
    RCLCPP_WARN(logger, "lost next waypoint");
    return std::make_pair(false, std::numeric_limits<double>::quiet_NaN());
  }

  loc_next_wp_ = curr_wps_ptr_->at(next_wp_idx).position;

  geometry_msgs::msg::Point next_tgt_pos;
  // if next waypoint is first
  if (next_wp_idx == 0) {
    next_tgt_pos = curr_wps_ptr_->at(next_wp_idx).position;
  } else {
    // linear interpolation
    std::pair<bool, geometry_msgs::msg::Point> lerp_pair = lerpNextTarget(next_wp_idx);

    if (!lerp_pair.first) {
      RCLCPP_WARN(logger, "lost target! ");
      return std::make_pair(false, std::numeric_limits<double>::quiet_NaN());
    }

    next_tgt_pos = lerp_pair.second;
  }
  loc_next_tgt_ = next_tgt_pos;

  double kappa = planning_utils::calcCurvature(next_tgt_pos, *curr_pose_ptr_);

  return std::make_pair(true, kappa);
}

// linear interpolation of next target
std::pair<bool, geometry_msgs::msg::Point> MlController::lerpNextTarget(int32_t next_wp_idx)
{
  constexpr double ERROR2 = 1e-5;  // 0.00001
  const geometry_msgs::msg::Point & vec_end = curr_wps_ptr_->at(next_wp_idx).position;
  const geometry_msgs::msg::Point & vec_start = curr_wps_ptr_->at(next_wp_idx - 1).position;
  const geometry_msgs::msg::Pose & curr_pose = *curr_pose_ptr_;

  Eigen::Vector3d vec_a(
    (vec_end.x - vec_start.x), (vec_end.y - vec_start.y), (vec_end.z - vec_start.z));

  if (vec_a.norm() < ERROR2) {
    RCLCPP_ERROR(logger, "waypoint interval is almost 0");
    return std::make_pair(false, geometry_msgs::msg::Point());
  }

  const double lateral_error =
    planning_utils::calcLateralError2D(vec_start, vec_end, curr_pose.position);

  if (fabs(lateral_error) > lookahead_distance_) {
    RCLCPP_ERROR(logger, "lateral error is larger than lookahead distance");
    RCLCPP_ERROR(
      logger, "lateral error: %lf, lookahead distance: %lf", lateral_error, lookahead_distance_);
    return std::make_pair(false, geometry_msgs::msg::Point());
  }

  /* calculate the position of the foot of a perpendicular line */
  Eigen::Vector2d uva2d(vec_a.x(), vec_a.y());
  uva2d.normalize();
  Eigen::Rotation2Dd rot =
    (lateral_error > 0) ? Eigen::Rotation2Dd(-M_PI / 2.0) : Eigen::Rotation2Dd(M_PI / 2.0);
  Eigen::Vector2d uva2d_rot = rot * uva2d;

  geometry_msgs::msg::Point h;
  h.x = curr_pose.position.x + fabs(lateral_error) * uva2d_rot.x();
  h.y = curr_pose.position.y + fabs(lateral_error) * uva2d_rot.y();
  h.z = curr_pose.position.z;

  // if there is a intersection
  if (fabs(fabs(lateral_error) - lookahead_distance_) < ERROR2) {
    return std::make_pair(true, h);
  } else {
    // if there are two intersection
    // get intersection in front of vehicle
    const double s = sqrt(pow(lookahead_distance_, 2) - pow(lateral_error, 2));
    geometry_msgs::msg::Point res;
    res.x = h.x + s * uva2d.x();
    res.y = h.y + s * uva2d.y();
    res.z = curr_pose.position.z;
    return std::make_pair(true, res);
  }
}

int32_t MlController::findNextPointIdx(int32_t search_start_idx)
{
  // if waypoints are not given, do nothing.
  if (curr_wps_ptr_->empty() || search_start_idx == -1) {
    return -1;
  }

  // look for the next waypoint.
  for (int32_t i = search_start_idx; i < (int32_t)curr_wps_ptr_->size(); i++) {
    // if search waypoint is the last
    if (i == ((int32_t)curr_wps_ptr_->size() - 1)) {
      return i;
    }

    // if waypoint direction is forward
    const auto gld = planning_utils::getLaneDirection(*curr_wps_ptr_, 0.05);
    if (gld == 0) {
      // if waypoint is not in front of ego, skip
      auto ret = planning_utils::transformToRelativeCoordinate2D(
        curr_wps_ptr_->at(i).position, *curr_pose_ptr_);
      if (ret.x < 0) {
        continue;
      }
    } else if (gld == 1) {
      // waypoint direction is backward

      // if waypoint is in front of ego, skip
      auto ret = planning_utils::transformToRelativeCoordinate2D(
        curr_wps_ptr_->at(i).position, *curr_pose_ptr_);
      if (ret.x > 0) {
        continue;
      }
    } else {
      return -1;
    }

    const geometry_msgs::msg::Point & curr_motion_point = curr_wps_ptr_->at(i).position;
    const geometry_msgs::msg::Point & curr_pose_point = curr_pose_ptr_->position;
    // if there exists an effective waypoint
    const double ds = planning_utils::calcDistSquared2D(curr_motion_point, curr_pose_point);
    if (ds > std::pow(lookahead_distance_, 2)) {
      return i;
    }
  }

  // if this program reaches here , it means we lost the waypoint!
  return -1;
}

void MlController::setCurrentPose(const geometry_msgs::msg::Pose & msg)
{
  curr_pose_ptr_ = std::make_shared<geometry_msgs::msg::Pose>();
  *curr_pose_ptr_ = msg;
}

void MlController::setWaypoints(const std::vector<geometry_msgs::msg::Pose> & msg)
{
  curr_wps_ptr_ = std::make_shared<std::vector<geometry_msgs::msg::Pose>>();
  *curr_wps_ptr_ = msg;
}

void MlController::preprocess(const std::vector<float> & data)
{
  const auto batch_size = 1;
  auto input_dims = trt_common_->getBindingDimensions(0);
  input_dims.d[0] = batch_size;
  trt_common_->setBindingDimensions(0, input_dims);

  input_h_ = data;
  CHECK_CUDA_ERROR(cudaMemcpy(
    input_d_.get(), input_h_.data(), input_h_.size() * sizeof(float), cudaMemcpyHostToDevice));
  // No Need for Sync
}


bool MlController::doInference(const std::vector<float> & data, Object & results)
{
  if (!trt_common_->isInitialized()) {
    return false;
  }
  preprocess(data);
  return feedforward(data, results);
  
}

bool MlController::feedforward(const std::vector<float> & data, Object & results)
{
  std::vector<void *> buffers = {
    input_d_.get(), output_d_.get()};

  trt_common_->enqueueV2(buffers.data(), *stream_, nullptr);

  auto out_results = std::make_unique<float[]>(output_Length_);
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    out_results.get(), output_d_.get(), sizeof(float) * output_d_,
    cudaMemcpyDeviceToHost, *stream_));
  cudaStreamSynchronize(*stream_);
    // results.x_offset = std::clamp(0, static_cast<int32_t>(x1), images[i].cols);
    // results.y_offset = std::clamp(0, static_cast<int32_t>(y1), images[i].rows);
    // results.width = static_cast<int32_t>(std::max(0.0F, x2 - x1));
    // results.height = static_cast<int32_t>(std::max(0.0F, y2 - y1));
    // results.score = out_results[i * max_detections_ + j];
    // results.type = out_classes[i * max_detections_ + j];
  
  return true;
}

}  // namespace ml_controller