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
#include <ament_index_cpp/get_package_share_directory.hpp>
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
    const std::string & model_path, const std::string & precision,const int trajectory_input_points,const float max_steer_angle,
    const float lookahead_distance,const float closest_thr_dist,const float closest_thr_ang,
    const tensorrt_common::BuildConfig build_config,
    const double norm_factor, [[maybe_unused]] const std::string & cache_dir,
    const tensorrt_common::BatchConfig & batch_config,
    const size_t max_workspace_size)
{
  lookahead_distance_=lookahead_distance;
  closest_thr_dist_=closest_thr_dist; 
  closest_thr_ang_=closest_thr_ang;
  norm_factor_ = norm_factor;
  std::string package_share_directory = ament_index_cpp::get_package_share_directory("ml_controller");
  std::string model_path_=package_share_directory+"/resources/"+model_path;
  trt_common_ = std::make_unique<tensorrt_common::TrtCommon>(
      model_path_, precision, nullptr, batch_config, max_workspace_size, build_config);
  trt_common_->setup();
  inputPoints_=trajectory_input_points;
  input_Length_=(inputPoints_+1)*4;
  max_steer_angle_=max_steer_angle;
  output_Length_=1;
  input_d_ = cuda_utils::make_unique<float[]>(input_Length_);
  output_d_ = cuda_utils::make_unique<float[]>(output_Length_);
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

std::pair<bool, Object> MlController::run()
{
  Object Results;
  if (!isDataReady()) {
    return std::make_pair(false,Results);
  }

  auto closest_pair = planning_utils::findClosestIdxWithDistAngThr(
    *curr_wps_ptr_, *curr_pose_ptr_, closest_thr_dist_, closest_thr_ang_);

  if (!closest_pair.first) {
    RCLCPP_WARN(
      logger, "cannot find, curr_bool: %d, closest_idx: %d", closest_pair.first,
      closest_pair.second);
    return std::make_pair(false,Results);
  }

  auto cur_pos=curr_pose_ptr_->position;
  auto cur_orient=curr_pose_ptr_->orientation;
  std::vector<float> data={(float)cur_pos.x,(float)cur_pos.y,(float)cur_orient.z,(float)cur_orient.w};
  int32_t starting_idx=findNextPointIdx(closest_pair.second);
  for(int i=0;i<inputPoints_;i++)
  {
    int32_t wp_idx=(starting_idx+i)%curr_wps_ptr_->size();
    auto next_wp=curr_wps_ptr_->at(wp_idx);
    auto next_pos=next_wp.position;
    auto next_ortient=next_wp.orientation;

    data.emplace_back(next_pos.x);
    data.emplace_back(next_pos.y);

    data.emplace_back(next_ortient.z);
    data.emplace_back(next_ortient.w);
  }
  
  if(!doInference(data,Results))
  {
     RCLCPP_WARN(logger, "Inference failed");
  };
  return std::make_pair(true,Results);
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
  float sum=0;
  for(auto point:input_h_)
  {
    sum+=point;
  }
  CHECK_CUDA_ERROR(cudaMemcpy(
    input_d_.get(), input_h_.data(), input_h_.size() * sizeof(float), cudaMemcpyHostToDevice));
  // No Need for Sync
}


bool MlController::doInference(const std::vector<float> & data, Object & results)
{
  if (!trt_common_->isInitialized()) {
     RCLCPP_WARN(logger, "trt_common no initialized");
    return false;
  }
  preprocess(data);
  if(!feedforward(results)){
    return false;
  }

  return true;
}

bool MlController::feedforward(Object & results)
{
  std::vector<void *> buffers = {
    input_d_.get(), output_d_.get()};

  trt_common_->enqueueV2(buffers.data(), *stream_, nullptr);

  auto out_results = std::make_unique<float[]>(output_Length_);
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    out_results.get(), output_d_.get(), sizeof(float)*output_Length_,
    cudaMemcpyDeviceToHost, *stream_));
  cudaStreamSynchronize(*stream_);
  results.steering_tire_angle=out_results[0]* 2 * max_steer_angle_ - max_steer_angle_;
  //results.steering_tire_rotation_rate=out_results[1];
  // results.acceleration=out_results[2];
  // results.speed=out_results[3];
  // results.jerk=out_results[4];
  
  return true;
}

}  // namespace ml_controller
