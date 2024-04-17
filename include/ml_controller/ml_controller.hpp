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

#ifndef ml_controller__ml_controller_HPP_
#define ml_controller__ml_controller_HPP_

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose.hpp>

#include <memory>
#include <utility>
#include <vector>

#define EIGEN_MPL2_ONLY
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cuda_utils/cuda_unique_ptr.hpp>
#include <cuda_utils/stream_unique_ptr.hpp>
#include <tensorrt_common/tensorrt_common.hpp>
#include <tensorrt_yolox/preprocess.hpp>


namespace ml_controller
{
struct Object
{
  float steering_tire_angle;
  float steering_tire_rotation_rate;
  float acceleration;
  float speed;
  float jerk;
};
using ObjectArray = std::vector<Object>;
using ObjectArrays = std::vector<ObjectArray>;

class MlController
{
public:
  MlController(
    const std::string & model_path, const std::string & precision, const int num_class = 8,
    const float score_threshold = 0.3, const float nms_threshold = 0.7,
    const tensorrt_common::BuildConfig build_config = tensorrt_common::BuildConfig(),
    const bool use_gpu_preprocess = false, std::string calibration_image_list_file = std::string(),
    const double norm_factor = 1.0, [[maybe_unused]] const std::string & cache_dir = "",
    const tensorrt_common::BatchConfig & batch_config = {1, 1, 1},
    const size_t max_workspace_size = (1 << 30),int inputPoints=2);
  ~MlController() = default;

  rclcpp::Logger logger = rclcpp::get_logger("ml_controller");
  // setter
  void setCurrentPose(const geometry_msgs::msg::Pose & msg);
  void setWaypoints(const std::vector<geometry_msgs::msg::Pose> & msg);
  void setLookaheadDistance(double ld) { lookahead_distance_ = ld; }
  void setClosestThreshold(double closest_thr_dist, double closest_thr_ang)
  {
    closest_thr_dist_ = closest_thr_dist;
    closest_thr_ang_ = closest_thr_ang;
  }

  // getter
  geometry_msgs::msg::Point getLocationOfNextWaypoint() const { return loc_next_wp_; }
  geometry_msgs::msg::Point getLocationOfNextTarget() const { return loc_next_tgt_; }

  bool isDataReady();
  std::pair<bool, double> run();  // calculate curvature
  void preprocess(const std::vector<float> & data);
  bool doInference(const std::vector<float> & data, Object & results);
  bool feedforward(const std::vector<float> & data, Object & results);
private:
  //tesnor rt
  std::unique_ptr<tensorrt_common::TrtCommon> trt_common_;
  std::vector<float> input_h_;
  int input_Length_;
  int output_Length_;
  int batch_size_;
  double norm_factor_;
  int inputPoints_;
  CudaUniquePtr<float[]> input_d_;
  CudaUniquePtr<float[]> output_d_;
  StreamUniquePtr stream_{makeCudaStream()};
  // variables for debug

  geometry_msgs::msg::Point loc_next_wp_;
  geometry_msgs::msg::Point loc_next_tgt_;

  // variables got from outside
  double lookahead_distance_, closest_thr_dist_, closest_thr_ang_;
  std::shared_ptr<std::vector<geometry_msgs::msg::Pose>> curr_wps_ptr_;
  std::shared_ptr<geometry_msgs::msg::Pose> curr_pose_ptr_;

  // functions
  int32_t findNextPointIdx(int32_t search_start_idx);
  std::pair<bool, geometry_msgs::msg::Point> lerpNextTarget(int32_t next_wp_idx);
};

}  // namespace ml_controller

#endif  // ml_controller__ml_controller_HPP_
