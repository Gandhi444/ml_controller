# ML Controller

The ML Controller module calculates the steering angle for tracking a desired trajectory using the pure pursuit algorithm. This is used as a lateral controller plugin in the `trajectory_follower_node`.

## Inputs

Set the following from the [controller_node](../trajectory_follower_node/README.md)

- `autoware_auto_planning_msgs/Trajectory` : reference trajectory to follow.
- `nav_msgs/Odometry`: current ego pose and velocity information

## Outputs

Return LateralOutput which contains the following to the controller node

- `autoware_auto_control_msgs/AckermannLateralCommand`: target steering angle
- LateralSyncData
  - steer angle convergence
## Parameters
Used via `f1tenth_awsim_data_recorder.param.yaml` file in config directory.
| Name         | Type | Description  |
| ------------ | ---- | ------------ |
| converged_steer_rad | double  | Threshold for when steering is considered converged |
| resampling_ds | double | Trajectory resampling distance |
| lookahead_distance | double | Waypoint look ahead distance |
| closest_thr_dist | double | Maximum distance for selecting closest waypoint |
| closest_thr_ang | double | Maximum angle for selecting closest waypoint |
| trajectory_input_points | int | Number of trajectory waypoints passed to the network |
| model_path | string | Path to the onnx model inside resources folder|
| precision | string | Quantisation used by the model |
## Model requirments
Model should be stored as .onnx inside resources folder. As inputs it takes current vechicle x and y position and z and w parts of orientation quaterion followed by same values of n trajectory waypoints. Network output should be steering angle in range 0-1 it is converted to min-max vechicle steering angle in postprocessing.
