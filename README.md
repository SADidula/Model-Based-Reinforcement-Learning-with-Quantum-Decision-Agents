# Model-Based-Reinforcement-Learning-with-Quantum-Decision-Agents.

This document describes each file in the ursina_sim project, its responsibilities, key APIs/topics/services, and how components interact. The project controls a robot using reinforcement learning (RL) and quantum-inspired paths to trace polygonal paths, supporting both simulation and hardware integration through ROS2.

## Top-Level Architecture

- ROS2 Nodes:
  - sim_node.py: Main simulation environment node and orchestrator for episodes, path generation, stepping, and metrics.
  - moveit_interface.py: Provides a MoveIt 2 interface for planning/executing trajectories and collision-aware motion for the arm.
- Core Logic Modules:
  - polygon_path.py: Generates polygonal reference paths and utilities for sampling, smoothing, and error computation.
  - metrics_tracker.py: Tracks per-step and per-episode metrics (path error, completion, timing, reward).
  - robotic_config.py: Loads and validates robot and environment configuration.
- Config and Metadata:
  - config/robotic_config.json: Robot kinematics, limits, topics, frame names, task parameters, and training/sim settings.
  - package.xml: Package metadata, build/run dependencies (rclpy, sensor_msgs, geometry_msgs, moveit_msgs, etc.).
- Learning/Control:
  - RL/agent modules: Policy selection and action computation for the robot during training/evaluation.
  - Quantum path modules: Alternative path generation and/or control heuristics based on quantum-inspired techniques.

## File Details

### ursina_sim/sim_node.py
- Role:
  - ROS2 node that simulates episodes for tracing polygonal paths with different control backends (classical RL, quantum-inspired).
  - Manages episode lifecycle: reset, step, reward calculation, done conditions, and logging.
  - Publishes commands directly to MoveIt via moveit_interface, depending on configuration.
  - Subscribes to joint states/pose feedback and computes tracking error versus reference polygonal path.
- Key Responsibilities:
  - Load configuration (robotic_config.json) and initialize environment parameters (workspace, frames, rate, noise).
  - Create or select reference path (polygon_path.py) and precompute samples/segments.
  - Bridge control loop: read observations, query policy/agent, send actions, advance simulation, and compute rewards.
  - Track and publish metrics (e.g., error magnitude) and episode summaries to logging and optional topics.
  - Orchestrate saving of trajectories and evaluation runs.
- Important APIs/Topics:
  - Publishers: trajectory or pose command topics (e.g., /arm_controller/command, /target_pose).
  - Subscribers: /joint_states, /tf, task-specific feedback.
  - Services/Actions: May call MoveIt motion planning and FollowJointTrajectory action.
- Interactions:
  - Uses polygon_path.py for path reference and error computation.
  - Uses metrics_tracker.py for episode/step aggregation.
  - Sends motion goals directly via moveit_interface.py.

### ursina_sim/moveit_interface.py
- Role:
  - Thin wrapper around MoveIt 2 planning scene and MoveGroup API for the robot arm.
- Responsibilities:
  - Initialize MoveIt components (PlanningSceneInterface, MoveGroupCommander-equivalent) and configure planning pipeline.
  - Provide methods: set_pose_target, plan_to_pose, plan_to_joints, execute_trajectory, stop, clear targets.
  - Handle frame transforms and end-effector link configuration.
- APIs:
  - moveit_msgs for planning/execution, geometry_msgs Pose, sensor_msgs JointState.
  - May expose convenience functions for Cartesian path planning and IK queries.
- Interactions:
  - Called by sim_node.py for executing moves within an episode.

### ursina_sim/polygon_path.py
- Role:
  - Generate polygonal reference paths (e.g., triangle, square, custom vertices) and sampling utilities.
- Responsibilities:
  - Parameterize path density, speed profile, smoothing, and coordinate frames.
  - Provide APIs for:
    - Generating vertex lists and interpolated points.
    - Fetching the next reference waypoint for a given step/time index.
    - Computing geometric error: distance to path, tangent/normal components, and progress.
- Interactions:
  - sim_node.py uses this for per-step target reference and to compute tracking error and completion.

### ursina_sim/metrics_tracker.py
- Role:
  - Record and aggregate step-wise and episode-wise metrics.
- Responsibilities:
  - Accumulate error magnitude, reward terms, action norms, time-to-complete.
  - Output summaries and optional CSV/JSON logs or ROS topics.
- Interactions:
  - sim_node.py updates tracker each step and on episode end.

### ursina_sim/robotic_config.py
- Role:
  - Typed loader/validator for configuration JSON and environment variables.
- Responsibilities:
  - Parse config/robotic_config.json and provide strongly-typed accessors.
  - Validate joint limits, topic names, frame IDs, and planner settings.
- Interactions:
  - sim_node.py retrieve configuration via this module.

### config/robotic_config.json
- Role:
  - Central configuration for robot, environment, and training.
- Likely Sections:
  - Robot: joint names, limits, controller/action names, end-effector link, base frame.
  - Planning: MoveIt group name, planner params, velocity/acceleration scaling.
  - Topics: command/feedback topics, QoS profiles.
  - Task: polygon type, number of sides, size, sampling rate, max steps/episodes.
  - Training: RL backend selection, reward weights, exploration parameters.
  - Simulation: noise, time step, seed, reset settings.
- Interactions:
  - Loaded by robotic_config.py and referenced throughout nodes.

### RL/Agent Modules (e.g., agents/rl_agent.py, agents/policy.py)
- Role:
  - Implement policy networks or controllers for selecting actions given observations.
- Responsibilities:
  - Define observation encoding (pose error, velocities, progress) and action decoding (joint increments, target pose).
  - Load/save checkpoints and support evaluation mode.
- Interactions:
  - sim_node.py invokes agent.compute_action(observation) each step.

### Quantum Path Modules (e.g., quantum/path_planner.py, quantum/policy.py)
- Role:
  - Generate or adapt reference paths using quantum-inspired heuristics or sampling.
- Responsibilities:
  - Provide alternative path candidates or control signals for the same polygonal tracing task.
- Interactions:
  - sim_node.py can switch between classical RL path/control and quantum path planning based on config.

## Component Interactions

- sim_node.py is the orchestration hub:
  - Loads configuration and initializes path generator (polygon_path.py) and metrics tracker.
  - For each step:
    - Reads robot state feedback.
    - Computes error to the current reference waypoint.
    - Queries the selected controller (RL or quantum) for the next action.
    - Sends the command via moveit_interface.py.
    - Logs step metrics: [metrics] step={k} err={e}.
  - Ends episode on completion criteria and records summaries.

- moveit_interface.py provides planning and execution when higher-level planning is required.

- Config drives behavior everywhere, keeping topics, frames, and robot specifics consistent.

## Key Runtime Notes

- Launch/Run:
  - Use ROS2 launch or direct node execution to start sim_node.py.
  - Ensure config/robotic_config.json is discoverable and matches the robot description and topics.
- Metrics:
  - Metrics are printed every simulation step and can be redirected to logs for analysis.
- Extensibility:
  - New polygon types or path generators can be added to polygon_path.py.
  - New controllers can be registered and selected via configuration without changing sim_node.py.

## Architecture at a Glance

- ROS 2 Python node coordinates simulation, targets, and arm control.
- MoveIt 2 provides inverse kinematics and trajectory generation.
- The simulator publishes visualization markers and target poses.
- Optional learning components (PyTorch) can drive policies.
- Configuration is JSON-backed and accessed via a typed Python helper.
- Packaging uses ament_python for ROS 2 and setuptools for Python.

## Technology Stack

- Language: Python 3.x
- Robotics middleware: ROS 2 (rclpy)
- Motion planning: MoveIt 2 (moveit_msgs)
- Messaging:
  - geometry_msgs (PoseStamped, Pose, Point, Quaternion)
  - sensor_msgs (JointState)
  - trajectory_msgs (JointTrajectory, JointTrajectoryPoint)
  - std_msgs (e.g., Header, ColorRGBA)
  - visualization_msgs (Marker/MarkerArray)
- Math/ML:
  - NumPy
  - PyTorch (for optional RL/learning modules)
- Packaging/build:
  - ament_python (ROS 2 integration)
  - setuptools
- Config: JSON files (e.g., robotic_config.json) + typed loader
- Visualization: RViz via visualization_msgs and published Pose(s)/Marker(s)

## Key Components

- sim node (PathFollowerSimulator)
  - Publishes visualization markers for the path and targets.
  - Maintains simulated bot state and path-following logic.
  - Emits target poses for the robot arm relative to base_link.
  - Integrates with MoveIt via a convenience interface.
  - Supports episodic runs and metrics logging.

- MoveIt 2 interface
  - Requests IK using moveit_msgs.srv/GetPositionIK.
  - Generates and publishes JointTrajectory messages.
  - Can plan or send single-point trajectories based on IK results.

- Configuration
  - Centralized JSON config for robot geometry, limits, frames, and environment settings.
  - Python helper exposes typed accessors and validation.

- Visualization
  - RViz markers for current path, waypoints, and active targets.
  - Published PoseStamped for external consumers and debugging.

- Optional learning hooks
  - PyTorch-based policy or reward computation can be integrated with the simulator loop.
  - Numpy utilities for sampling, interpolation, and metrics.

## CLI and Runtime Flags

- Common node flags include:
  - General: verbosity, seeding, episode length, step period.
  - MoveIt control cadence: moveit_every (seconds; 0 disables), to avoid flooding IK.
  - Paths/IO: config file path, logs/metrics output.
- Topics/services:
  - Publishes:
    - /arm_target_pose (geometry_msgs/PoseStamped)
    - /arm_command (trajectory_msgs/JointTrajectory) or similar topic for trajectories
    - Visualization markers topic(s) for RViz
  - Consumes (typical):
    - Joint states and TF frames (depending on deployment)
  - Services:
    - MoveIt IK and (optionally) motion planning services

## Data Flow

1. Simulator advances the bot along a path and computes a desired target pose near the bot marker.
2. Target pose is published for visualization and external tools.
3. If enabled, a MoveIt command loop periodically:
   - Requests IK for the target pose.
   - Publishes a JointTrajectory command when IK succeeds.
4. Metrics and state are collected per step and per episode.

## File/Module Highlights

- ursina_sim/sim_node.py
  - Node creation and timers, publishers, and MoveIt hooks.
  - moveit_command that computes IK and publishes a trajectory.
  - Optional periodic timer to drive MoveIt at a controlled cadence.

- ursina_sim/robotic_config.py
  - Loads JSON config, validates keys, exposes fields like base_link, planning group, joint names, limits, and workspace bounds.

- config/robotic_config.json
  - Defines frames, planning group, joints, and workspace parameters.

- Packaging
  - setup.py: setuptools configuration for Python package.
  - package.xml: ROS 2 package metadata (ament_python).

## Current Status and Known Issues

- package.xml needs cleanup:
  - Ensure valid XML header and version tags.
  - Add explicit dependencies for moveit_msgs, geometry_msgs, sensor_msgs, trajectory_msgs, visualization_msgs, rclpy.
  - Remove malformed/duplicated test_depend entries.
  - Fill required metadata (name, version, maintainers, licenses).

- setup.py
  - Ensure non-empty version, description, and classifiers.
  - Verify install_requires lists ROS Python deps and third-party libs (numpy, torch) appropriately.

- Topic names and frames
  - Confirm that the target pose frame_id matches the MoveIt planning frame (e.g., base_link).
  - Ensure joint names in trajectories align with the planning group.

- Robustness
  - Add error handling around IK failures and log throttling.
  - Optionally debounce or rate-limit MoveIt calls via moveit_every flag.

- Testing
  - Add unit tests for config loader, IK request construction, and path utilities.
  - Add simulation smoke tests that validate topic publication.

## Getting Started

- Build
  - rosdep install --from-paths . --ignore-src -y
  - colcon build --packages-select ursina_sim

- Run
  - Source the workspace: source install/setup.bash
  - Launch the simulator node with MoveIt control enabled:
    - _Train_
      - ros2 run ursina_sim sim_node --max_radius=0.12 --offset_x=0.4 --verbose --agent rl --mode train --path circle --save_every 1000 --    stop_after_steps 10000 (for single selected path)
      - ros2 run ursina_sim sim_node --max_radius=2.5 --verbose --agent rl --mode train --save_every 1000 --multi_paths circle,triangle,square,pentagon,star,figure8 --steps_per_path 10000 (for multiple paths)
    - _Evaluation_
      - ros2 run ursina_sim sim_node --max_radius=0.12 --offset_x=0.4 --verbose --agent rl --mode eval --path circle

- Visualization (_Allways start your ROS/Gazebo environment in Moveit_)
  - No need to start ROS2/Gazebo Environments:
    - ros2 launch ur_simulation_gazebo ur_sim_moveit.launch.py ur_type:=ur5 (since the project uses the default Universal Robot Description and Universal Robot Simulation)

## Roadmap

- Stabilize package.xml and setup.py for CI/CD.
- Add ROS 2 launch files for repeatable bringup and RViz configuration.
- Expand metrics module and export to CSV/JSON.
- Provide example policies for learning integration and a replay dataset.
- Introduce unit/integration test suites and GitHub Actions workflow.
