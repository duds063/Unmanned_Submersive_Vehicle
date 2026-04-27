# Unmanned_Submersive_Vehicle
USV Digital Twin — Naval Autonomy via Hierarchical Reinforcement Learning

![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-orange)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains a high-fidelity Digital Twin framework for an Autonomous Surface Vehicle (USV). The project integrates advanced hydrodynamic modeling, statistical sensor fusion, and Artificial Intelligence to solve autonomous navigation challenges in noisy maritime environments.

Technical Highlights
The core value of this project is its simulation fidelity, designed to enable Sim-to-Real policy transfers with minimal performance degradation.

Physics Engine (Fossen 6-DOF): Implementation of the unified equations of motion based on Thor I. Fossen (2011), accounting for Added Mass, Coriolis Forces, Non-linear Damping, and Restoring Forces.

EKF-based Perception: An Extended Kalman Filter for fusing noisy sensor data (IMU, Barometer, and Sonar), ensuring robust state estimation despite sensor interference.

Hierarchical Reinforcement Learning (HRL): A three-level architecture based on PPO (Proximal Policy Optimization):

L1 (Stabilization): Attitude and depth control.

L2 (Obstacle Avoidance): Sonar-based reactive maneuvers.

L3 (Navigation): Global path planning and waypoint tracking.

Numerical Integration: High-precision solvers using the Runge-Kutta 4th Order (RK4) method for dynamic stability.

Project Structure
physics_engine.py: Core simulator. Handles numerical integration and hydrodynamic matrices.

control_engine.py: Implements LQR controllers and custom gain scheduling logic.

rl_controller.py: Custom PPO agent implementation and hierarchical policy architecture.

sensor_engine.py: Simulates hardware noise for sensors like Open Echo Sonar and MS5837.

geometry_engine.py: Hull geometry calculations (Von Kármán ogive) and coefficient estimation.

visualization_server.py: Flask + SocketIO bridge for real-time Three.js rendering.

🛠️ Installation & Usage
Clone the repository:

Bash
```
git clone https://github.com/duds063/Unmanned_Submersive_Vehicle.git
Install dependencies:
```
Bash
```
pip install -r requirements.txt
Run validation tests:
```
Bash
```
python physics_engine.py
python control_engine.py
```
Scientific Methodology
Developed as an evolution of the Inertial Control Sandbox (ICS), this research focuses on the transition from simple inertial systems to complex maritime dynamics. By utilizing Domain Randomization during training, the AI controller becomes resilient to variations in water density and electromagnetic sensor noise.

Developed by: Eduardo Souza Costa and Marcelo Henrique Valdiero.
