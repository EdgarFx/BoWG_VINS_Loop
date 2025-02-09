# BoWG_VINS_Loop
## VINS-Fusion with BoWG Loop Closure Detection
This repository presents an enhanced version of [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) that incorporates our proposed [Bag-of-Word-Groups (BoWG)](https://github.com/EdgarFx/BoWG) method for robust loop closure detection. The integration primarily focuses on improving the loop closure capabilities within the *loop_fusion* package, aiming to achieve reliable and efficient visual place recognition in SLAM applications.

<p align="center">
  <img src="/support_files/image/nc_demo.gif" alt="Method" width="90%"/>
</p>
<p align="center">
  <em>BoWG demo detector on New College dataset</em>
</p>

<p align="center">
  <img src="support_files/image/MH03_vins_bowg.gif" alt="Method" width="90%"/>
</p>
<p align="center">
  <em>VINS-Fusion with BoWG on EuRoC MH_03_medium sequence</em>
</p>

## 1. Prerequisites
### 1.1 Ubuntu and ROS
Ubuntu 20.04. ROS Noetic. [ROS Installation](http://wiki.ros.org/ROS/Installation).

### 1.2. **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html).

## 2. Build BoWG_VINS_Loop
Clone the repository and catkin build:
```
cd ~/catkin_ws/src
git clone https://github.com/EdgarFx/BoWG_VINS_Loop.git
cd ../
catkin build
source ~/catkin_ws/devel/setup.bash
```
(if you fail in this step, try to find another computer with clean system or reinstall Ubuntu and ROS)


## 3. EuRoC Example
Download [EuRoC MAV Dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) to YOUR_DATASET_FOLDER. Take MH_01 for example, you can run BoWG_VINS_Loop with three sensor types (monocular camera + IMU, stereo cameras + IMU and stereo cameras). 
Open four terminals, run vins odometry, visual loop closure, rviz and play the bag file respectively. 
Green path is VIO odometry; red path is odometry under visual loop closure.

### 3.1 Monocualr camera + IMU

```
    roslaunch vins vins_rviz.launch
    rosrun vins vins_node ~/catkin_ws/src/BoWG_VINS_Loop/config/euroc/euroc_mono_imu_config.yaml 
    rosrun loop_fusion loop_fusion_node ~/catkin_ws/src/BoWG_VINS_Loop/config/euroc/euroc_mono_imu_config.yaml ~/catkin_ws/src/BoWG_VINS_Loop/config/euroc/euroc_bowg_config.yaml
    rosbag play YOUR_DATASET_FOLDER/MH_01_easy.bag
```

### 3.2 Stereo cameras + IMU

```
    roslaunch vins vins_rviz.launch
    rosrun vins vins_node ~/catkin_ws/src/BoWG_VINS_Loop/config/euroc/euroc_stereo_imu_config.yaml 
    rosrun loop_fusion loop_fusion_node ~/catkin_ws/src/BoWG_VINS_Loop/config/euroc/euroc_stereo_imu_config.yaml ~/catkin_ws/src/BoWG_VINS_Loop/config/euroc/euroc_bowg_config.yaml
    rosbag play YOUR_DATASET_FOLDER/MH_01_easy.bag
```

### 3.3 Stereo cameras

```
    roslaunch vins vins_rviz.launch
    rosrun vins vins_node ~/catkin_ws/src/BoWG_VINS_Loop/config/euroc/euroc_stereo_config.yaml 
    rosrun loop_fusion loop_fusion_node ~/catkin_ws/src/BoWG_VINS_Loop/config/euroc/euroc_stereo_config.yaml ~/catkin_ws/src/BoWG_VINS_Loop/config/euroc/euroc_bowg_config.yaml
    rosbag play YOUR_DATASET_FOLDER/MH_01_easy.bag
```

For more information about the use of the SLAM system, please refer to [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion).


## 4. Acknowledgements
We use [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) as the SLAM framework, [ceres solver](http://ceres-solver.org/) for non-linear optimization and [BoWG](https://github.com/EdgarFx/BoWG) for loop detection, a generic [camera model](https://github.com/hengli/camodocal) and [GeographicLib](https://geographiclib.sourceforge.io/).

## 5. License
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.