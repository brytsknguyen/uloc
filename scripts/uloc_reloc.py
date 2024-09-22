import os
import math
import numpy as np

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseStamped  # Message types for position data
import tf.transformations as tf
import threading

import evo.core.trajectory as et
import evo.core.sync as sync
import evo.core.transformations as tr
import evo.tools.file_interface as ef

from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from sklearn.linear_model import RANSACRegressor

import csv  # For reading/writing CSV files
import time  # Import time module for tracking the last publish time

mutex = threading.Lock()

uloc_poses = []
slam_poses = []

first_tslam = None
first_tuloc = None

minTrajTime = 20

reloc_pub = rospy.Publisher('/reloc_pose', PoseStamped, queue_size=10)

def slam_callback(msg):
    
    global first_tslam

    if first_tslam == None:
        first_tslam = msg.header.stamp
        
    ps = PoseStamped()
    ps.header = msg.header
    ps.pose = msg.pose.pose

    with mutex:
        slam_poses.append(ps)
    # print("slam trajectory", len(slam_poses), (msg.header.stamp - first_tslam).to_sec())


def uloc_callback(msg):
    
    global first_tuloc

    if first_tuloc == None:
        first_tuloc = msg.header.stamp

    with mutex:
        uloc_poses.append(msg)
    # print("uloc trajectory", len(uloc_poses), (msg.header.stamp - first_tuloc).to_sec())

def poses_to_traj(poses):

    # Create trajectories from traj log
    pos = []
    rot = []
    ts_  = []
    for pose in poses:
        pos.append([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
        rot.append([1, 0, 0, 0])
        ts_.append(pose.header.stamp.to_sec())
    traj = et.Trajectory(positions_xyz = pos, orientations_quat_wxyz = rot, timestamps = ts_)
    return traj

# def resample_trajectory(timestamps_src, poses_src, timestamps_target):
#     poses_src = np.array(poses_src)
#     interpolators = [interp1d(timestamps_src, poses_src[:, i], kind='linear', fill_value='extrapolate') for i in range(poses_src.shape[1])]
#     resampled_poses = np.column_stack([interp(timestamps_target) for interp in interpolators])
#     return resampled_poses

def detect_outliers(trajectory, max_residual=0.5):
    """
    Detect outliers in a trajectory using RANSAC.
    """
    timestamps, poses = trajectory
    
    # Assume poses are 3D points
    poses = np.array([pose[:3, 3] for pose in poses])  # Extract positions

    # Fit RANSAC model
    ransac = RANSACRegressor(residual_threshold=max_residual)
    ransac.fit(timestamps.reshape(-1, 1), poses)
    
    # Identify inliers and outliers
    inlier_mask = ransac.inlier_mask_
    outliers = np.where(~inlier_mask)[0]

    return outliers

def find_relative_transform():

    # Create trajectories from traj log
    uloc_traj = poses_to_traj(uloc_poses)
    slam_traj = poses_to_traj(slam_poses)

    print(slam_traj.timestamps[0], slam_traj.timestamps[-1], uloc_traj.timestamps[0], uloc_traj.timestamps[-1])
    
    # Sync the trajectories if necessary (based on timestamps)
    slam_traj, uloc_traj = sync.associate_trajectories(slam_traj, uloc_traj, max_diff=0.1)

    print(len(slam_traj.positions_xyz), len(uloc_traj.positions_xyz))

    # Fit RANSAC model
    # ransac = RANSACRegressor(residual_threshold=1.0)
    # ransac.fit(slam_traj.positions_xyz - uloc_traj.positions_xyz)

    # Align the trajectories
    R_pm_sp, t_pm_sp, _ = slam_traj.align(uloc_traj)

    r = Rotation.from_matrix(R_pm_sp)
    q = r.as_quat()
    e = r.as_euler('zyx', degrees=True)

    # Print the alignment transform
    print(R_pm_sp, t_pm_sp, e, q)

    relocPose = PoseStamped()

    relocPose.header.stamp = rospy.Time.now()
    relocPose.pose.position.x = t_pm_sp[0]
    relocPose.pose.position.y = t_pm_sp[1]
    relocPose.pose.position.z = t_pm_sp[2]
    relocPose.pose.orientation.x = q[0]
    relocPose.pose.orientation.y = q[1]
    relocPose.pose.orientation.z = q[2]
    relocPose.pose.orientation.w = q[3]

    reloc_pub.publish(relocPose)


def calculate_reloc():

    # Loop every 1s
    rate = rospy.Rate(1)
    while True:

        # Check the two buffers
        print("uloc and slam trajectory: ", len(uloc_poses), len(slam_poses))

        uloc_tspan = -1
        if len(uloc_poses) > 1:
            uloc_tspan = (uloc_poses[-1].header.stamp - uloc_poses[0].header.stamp).to_sec()
            print("uloc time span", uloc_tspan)

        slam_tspan = -1
        if len(slam_poses) > 1:
            slam_tspan = (slam_poses[-1].header.stamp - slam_poses[0].header.stamp).to_sec()
            print("slam time span", slam_tspan)

        if uloc_tspan > minTrajTime and slam_tspan > minTrajTime:
            find_relative_transform()

        rate.sleep()

# Main function
if __name__ == '__main__':

    # Initialize the node
    rospy.init_node('uloc_reloc', anonymous=False)
    
    # Subscribe to slam pose estimate
    rospy.Subscriber("/opt_odom", Odometry, slam_callback)
    
    # Subscribe to the uloc estimate
    rospy.Subscriber("/uwb_pose", PoseStamped, uloc_callback)

    # Create and start a background thread
    thread = threading.Thread(target=calculate_reloc)
    thread.daemon = True  # Set thread as a daemon so it will automatically exit with the main thread
    thread.start()

    # Use MultiThreadedSpinner to process callbacks in multiple threads
    rospy.spin()