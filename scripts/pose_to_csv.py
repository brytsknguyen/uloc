#!/usr/bin/env python

import rospy
import csv
from geometry_msgs.msg import PoseStamped

class PoseToCSV:
    def __init__(self):
        # Create/Open the CSV file in write mode
        self.csv_file = open('/home/dat/slict_ws/src/uwbreloc/evo/output/ntu_day_10.csv', mode='w')
        self.csv_writer = csv.writer(self.csv_file)
        # Write CSV headers
        self.csv_writer.writerow(['Time', 'Position_X', 'Position_Y', 'Position_Z', 'Orientation_X', 'Orientation_Y', 'Orientation_Z', 'Orientation_W'])
        # Subscriber to the pose topic
        rospy.Subscriber("uwb_pose", PoseStamped, self.pose_callback)

    def pose_callback(self, msg):
        # Extract the data from PoseStamped message
        position = msg.pose.position
        orientation = msg.pose.orientation
        timestamp = msg.header.stamp.to_sec()
        # Write pose data to CSV
        self.csv_writer.writerow([timestamp, position.x, position.y, position.z, orientation.x, orientation.y, orientation.z, orientation.w])

    def cleanup(self):
        # Close the file on shutdown
        self.csv_file.close()

if __name__ == '__main__':
    rospy.init_node('pose_to_csv_node', anonymous=True)
    pose_to_csv = PoseToCSV()
    rospy.on_shutdown(pose_to_csv.cleanup)
    rospy.spin()
