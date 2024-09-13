#!/usr/bin/env python

import os
import numpy as np
import math
from collections import deque

import rospy
from nlink_parser.msg import LinktrackNodeframe3  # Message type for UWB data
from uwbreloc.msg import X

from sklearn.preprocessing import MinMaxScaler  # For scaling input data
import pickle  # For saving/loading models and data

import torch  # PyTorch for deep learning
import torch.nn as nn  # Neural network modules
from torch.utils.data import Dataset, DataLoader  # Data utilities

from geometry_msgs.msg import Pose, PoseStamped  # Message types for position data
import tf.transformations as tf

import csv  # For reading/writing CSV files
import time  # For tracking the last publish time
import threading  # For managing thread-safe access to the buffer


# Initialize global variables for publishers, scalers, model, and buffer
pub = rospy.Publisher('uwb_pose', PoseStamped, queue_size=10)
Xscaler = MinMaxScaler()
Yscaler = MinMaxScaler()

# Buffer to store incoming UWB data
x_buffer = []
buffer_mutex = threading.Lock()
buffer_condition = threading.Condition(buffer_mutex)

# Placeholder for the machine learning model
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batchsize = 32
seqlen = 20
xdim = 20
ydim = 6
coding_dim = 128
hiddendim = 256
lr = 0.001
weight_decay = 0.9
decay_rate = 0.5
decay_step = 20

model_path = '/home/dat/slict_ws/src/uwbreloc/model/model_weights.pth'
all_path   = "/home/dat/slict_ws/src/uwbreloc/data/MCDUWB_nodelay/all"
train_path = "/home/dat/slict_ws/src/uwbreloc/data/MCDUWB_nodelay/train"

# Queue to store positions for calculating heading
queue_xy = deque(maxlen=50)

timeout = 20  # Timeout period in seconds

# Function to initialize the model and scalers
def initialize_model():
    global model
    rospy.loginfo("Initializing model and scalers...")
    
    # Load and fit scalers with training data
    csv_files = os.listdir(train_path)
    for file in csv_files:
        if file.endswith('.csv'):
            rospy.loginfo(f"Loading CSV file: {file}")
            data = np.loadtxt(all_path + '/' + file, delimiter=',')
            X_ = data[:, 7::3]
            Y_ = data[:, [1, 2, 3, 4, 5, 6]]
            Xscaler.partial_fit(X_)
            Yscaler.partial_fit(Y_)
            print(f"Successfully loaded and processed CSV file: {file}")

    # Initialize the custom RNN model
    model = MyRNN(input_dim=xdim, hidden_dim=hiddendim, output_dim=ydim, coding_dim=coding_dim).to(device)
    print(f'Loading model from {model_path}')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)

    rospy.loginfo("Model initialized successfully")


# Callback function for processing incoming UWB messages
def uwb_callback(msg):
    with buffer_condition:
        x_buffer.append(msg)
        buffer_condition.notify()

# Function to process the buffer and handle data in batches
def process_buffer():
    global model, timeout
    rospy.loginfo("Processing buffer...")
    
    last_received_time = time.time()  # Store the initial time

    while not rospy.is_shutdown():  # Loop until ROS is shutdown
        with buffer_condition:
            if len(x_buffer) == 0:  # Wait if the buffer is empty
                buffer_condition.wait(timeout=timeout)

            if len(x_buffer) == 0:  # Check if buffer is still empty after the timeout
                current_time = time.time()
                if current_time - last_received_time > timeout:
                    rospy.logwarn("Buffer has been empty. Shutting down the node.")
                    rospy.signal_shutdown("No data received.")
                    break
            else:
                last_received_time = time.time()  # Reset the last received time when new data comes in

            msg = x_buffer.pop(0)  # Get the first element from the buffer
            distances_1d = msg.distances  # Extract distances from UWB message
            distances_2d = np.reshape(distances_1d, (20, 20))  # Reshape to 2D array

            # Normalize the input data using the scaler
            X = Xscaler.transform(distances_2d)
            X = torch.from_numpy(np.array(X)).unsqueeze(0)  # Convert to PyTorch tensor

            # Make predictions using the trained model
            Ypred = model(X.float().to(device)).cpu().detach()
            Ypred = torch.cat((Ypred[0, :, :], Ypred[:, -1, :]), dim=0)
            Ypred = Yscaler.inverse_transform(Ypred.cpu())  # Inverse transform to get original scale

            stamp = msg.header.stamp  # Get the timestamp from the UWB message
            
            # Calculate average positions from predictions
            pos_x = np.mean(Ypred[0, [0, 3]])
            pos_y = np.mean(Ypred[0, [1, 4]])
            pos_z = np.mean(Ypred[0, [2, 5]])

            queue_xy.append((pos_x, pos_y))  # Append current positions to the queue
            if len(queue_xy) > 0:
                data_to_compute = list(queue_xy)  # Convert queue to list
                X = [pair[0] for pair in data_to_compute]  # Extract x-coordinates
                Y = [pair[1] for pair in data_to_compute]  # Extract y-coordinates
                
                if len(X) > 1:
                    # Calculate differences in positions
                    dX = np.diff(X)
                    dY = np.diff(Y)
                    
                    # Calculate heading angle
                    heading = np.arctan2(dY, dX)

                    # Calculate unit vectors for heading direction
                    unit_vectors = np.array([np.cos(heading), np.sin(heading)])
                      
                    # Compute the mean heading vector
                    mean_vector = np.mean(unit_vectors, axis=1)
                    
                    # Calculate the final heading angle
                    mean_heading = np.arctan2(mean_vector[1], mean_vector[0])
                    
                    # Convert the heading angle to degrees and radians
                    theta_radian = mean_heading
                    theta_degree = math.degrees(theta_radian)
                   
                    # Publish the computed pose
                    publish_pose(pos_x, pos_y, pos_z, theta_radian, stamp)


def publish_pose(pos_x, pos_y, pos_z, theta_radian, stamp):
    """Function to publish the PoseStamped message."""
    global pub
    pose_msg = PoseStamped()
    pose_msg.header.stamp = stamp
    pose_msg.header.frame_id = "map"
    pose_msg.pose.position.x = pos_x
    pose_msg.pose.position.y = pos_y
    pose_msg.pose.position.z = pos_z
    
    quaternion = tf.quaternion_from_euler(0.0, 0.0, theta_radian)


    pose_msg.pose.orientation.x = quaternion[0]
    pose_msg.pose.orientation.y = quaternion[1]
    pose_msg.pose.orientation.z = quaternion[2]
    pose_msg.pose.orientation.w = quaternion[3]
    pub.publish(pose_msg)



# Function to start the ROS node and listener
def start_uwb_listener():
    rospy.init_node('uwb_subscriber_node', anonymous=True)
    rospy.Subscriber('X_topic', X, uwb_callback)

    buffer_thread = threading.Thread(target=process_buffer)
    buffer_thread.start()

    rospy.spin()


# ECA layer definition remains unchanged
class eca_layer(nn.Module):
    """Constructs an ECA (Efficient Channel Attention) module."""

    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# Define the RFB (Receptive Field Block) model with LSTM and attention layers
class rfbModel(nn.Module):
    """Receptive Field Block (RFB) Model with LSTM and attention layers."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.eca = eca_layer(1)
        self.relu = torch.nn.ReLU(inplace=False)
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, bidirectional=False, num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim, bias=True)
        self.bn = torch.nn.BatchNorm1d(output_dim, affine=False)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.relu(x)
        x_p = x
        x = x.transpose(1, 2)
        x = self.eca(x)
        x = x.transpose(1, 2)
        x = x + x_p
        
        bs, seq, hs = x.size()
        x = x.reshape(bs * seq, hs)
        x = self.fc(x)
        x = self.bn(x)
        x = x.view(bs, seq, self.output_dim)

        return x

# Define the custom RNN model with position coding and RFB layers
class MyRNN(nn.Module):
    """Custom RNN model with position coding and RFB layers."""

    def __init__(self, input_dim, hidden_dim, output_dim, coding_dim, depth=3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth

        self.rfb = torch.nn.Sequential()

        for level in range(depth):
            chs = int(hidden_dim / 2 ** (level))
            nhs = int(hidden_dim / 2 ** (level + 1))
            if level == 0:
                self.rfb.append(rfbModel(coding_dim, chs, nhs))
            else:
                self.rfb.append(rfbModel(chs, chs, nhs))

        self.relu = torch.nn.ReLU(inplace=False)
        lhs = int(self.hidden_dim / 2 ** (depth))
        self.fcend = torch.nn.Linear(lhs, output_dim, bias=True)
        self.ps_coding = torch.nn.Linear(input_dim, coding_dim, bias=True)
        self.coding = torch.nn.Linear(input_dim, coding_dim, bias=True)

    def forward(self, x):
        bs, seq_len, input_dim = x.size()
        position_coding = torch.arange(1, 21).view(1, 1, 20).repeat(bs, seq_len, 1).to(device).float()
        position_coding = self.relu(self.ps_coding(position_coding))
        x = self.relu(self.coding(x))
        x = x + position_coding
        
        x = self.rfb(x)
        
        bs, seq, hs = x.size()
        x = x.reshape(bs * seq, hs)
        x = self.relu(x)
        x = self.fcend(x)
        x = x.view(bs, seq, self.output_dim)

        return x


# Main function
if __name__ == '__main__':
    try:
        initialize_model()
        start_uwb_listener()
    except rospy.ROSInterruptException:
        pass
