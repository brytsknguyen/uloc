#!/usr/bin/env python

import os
import numpy as np
import math
import rospy
from nlink_parser.msg import LinktrackNodeframe3  # Message type for UWB data

from sklearn.preprocessing import MinMaxScaler  # For scaling input data
import pickle  # For saving/loading models and data

import torch  # PyTorch for deep learning
import torch.nn as nn  # Neural network modules
from torch.utils.data import Dataset, DataLoader  # Data utilities

from geometry_msgs.msg import Pose, PoseStamped  # Message types for position data
import tf.transformations as tf

import csv  # For reading/writing CSV files
import time  # Import time module for tracking the last publish time


# Publisher to send PoseStamped messages
pub = rospy.Publisher('pose_uwb', PoseStamped, queue_size=10)
last_stamp = 0
# Initialize scalers and distance arrays
Xscaler = MinMaxScaler()
Yscaler = MinMaxScaler()
distances = np.zeros((0, 20))
distances_list = []
seq_leng = 49

# Hyperparameters
batchsize   = 32
seqlen      = 20
xdim        = 20
ydim        = 6
coding_dim  = 128
hiddendim   = 256
lr          = 0.001
weigt_decay = 0.9
decay_rate  = 0.5
decay_step  = 20

num_epoch = 200

# Temporary storage for incoming messages
buffer_tag0 = None
buffer_tag1 = None

# Paths to data directories
model_path = '/home/dat/slict_ws/src/mcduwb/model/model_weights.pth'
all_path = "/home/dat/ai_ws/catkin_ws/src/mcduwb/data/MCDUWB_nodelay/all"
train_path = "/home/dat/ai_ws/catkin_ws/src/mcduwb/data/MCDUWB_nodelay/train"

# Device configuration for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize global variables to track the last published pose and time
last_pose = None
last_publish_time = None
publish_interval = 0.05  # 0.05 seconds

fst_time = True


# Define the ECA (Efficient Channel Attention) layer
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

# Initialize and load the model
def initialize_model():
    global model, Xscaler, Yscaler
    rospy.loginfo("Initializing model and scalers...")

    Xscaler = MinMaxScaler()
    Yscaler = MinMaxScaler()

    # Load and fit scalers with data
    csv_files = os.listdir(train_path) 
    for file in csv_files:
        if file.endswith('.csv'):
            rospy.loginfo(f"Loading CSV file: {file}")
            data = np.loadtxt(all_path + '/' + file, delimiter=',')
            X_ = data[:, 7::3]
            Y_ = data[:, [1,2,3,4,5,6]]
            Xscaler.partial_fit(X_)
            Yscaler.partial_fit(Y_)
            print(f"Successfully loaded and processed CSV file: {file}")
    
    model = MyRNN(input_dim=xdim, hidden_dim=hiddendim, output_dim=ydim, coding_dim=coding_dim)
    

    # Model path
    print(f'Loading model from {model_path}')
    
    # Uncomment and modify the following line to load the model if needed
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)
    
    rospy.loginfo("Model initialized successfully")

# Extract distances from incoming data messages
def extract_distances(data):
    """Extract distances from the received data and map them to their corresponding IDs."""
    new_distances = [0] * 10  # Initialize a list of 10 elements with default value 0

    id_to_distance = {node.id: node.dis for node in data.nodes}

    for i in range(10):
        if i in id_to_distance:
            new_distances[i] = id_to_distance[i]
    
    return new_distances

# Add new distance data to the global distances array and process it
def add_to_distances(tag0_distances, tag1_distances):
    """Add new distance data to the global distances array and process it."""
    new_distances_row = np.array(tag0_distances + tag1_distances).reshape(1, 20).astype(np.float64)

    global distances, distances_list, last_pose

    if not (np.all(new_distances_row[0, :10] == 0) or np.all(new_distances_row[0, 10:] == 0)):
        new_distances_row = Xscaler.transform(new_distances_row)
        distances = np.vstack([distances, new_distances_row])
    if distances.shape[0] == seq_leng:
        distances_list.append(distances)
        distances = distances[1:, :]
        X = distances_list
        X = torch.from_numpy(np.array(X))

        distances_list = []

        Ypred = model(X.float().to(device)).cpu().detach()
        Ypred = torch.cat((Ypred[0, :, :], Ypred[:, -1, :]), dim=0)
        Ypred = Yscaler.inverse_transform(Ypred.cpu())

        timestamp = last_stamp.secs  

        X = (Ypred[:, 0] + Ypred[:, 3]) / 2
        Y = (Ypred[:, 1] + Ypred[:, 4]) / 2      
        dX = np.diff(X, axis=0)
        dY = np.diff(Y, axis=0)
        heading = np.arctan2(dY, dX) 
        unit_vectors = np.array([np.cos(heading), np.sin(heading)])
        mean_vector = np.mean(unit_vectors, axis=1)
        mean_heading = np.arctan2(mean_vector[1], mean_vector[0])
        theta_radian =  mean_heading
        theta_degree = math.degrees(theta_radian)

        pos_x = np.mean(Ypred[seq_leng, [0, 3]])
        pos_y = np.mean(Ypred[seq_leng:, [1, 4]])
        pos_z = np.mean(Ypred[seq_leng:, [2, 5]])
       
        # print(pos_x , pos_y , pos_z, theta_degree)
        publish_pose(pos_x, pos_y, pos_z, theta_radian)  # Publish the new pose

# Function to publish the PoseStamped message
def publish_pose(pos_x, pos_y, pos_z, theta_radian):
    """Function to publish the PoseStamped message."""
    global last_pose, last_publish_time

    pose_msg = PoseStamped()
    pose_msg.header.stamp = last_stamp
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

    last_pose = pose_msg  # Store the last published pose
    last_publish_time = time.time()  # Update the last publish time

# Function to check the time since the last publish and resend the last pose if needed
def check_and_publish_last_pose():
    """Function to check the time since the last publish and resend the last pose if needed."""
    global last_pose, last_publish_time

    if last_pose is not None and (time.time() - last_publish_time > publish_interval):
        new_stamp = rospy.Time(last_stamp.secs, last_stamp.nsecs) + rospy.Duration(0.05)
        last_pose.header.stamp = new_stamp
        pub.publish(last_pose)
        last_publish_time = time.time()  # Update the last publish time

# Callback function for data received from tag 0
def callback_tag0(data):
    """Callback function for data received from tag 0."""
    global buffer_tag0, buffer_tag1, last_stamp
    buffer_tag0 = extract_distances(data)
    last_stamp = data.header.stamp
    if buffer_tag1 is not None:
        add_to_distances(buffer_tag0, buffer_tag1)
        buffer_tag0 = None
        buffer_tag1 = None

# Callback function for data received from tag 1
def callback_tag1(data):
    """Callback function for data received from tag 1."""
    global buffer_tag0, buffer_tag1, last_stamp
    buffer_tag1 = extract_distances(data)
    last_stamp = data.header.stamp
    if buffer_tag0 is not None:
        add_to_distances(buffer_tag0, buffer_tag1)
        buffer_tag0 = None
        buffer_tag1 = None

# Initialize the ROS node and subscribe to the relevant topics
def listener():
    """Initialize the ROS node and subscribe to the relevant topics."""
    rospy.init_node('nlink_listener', anonymous=True)
    try:
        rospy.Subscriber("/ltp_tag0/nlnf3", LinktrackNodeframe3, callback_tag0)
        rospy.Subscriber("/ltp_tag1/nlnf3", LinktrackNodeframe3, callback_tag1)
    except Exception as e:
        rospy.logerr("Failed to subscribe to topics: %s", str(e))  # Log error if subscription fails
    
    rate = rospy.Rate(5) 
    while not rospy.is_shutdown():
        check_and_publish_last_pose()
        rate.sleep()

# Main function to initialize model and start listener
def main():
    initialize_model()
    listener()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
