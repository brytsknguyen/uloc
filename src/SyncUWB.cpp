#include "ros/ros.h"
#include "nlink_parser/LinktrackNodeframe3.h"
#include <uwbreloc/X.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <vector>
#include <deque>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>

// Define a comparator function for sorting nodes by their ID
bool compareByID(const nlink_parser::LinktrackNode2& node1, const nlink_parser::LinktrackNode2& node2)
{
    return node1.id < node2.id;
}

class SyncUWB
{
private:
    // ROS NodeHandle pointer for managing the ROS node
    ros::NodeHandlePtr nh_ptr;

    // Message filter subscribers
    message_filters::Subscriber<nlink_parser::LinktrackNodeframe3> sub0_;
    message_filters::Subscriber<nlink_parser::LinktrackNodeframe3> sub1_;

    // Synchronizer for Approximate Time
    typedef message_filters::sync_policies::ApproximateTime<nlink_parser::LinktrackNodeframe3, nlink_parser::LinktrackNodeframe3> SyncPolicy;
    message_filters::Synchronizer<SyncPolicy> sync_;

    // Buffer for storing synchronized messages
    std::deque<std::pair<nlink_parser::LinktrackNodeframe3::ConstPtr, nlink_parser::LinktrackNodeframe3::ConstPtr>> sync_buffer_;

    // Mutex for safe access to shared resources
    std::mutex buffer_mutex_;

    // Condition variable to notify the processing thread
    std::condition_variable data_ready_;

    // Thread for processing buffer
    std::thread processing_thread_;

    // Flag to indicate if processing should continue
    bool processing_active_;
    
    // ROS Publisher to publish uwbreloc::X message
    ros::Publisher pub_;

    int seqlen = 20;

public:
    // Destructor
    ~SyncUWB()
    {
        // Stop the processing thread
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            processing_active_ = false;
        }
        data_ready_.notify_all();
        if (processing_thread_.joinable())
        {
            processing_thread_.join();
        }
    }

    // Constructor
    SyncUWB(ros::NodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_), 
                                           sub0_(*nh_ptr_, "/ltp_tag0/nlnf3", 1), 
                                           sub1_(*nh_ptr_, "/ltp_tag1/nlnf3", 1), 
                                           sync_(SyncPolicy(10), sub0_, sub1_),
                                           processing_active_(true)
    {
        sync_.registerCallback(boost::bind(&SyncUWB::syncCallback, this, _1, _2));

        // Initialize publisher for uwbreloc::X message
        pub_ = nh_ptr->advertise<uwbreloc::X>("X_topic", 10);

        // Start the processing thread
        processing_thread_ = std::thread(&SyncUWB::ProcessBuffer, this);
    }

    // Callback function that handles synchronized messages from both tags
    void syncCallback(const nlink_parser::LinktrackNodeframe3::ConstPtr& msg0, const nlink_parser::LinktrackNodeframe3::ConstPtr& msg1)
    {
        // Check if both messages have data in their nodes
        if (msg0->nodes.empty() || msg1->nodes.empty())
        {
            return; // Do not add to the buffer if either message has no nodes
        }

        {
            // Lock mutex to safely add synchronized messages to the buffer
            std::lock_guard<std::mutex> lock(buffer_mutex_);

            // Push synchronized messages into the buffer as a pair
            sync_buffer_.emplace_back(msg0, msg1);
        }

        // Notify the processing thread that new data is available
        data_ready_.notify_one();
        // ROS_INFO("Synchronized Data Added to Buffer. Buffer size: %ld", sync_buffer_.size());
    }

    // Process messages in the buffer without removing them
    void ProcessBuffer()
    {   
        std::vector<float> concatenated_400;  // Create a vector to store 400 distances
        concatenated_400.reserve(400);  // Reserve space for 400 elements

        while (processing_active_)
        {
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            // Wait until we have at least 20 pairs in the buffer or stop processing
            data_ready_.wait(lock, [this]{ return sync_buffer_.size() >= 20 || !processing_active_; });

            if (!processing_active_) break;

            // Check if we have enough data (at least 20 pairs)
            if (sync_buffer_.size() >= 20)
            {
                // Get the timestamp of the first and the twentieth message in the buffer
                auto first_timestamp = sync_buffer_.front().first->header.stamp;
                auto last_timestamp = sync_buffer_[19].first->header.stamp;

                // Calculate the time difference between the first and twentieth message
                ros::Duration time_span = last_timestamp - first_timestamp;

                // Process the data only if the time span is less than or equal to 2 seconds
                if (time_span.toSec() <= (seqlen * 0.05 * 2))
                {
                    // ROS_INFO("Processing 20 synchronized pairs from the buffer with a time span of %f seconds.", time_span.toSec());

                    for (size_t i = 0; i < seqlen; ++i)
                    {
                        auto sync_pair = sync_buffer_.front(); // Get the oldest pair

                        // Create distance vectors for both Tag 0 and Tag 1
                        std::vector<float> distances0(10, 0.0);  // 10 nodes, all initialized to 0
                        std::vector<float> distances1(10, 0.0);  // 10 nodes, all initialized to 0

                        // Get nodes from the current synchronized messages
                        std::vector<nlink_parser::LinktrackNode2> nodes0 = sync_pair.first->nodes;
                        std::vector<nlink_parser::LinktrackNode2> nodes1 = sync_pair.second->nodes;

                        // Populate distances0 for Tag 0
                        for (const auto& node : nodes0)
                        {
                            if (node.id < 10)  // Ensure node ID is within the valid range
                            {
                                distances0[node.id] = node.dis;
                            }
                        }

                        // Populate distances1 for Tag 1
                        for (const auto& node : nodes1)
                        {
                            if (node.id < 10)  // Ensure node ID is within the valid range
                            {
                                distances1[node.id] = node.dis;
                            }
                        }

                        // Concatenate distances0 and distances1 into a single array
                        std::vector<float> distances_20;
                        distances_20.reserve(20);  // Reserve space for 20 elements
                        distances_20.insert(distances_20.end(), distances0.begin(), distances0.end());  // Insert Tag 0 distances
                        distances_20.insert(distances_20.end(), distances1.begin(), distances1.end());  // Insert Tag 1 distances

                        // Add this 20-element array to the concatenated_400 array
                        concatenated_400.insert(concatenated_400.end(), distances_20.begin(), distances_20.end());


                        // Check if we have reached 400 elements
                        if (concatenated_400.size() >= 400)
                        {
                            ROS_INFO("Collected 400 distances. Publishing message.");

                            // Create the message of type uwbreloc::X
                            uwbreloc::X msg;

                            // Fill in the message data
                            msg.header.stamp = first_timestamp;  // Set the timestamp
                            std::copy(concatenated_400.begin(), concatenated_400.end(), msg.distances.begin());  // Fill the distances array

                            // Publish the message
                            pub_.publish(msg);

                            // // Optional: Display the 400 distances (for debugging)
                            // for (size_t j = 0; j < concatenated_400.size(); ++j)
                            // {
                            //     ROS_INFO("Index: %lu, Distance: %f", j, concatenated_400[j]);
                            // }

                            // Clear the concatenated_400 vector after processing
                            concatenated_400.clear();
                        }
                    }
                    ROS_INFO("----------------------------");
                }
                else
                {
                    sync_buffer_.pop_front();
                    // ROS_WARN("Time span between the first and the twentieth message exceeds 2 second. Waiting for more data.");
                }
                // Remove the oldest processed pair from the buffer
                sync_buffer_.pop_front();
            }
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "SyncUWB");

    // Create a NodeHandle pointer to manage the ROS node
    ros::NodeHandlePtr nh_ptr(new ros::NodeHandle);

    // Create an instance of the SyncUWB class
    SyncUWB syncuwb(nh_ptr);

    // Create a MultiThreadedSpinner to handle callbacks
    ros::MultiThreadedSpinner spinner(0);
    spinner.spin();

    return 0;
}
