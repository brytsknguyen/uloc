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
    ros::NodeHandlePtr nh_ptr;

    message_filters::Subscriber<nlink_parser::LinktrackNodeframe3> sub0_;
    message_filters::Subscriber<nlink_parser::LinktrackNodeframe3> sub1_;

    typedef message_filters::sync_policies::ApproximateTime<nlink_parser::LinktrackNodeframe3, nlink_parser::LinktrackNodeframe3> SyncPolicy;
    message_filters::Synchronizer<SyncPolicy> sync_;

    std::deque<std::pair<nlink_parser::LinktrackNodeframe3::ConstPtr, nlink_parser::LinktrackNodeframe3::ConstPtr>> sync_buffer_;
    std::mutex buffer_mutex_;
    std::condition_variable data_ready_;
    std::thread processing_thread_;
    bool processing_active_;

    ros::Publisher pub_;

    int seqlen = 20;
    ros::Timer empty_buffer_timer_;  // Timer to check if buffer is empty
    ros::Time last_data_time_;       // Tracks the last time data was received
    float  timeout = 20.0;

public:
    ~SyncUWB()
    {
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

    SyncUWB(ros::NodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_), 
                                           sub0_(*nh_ptr_, "/ltp_tag0/nlnf3", 1), 
                                           sub1_(*nh_ptr_, "/ltp_tag1/nlnf3", 1), 
                                           sync_(SyncPolicy(10), sub0_, sub1_),
                                           processing_active_(true)
    {
        sync_.registerCallback(boost::bind(&SyncUWB::syncCallback, this, _1, _2));

        pub_ = nh_ptr->advertise<uwbreloc::X>("X_topic", 10);

        processing_thread_ = std::thread(&SyncUWB::ProcessBuffer, this);

        // Initialize the timer to check if the buffer is empty every second
        empty_buffer_timer_ = nh_ptr->createTimer(ros::Duration(1.0), &SyncUWB::checkEmptyBuffer, this);

        last_data_time_ = ros::Time::now();  // Initialize the last data time
    }

    void syncCallback(const nlink_parser::LinktrackNodeframe3::ConstPtr& msg0, const nlink_parser::LinktrackNodeframe3::ConstPtr& msg1)
    {
        if (msg0->nodes.empty() || msg1->nodes.empty())
        {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            sync_buffer_.emplace_back(msg0, msg1);
        }

        last_data_time_ = ros::Time::now();  // Update the last data time
        data_ready_.notify_one();
    }

    void ProcessBuffer()
    {   
        std::vector<float> concatenated;
        concatenated.reserve(seqlen * 20);

        while (processing_active_)
        {
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            data_ready_.wait(lock, [this]{ return sync_buffer_.size() >= seqlen || !processing_active_; });

            if (!processing_active_) break;

            if (sync_buffer_.size() >= seqlen)
            {
                auto first_timestamp = sync_buffer_.front().first->header.stamp;
                auto last_timestamp = sync_buffer_[(seqlen - 1)].first->header.stamp;

                ros::Duration time_span = last_timestamp - first_timestamp;

                if (time_span.toSec() <= (seqlen * 0.05 * 2))
                {
                    for (size_t i = 0; i < seqlen; ++i)
                    {
                        auto sync_pair = sync_buffer_.front();

                        std::vector<float> distances0(10, 0.0);
                        std::vector<float> distances1(10, 0.0);

                        std::vector<nlink_parser::LinktrackNode2> nodes0 = sync_pair.first->nodes;
                        std::vector<nlink_parser::LinktrackNode2> nodes1 = sync_pair.second->nodes;

                        for (const auto& node : nodes0)
                        {
                            if (node.id < 10)
                            {
                                distances0[node.id] = node.dis;
                            }
                        }

                        for (const auto& node : nodes1)
                        {
                            if (node.id < 10)
                            {
                                distances1[node.id] = node.dis;
                            }
                        }

                        std::vector<float> distances;
                        distances.reserve(seqlen);
                        distances.insert(distances.end(), distances0.begin(), distances0.end());
                        distances.insert(distances.end(), distances1.begin(), distances1.end());

                        concatenated.insert(concatenated.end(), distances.begin(), distances.end());

                        if (concatenated.size() >= (seqlen * 20))
                        {
                            ROS_INFO("Collected seqlen distances. Publishing message.");

                            uwbreloc::X msg;
                            msg.header.stamp = first_timestamp;
                            std::copy(concatenated.begin(), concatenated.end(), msg.distances.begin());

                            pub_.publish(msg);
                            concatenated.clear();
                        }
                    }
                }

                sync_buffer_.pop_front();
            }
        }
    }

    // Timer callback to check if buffer has been empty for 10 seconds
    void checkEmptyBuffer(const ros::TimerEvent&)
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        ros::Duration time_since_last_data = ros::Time::now() - last_data_time_;

        if ((time_since_last_data.toSec() >= timeout) && (sync_buffer_.size() < seqlen))
        {
            ROS_WARN("No data received. Stopping the node.");
            ros::shutdown();
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "SyncUWB");

    ros::NodeHandlePtr nh_ptr(new ros::NodeHandle);

    SyncUWB syncuwb(nh_ptr);

    ros::MultiThreadedSpinner spinner(0);
    spinner.spin();

    return 0;
}
