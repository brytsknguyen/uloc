#include "ros/ros.h"
#include "nlink_parser/LinktrackNodeframe3.h"
#include <uloc/X.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <vector>
#include <deque>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"
#define RESET "\033[0m"

using namespace std;
typedef nlink_parser::LinktrackNodeframe3 ltpbUwbMsg;
typedef nlink_parser::LinktrackNodeframe3::Ptr ltpbUwbMsgPtr;

// Define a comparator function for sorting nodes by their ID
bool compareByID(const nlink_parser::LinktrackNode2& node1, const nlink_parser::LinktrackNode2& node2)
{
    return node1.id < node2.id;
}

class UwbSync
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
    int inputSize = 20;
    int anchorsNum = 10;
    int minimum_anchors_seen = 1;
    ros::Timer empty_buffer_timer_;  // Timer to check if buffer is empty
    ros::Time last_data_time_;       // Tracks the last time data was received
    float  timeout = 20.0;

public:
    ~UwbSync()
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

    UwbSync(ros::NodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_), 
                                           sub0_(*nh_ptr_, "/ltp_tag0/nlnf3", 1), 
                                           sub1_(*nh_ptr_, "/ltp_tag1/nlnf3", 1), 
                                           sync_(SyncPolicy(50), sub0_, sub1_),
                                           processing_active_(true)
    {
        // Subscribe to the uwb messages
        sync_.registerCallback(boost::bind(&UwbSync::syncCallback, this, _1, _2));
        sync_.setMaxIntervalDuration(ros::Duration(0.05));

        // Advertise the input for the learning model
        pub_ = nh_ptr->advertise<uloc::X>("X_topic", 10);

        // Create the processing thread
        processing_thread_ = std::thread(&UwbSync::ProcessBuffer, this);
    }

    void syncCallback(const nlink_parser::LinktrackNodeframe3::ConstPtr& msg0_, const nlink_parser::LinktrackNodeframe3::ConstPtr& msg1_)
    {
        if (msg0_->nodes.empty() || msg1_->nodes.empty())
            return;

        // Compare with the last range to determine if the range measurements are outliers
        struct dstamped
        {
            dstamped(){};
            dstamped(const double &stamp_, const double &dis_) : stamp(stamp_), dis(dis_){};
            double stamp = -1;
            double dis = -1;
        };

        ltpbUwbMsgPtr msg0(new ltpbUwbMsg()); *msg0 = *msg0_;
        ltpbUwbMsgPtr msg1(new ltpbUwbMsg()); *msg1 = *msg1_;
        
        static vector<dstamped> d0stamped(10, dstamped());
        static vector<dstamped> d1stamped(10, dstamped());
        
        auto checkdis = [](vector<dstamped> &dbuf, ltpbUwbMsgPtr &msg, int tagid)
        {
            // Check the distances measurements
            for(auto &node : msg->nodes)
            {
                const int &nodeId = node.id;
                if (dbuf[nodeId].stamp == -1)
                {
                    dbuf[nodeId] = dstamped(msg->header.stamp.toSec(), node.dis);
                    continue;
                }
                else
                {
                    double dt = fabs(msg->header.stamp.toSec() - dbuf[nodeId].stamp);
                    double dd = fabs(node.dis - dbuf[nodeId].dis);

                    if (dt > 1.0)
                        dbuf[nodeId] = dstamped(msg->header.stamp.toSec(), node.dis);
                    else
                    {
                        if(dd/dt < 10.0)
                            dbuf[nodeId] = dstamped(msg->header.stamp.toSec(), node.dis);
                        else
                        {
                            printf(KRED "Remove the range %d->%d. dis: %f. dt: %f, dd: %f.\n" RESET,
                                    tagid, node.id, node.dis, node.id, dt, dd);
                            node.dis = -1;
                        }
                    }
                }
            }
        };
        // Filter large spike
        checkdis(d0stamped, msg0, 0);
        checkdis(d1stamped, msg1, 1);

        // Store the message
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            sync_buffer_.emplace_back(msg0, msg1);
        }
    }

    void ProcessBuffer()
    {
        std::vector<float> concatenated;
        concatenated.reserve(seqlen * inputSize);

        while (processing_active_)
        {
            if(sync_buffer_.size() < seqlen)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // Now there are at least seqlen messages in buffer
            auto &msg0f = sync_buffer_.front().first;
            auto &msg1f = sync_buffer_.front().second;
            auto &msg0b = sync_buffer_.back().first;
            auto &msg1b = sync_buffer_.back().second;

            // Calculate the time span
            auto first_timestamp = std::min(msg0f->header.stamp, msg1f->header.stamp);
            auto last_timestamp = std::max(msg0b->header.stamp, msg1b->header.stamp);
            ros::Duration time_span = last_timestamp - first_timestamp;

            // Extracto patronum
            if (time_span.toSec() <= (seqlen * 0.05 * 2))
            {
                for (size_t i = 0; i < seqlen; ++i)
                {
                    auto sync_pair = sync_buffer_.front();

                    std::vector<float> distances0(10, 0.0);
                    std::vector<float> distances1(10, 0.0);

                    std::vector<nlink_parser::LinktrackNode2> nodes0 = sync_pair.first->nodes;
                    std::vector<nlink_parser::LinktrackNode2> nodes1 = sync_pair.second->nodes;

                    // Count the number of anchors seen
                    std::map<int, int> anc_checklist;

                    for (const auto& node : nodes0)
                    {
                        if (node.id < anchorsNum && node.id > -1)
                        {
                            distances0[node.id] = node.dis;
                            anc_checklist[node.id] = 1;
                        }
                    }

                    for (const auto& node : nodes1)
                    {
                        if (node.id < anchorsNum && node.id > -1)
                        {
                            distances1[node.id] = node.dis;
                            anc_checklist[node.id] = 1;
                        }
                    }

                    if (anc_checklist.size() >= minimum_anchors_seen)
                    {
                        std::vector<float> distances;
                        distances.reserve(seqlen);
                        distances.insert(distances.end(), distances0.begin(), distances0.end());
                        distances.insert(distances.end(), distances1.begin(), distances1.end());

                        concatenated.insert(concatenated.end(), distances.begin(), distances.end());

                        if (concatenated.size() >= (seqlen * inputSize))
                        {
                            // ROS_INFO("Collected seqlen distances. Publishing message.");
                            uloc::X msg;
                            msg.header.stamp = last_timestamp;
                            std::copy(concatenated.begin(), concatenated.end(), msg.distances.begin());

                            pub_.publish(msg);
                            concatenated.clear();
                        }
                    }
                }
            }

            buffer_mutex_.lock();
            sync_buffer_.pop_front();
            buffer_mutex_.unlock();
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "UwbSync");

    ros::NodeHandlePtr nh_ptr(new ros::NodeHandle);

    UwbSync syncuwb(nh_ptr);

    ros::MultiThreadedSpinner spinner(0);
    spinner.spin();

    return 0;
}
