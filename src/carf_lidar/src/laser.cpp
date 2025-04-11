#include<cmath>
#include <iostream>
#include <ros/ros.h>
#include<Eigen/Dense>
#include "pcl/point_cloud.h"
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/ModelCoefficients.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include<pcl/filters/crop_box.h>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<cv_bridge/cv_bridge.h>
#include <vector>
#include<image_transport/image_transport.h>
#include <pcl/filters/passthrough.h>
#include<tuple>
#include "std_msgs/Float32MultiArray.h"

# define M_PI  3.14159265358979323846 

class pathCreator{
public:
    ros::Subscriber laser_sub;
    ros::Publisher laser_pub;
    ros::Publisher usbCom_pub;

    image_transport::Publisher image_pub;
    pathCreator(ros::NodeHandle &);
    void get_lidar_data(const sensor_msgs::PointCloud2ConstPtr&);
    std::tuple<float,float> get_vel_tetha(int goal_X, int goal_Y);

    const int map_size = 200;
    const int map_m_size = 15;
    float minX, maxX, minY, maxY, minZ, maxZ;
};

pathCreator::pathCreator(ros::NodeHandle &nh){
    laser_sub = nh.subscribe("/velodyne_points", 10, &pathCreator::get_lidar_data, this);
    laser_pub = nh.advertise<sensor_msgs::PointCloud2>("/carf", 10);
    usbCom_pub = nh.advertise<std_msgs::Float32MultiArray>("/usb_com", 10);
    image_transport::ImageTransport it(nh);
    image_pub = it.advertise("/carf_mapImg", 10);

    minX =  0.0;
    minY = -2.5;
    maxX =  5.0;
    maxY =  2.5;
    minZ = -0.2;
    maxZ =  0.0;
}

std::tuple<float,float> pathCreator::get_vel_tetha(int goal_X, int goal_Y){
    float linear_vel = goal_X;
    float alpha = std::atan2(goal_Y, goal_X);
    return std::tuple<float, float>(linear_vel, alpha);
}


void pathCreator::get_lidar_data(const sensor_msgs::PointCloud2ConstPtr& input)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::fromROSMsg(*input, *cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> filter;
    filter.setInputCloud (cloud);
    filter.setFilterFieldName ("x");
    filter.setFilterLimits (minX, maxX);
    filter.filter (*cloudFiltered);
    filter.setInputCloud (cloudFiltered);
    filter.setFilterFieldName ("y");
    filter.setFilterLimits (minY, maxY);
    filter.filter (*cloudFiltered);
    filter.setInputCloud (cloudFiltered);
    filter.setFilterFieldName ("z");
    filter.setFilterLimits (minZ, maxZ);
    filter.filter (*cloudFiltered);

    sensor_msgs::PointCloud2 out;
    pcl::toROSMsg(*cloudFiltered, out);
    out.header = input->header;
    laser_pub.publish(out);

    int map2D[map_size][map_size]{};
    int block = 1;
    cv::Mat mapImage(map_size,map_size, CV_8UC3, cv::Scalar(255,255,255));
    cv::Vec3b block_color(0,0,0);
    int lidarX = (map_size/2), lidarY = (map_size/2);

    cv::circle(mapImage, cv::Point(lidarX, lidarY), 1, cv::Scalar(0,0,255), cv::FILLED, cv::LINE_4);

    int index_X, index_Y;
    for (int i = 0; i < cloudFiltered->size(); i++)
    {
        index_X = int((cloudFiltered->points[i].x*100) / map_m_size) + (map_size/2);
        index_Y = int((cloudFiltered->points[i].y*100) / map_m_size) + (map_size/2);

        if(index_X < 0 ||index_X > 199 ||index_Y < 0 ||index_Y > 199 ){

            std::cout << "x = " << index_X << " y= " << index_Y << std::endl;
            std::cout << "r_x = " << cloudFiltered->points[i].x << ", r_y = "
                      << cloudFiltered->points[i].y << std::endl;
        }

        mapImage.at<cv::Vec3b>(index_X,index_Y) = block_color;
        map2D[index_X][index_Y] = block;
    }

    int y1, y2, y;
    int path = 100;
    cv::Vec3b colorPath(255,0,0);

    std_msgs::Float32MultiArray steeringAngle;

    for (int i = 0; i < map_size; i++)
    {
        y1 = 0;
        y2 = 0;
        y = 0;

        for (int j = 0; j < map_size; j++)
        {
            if (map2D[i][j] == block){y1 = j; break;}
        }

        for (int k = map_size-1; k >=0; k--)
        {
            if (map2D[i][k] == block ){y2 = k; break;}
        }

        if (abs(y1 - y2) > 7)
        {
            y = int((y1 + y2) / 2);
            map2D[i][y] = path;
            mapImage.at<cv::Vec3b>(i,y) = colorPath;
        }
    }
    
    int target_X_axis = 110; 
    int j, goal_X, goal_Y;
    cv::Vec3b colorGoal(0,255,0);
    for (j = 0; j < map_size; j++)
    {
        if (map2D[target_X_axis][j] == path){
            cv::circle(mapImage, cv::Point(j, target_X_axis), 1, colorGoal, cv::FILLED, cv::LINE_4);
            goal_X = target_X_axis - lidarX;
            goal_Y = j - lidarY;
            std::tuple<float, float> goal(get_vel_tetha(goal_X, goal_Y));
            float lin_vel = (std::get<0>(goal) * 30 - 15) / 100.0;
            float alpha = std::get<1>(goal) * 180.0 / M_PI;
            std::cout << "linear_vel: "<< lin_vel << " m/s, alpha: " <<  alpha << " derece" << std::endl;

            if ( alpha < 0){ steeringAngle.data.push_back(200 + alpha); }
            if ( alpha == 0){ steeringAngle.data.push_back(0.01); }
            else if ( alpha > 0){ steeringAngle.data.push_back(alpha); }

            steeringAngle.data.push_back(lin_vel);
            usbCom_pub.publish(steeringAngle);

            break;
        }
    }

    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", mapImage).toImageMsg();
    image_pub.publish(msg);
}
 
int main(int argc, char **argv)  
{
    ros::init(argc,argv,"laser_node");
    ros::NodeHandle nh;
    pathCreator p(nh);
    ros::spin();
  
    return 0;  
}
