#ifndef _TESTFORCE_PLUGIN_HH_
#define _TESTFORCE_PLUGIN_HH_

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
// #include <gazebo/math/gzmath.hh>
// #include <ignition/math/Vector3.hh>
// #include <ignition/math/Quaternion.hh>
#include <ignition/math/Pose3.hh>
// #include <math.h>
#include<cmath>
#include<geometry_msgs/Pose.h>
#include <gazebo/physics/Base.hh>
#include <gazebo/physics/Link.hh>
#include <thread>
#include "ros/ros.h"
#include "ros/callback_queue.h"
#include "ros/subscribe_options.h"
#include "std_msgs/Float32MultiArray.h"
#include <gazebo/common/common.hh>
//#include "ros/Quaternion.h"
//#include "ros/Matrix3x3.h"
//#include "sensor_msgs/ChannelFloat32.h"

namespace gazebo
{
  /// \brief A plugin to control a Velodyne sensor.
  class TestforcePlugin : public WorldPlugin
  {
    /// \brief Constructor
    public: TestforcePlugin() {
    printf("=============================\n");
    printf("load rcdpr_plugin success!!!!!!!!!\n");
    printf("===========================\n");
    }

    public: virtual void Load(physics::WorldPtr _parent, sdf::ElementPtr _sdf)
    {
     this->world=_parent;
     // this->model = this->world->GetModel("testforce3");
     this->logger0_model_base = this->world->ModelByName ("omni_car_0");
     this->logger1_model_base = this->world->ModelByName ("omni_car_1");
     this->logger2_model_base = this->world->ModelByName ("omni_car_2");
     this->model_child = this->world->ModelByName ("payload");
     this->drone1_model = this->world->ModelByName("drone1");
     this->drone2_model = this->world->ModelByName("drone2");
     this->drone3_model = this->world->ModelByName("drone3");
     this->drone4_model = this->world->ModelByName("drone4");
     // this->model=_model;
     this->toplink=this->model_child->GetLink	(	"payload_ball::base_link"	);

     this->toprod0=this->logger0_model_base->GetLink	(	"omni_car_0::dummy"	);
     this->toprod1=this->logger1_model_base->GetLink	(	"omni_car_1::dummy"	);
     this->toprod2=this->logger2_model_base->GetLink	(	"omni_car_2::dummy"	);
    //UAV
     this->toprod4=this->drone1_model->GetLink	(	"iris_demo::iris::base_link"	);
     this->toprod5=this->drone2_model->GetLink	(	"iris_demo::iris::base_link"	);
     this->toprod6=this->drone3_model->GetLink	(	"iris_demo::iris::base_link"	);
     this->toprod7=this->drone4_model->GetLink	(	"iris_demo::iris::base_link"	);


     // Initialize ros, if it has not already bee initialized.
     if (!ros::isInitialized())
     {
       int argc = 0;
       char **argv = NULL;
       ros::init(argc, argv, "gazebo_client",
       ros::init_options::NoSigintHandler);
     }

     // Create our ROS node. This acts in a similar manner to
     // the Gazebo node
     this->rosNode.reset(new ros::NodeHandle("gazebo_client"));
     this->prevtime=this->world->SimTime();//获取世界模拟时间，
// Create a named topic, and subscribe to it.
//++++++++++++++++++++++++++++++++++++++++++

    ros::SubscribeOptions so0 =
    ros::SubscribeOptions::create<std_msgs::Float32MultiArray>(
       "/rcdpr_force",
      100,
      boost::bind(&TestforcePlugin::OnRosMsg0, this, _1),
      ros::VoidPtr(), &this->rosQueue);
      this->rosSub0 = this->rosNode->subscribe(so0);

// Spin up the queue helper thread.
//  this->rosQueueThread =
//    std::thread(std::bind(&TestforcePlugin::QueueThread, this));
//    +++++++++++++++++++++++++++++++++++++++

// Spin up the queue helper thread.
  this->rosQueueThread =
    std::thread(std::bind(&TestforcePlugin::QueueThread, this));
//    &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    std::cerr <<"testcycle\n";
    }

   public: void ApplyForce0(const double &_force0,const double &_force1,const double &_force2,const double &_force4, const double &_force5, const double &_force6, const double &_force7)
    {
        count_drone_and_car +=1;
      //calculate the force
      this->bot_pos0=this->toprod0->WorldCoGPose();
      this->bot_position0=this->bot_pos0.Pos();
      this->force0=this->CalForce(_force0,this->pos1,this->bot_position0);
      this->bot_pos1=this->toprod1->WorldCoGPose();
      this->bot_position1=this->bot_pos1.Pos();
      this->force1=this->CalForce(_force1,this->pos2,this->bot_position1);
      this->bot_pos2=this->toprod2->WorldCoGPose();
      this->bot_position2=this->bot_pos2.Pos();
      this->force2=this->CalForce(_force2,this->pos3,this->bot_position2);

    this->bot_pos4=this->toprod4->WorldCoGPose();
    this->bot_position4=this->bot_pos4.Pos();
    this->force4=this->CalForce(_force4,this->pos4,this->bot_position4);

    this->bot_pos5=this->toprod5->WorldCoGPose();
    this->bot_position5=this->bot_pos5.Pos();
    this->force5=this->CalForce(_force5,this->pos5,this->bot_position5);

    this->bot_pos6=this->toprod6->WorldCoGPose();
    this->bot_position6=this->bot_pos6.Pos();
    this->force6=this->CalForce(_force6,this->pos6,this->bot_position6);

    this->bot_pos7=this->toprod7->WorldCoGPose();
    this->bot_position7=this->bot_pos7.Pos();
    this->force7=this->CalForce(_force7,this->pos7,this->bot_position7);

      //count is the number of loops for force application
      this->toplink->AddForceAtRelativePosition(	this->force0,this->pos1);
      this->toplink->AddForceAtRelativePosition(	this->force1,this->pos2);
      this->toplink->AddForceAtRelativePosition(	this->force2,this->pos3);
      // this->toprod0->AddForceAtRelativePosition(	-1.0*this->force0,this->pos_car1);
      // this->toprod1->AddForceAtRelativePosition(	-1.0*this->force1,this->pos_car2);
      // this->toprod2->AddForceAtRelativePosition(	-1.0*this->force2,this->pos_car3);

     //末端执行器所受的拉力
    this->toplink->AddForceAtRelativePosition(	this->force4,this->pos4);
      //无人机所受到的绳子拉力
    // this->toprod4->AddForceAtRelativePosition(	-1.0*this->force4,this->pos_drone1);
       //末端执行器所受的拉力
    this->toplink->AddForceAtRelativePosition(	this->force5,this->pos5);
      //无人机所受到的绳子拉力
    // this->toprod5->AddForceAtRelativePosition(	-1.0*this->force5,this->pos_drone2);
    //末端执行器所受的拉力
    this->toplink->AddForceAtRelativePosition(	this->force6,this->pos6);
      //无人机所受到的绳子拉力
    // this->toprod6->AddForceAtRelativePosition(	-1.0*this->force6,this->pos_drone3);
     //末端执行器所受的拉力
    this->toplink->AddForceAtRelativePosition(	this->force7,this->pos7);
      //无人机所受到的绳子拉力
    // this->toprod7->AddForceAtRelativePosition(	-1.0*this->force7,this->pos_drone4);
    
  }

     //std::cerr << "Force applied\n";
  public: ignition::math::Vector3d CalForce(const double &_force, const ignition::math::Vector3d &pos, const ignition::math::Vector3d &bot_pos)
  {
    //Calculate the force vector according to the pose of link
    //Get the absolute position of
    this->toplinkpose=this->toplink->WorldCoGPose();
    this->toplinkposition=this->toplinkpose.Pos();
    this->toplinkattitude=this->toplinkpose.Rot();
//    std::cerr <<"toplink pos:"<< this->toplinkposition<<'\n';
//    std::cerr <<"bottlem pos:"<< bot_pos<<'\n';
//    std::cerr <<"base_link rot:"<< this->toplinkattitude<<'\n';
    this->force_dir=bot_pos- this->toplinkposition - this->toplinkattitude.RotateVector(pos)  ;
//    this->force_dir=bot_pos- this->toplinkposition  ;
//     std::cerr <<"distance :"<< this->distance<<'\n';
//    this->force_dir=this->test  ;
//     std::cerr <<"force direction:"<< this->force_dir.Normalize()<<'\n';
    return _force * this->force_dir.Normalize();

  }

    /// \brief Handle an incoming message from ROS

// car和无人机整合
public: void OnRosMsg0(const std_msgs::Float32MultiArrayConstPtr &_msg)
{

  if (this->world->SimTime() - this->prevtime >= this->timeinterval||this->world->SimTime() - this->prevtime<0)
  {
    std::cerr <<"timedifference:"<<this->world->SimTime() - this->prevtime<<'\n';
    this->ApplyForce0(_msg->data[0], _msg->data[1], _msg->data[2],_msg->data[3], _msg->data[4], _msg->data[5], _msg->data[6]);
    this->prevtime=this->world->SimTime();
    std::cerr <<"===========1============"<<'\n';
    std::cerr <<"force_logger0: "<<_msg->data[0]<<'\n'<<'\n';
    std::cerr <<"force_logger1: "<<_msg->data[1]<<'\n'<<'\n';
    std::cerr <<"force_logger2: "<<_msg->data[2]<<'\n'<<'\n';
    std::cerr <<"force_drone1: "<<_msg->data[3]<<'\n'<<'\n';
    std::cerr <<"force_drone2: "<<_msg->data[4]<<'\n'<<'\n';
    std::cerr <<"force_drone3: "<<_msg->data[5]<<'\n'<<'\n';
    std::cerr <<"force_drone4: "<<_msg->data[6]<<'\n'<<'\n';
    std::cerr <<"count: "<<count_drone_and_car << '\n'<<'\n';
    std::cerr <<common::Time::GetWallTime()<<'\n'<<'\n';
    std::cerr <<"+++++++++++++2+++++++++++++++"<<'\n';
  }
    
}

/// \brief ROS helper function that processes messages
private: void QueueThread()
{
  static const double timeout = 0.01;
  while (this->rosNode->ok())
  {
    this->rosQueue.callAvailable(ros::WallDuration(timeout));
  }
}
    //set the forces and positions
  private:
    //4 cable-drag force
    //初始化力
       double fx1=0, fy1=0, fz1=0;
    //parameters of 8 cable nodes' position
       //end-effector link nodes positions in its own coordinate
       //末端执行器链接节点在其自身坐标中的位置----球半径是0.05，圆心在中心
      double const posx1=00, posy1=0.0,posz1=-0.0;//对应ugv1
      double const posx2=-0.0, posy2=0.0,posz2=0.0; //对应ugv2
      double const posx3=0.0, posy3=0.0,posz3=-0.0;//对应ugv3

       double const posx4=0.0, posy4=0.0,posz4=0.0;//对应无人机1
       double const posx5=-0.0, posy5=-0.0,posz5=0.0; //对应无人机2
       double const posx6=0.0, posy6=-0.0,posz6=0.0;//对应无人机3
       double const posx7=0.0, posy7=0.0,posz7=0.0;//对应无人机4
       //无人机端链接节点在其自身坐标中的位置
       double const bot_posx1=0.0, bot_posy1=0.0,bot_posz1=0.0;
       double const bot_posx2=0.0, bot_posy2=0.0,bot_posz2=0.0;
       double const bot_posx3=0.0, bot_posy3=0.0,bot_posz3=0.0;
       double const bot_posx4=0.0, bot_posy4=0.0,bot_posz4=0.0;

       //小车端链接节点在其自身坐标中的位置
       double const bot_carposx1=0.0, bot_carposy1=0.0,bot_carposz1=0.0;
       double const bot_carposx2=0.0, bot_carposy2=0.0,bot_carposz2=0.0;
       double const bot_carposx3=0.0, bot_carposy3=0.0,bot_carposz3=0.0;

    //state the class of the variable used
    private:
    physics::ModelPtr logger0_model_base;
    physics::ModelPtr logger1_model_base;
    physics::ModelPtr logger2_model_base;
    physics::ModelPtr drone1_model;
    physics::ModelPtr drone2_model;
    physics::ModelPtr drone3_model;
    physics::ModelPtr drone4_model;

    physics::ModelPtr model_child;
    physics::WorldPtr world;
    physics::LinkPtr toplink;
    physics::LinkPtr toprod0;
    physics::LinkPtr toprod1;
    physics::LinkPtr toprod2;
    //UAV
    physics::LinkPtr toprod4;
    physics::LinkPtr toprod5;
    physics::LinkPtr toprod6;
    physics::LinkPtr toprod7;
    //four forces on the cable
    ignition::math::Vector3d force0=ignition::math::Vector3d(fx1,fy1,fz1);
    ignition::math::Vector3d force1=ignition::math::Vector3d(fx1,fy1,fz1);
    ignition::math::Vector3d force2=ignition::math::Vector3d(fx1,fy1,fz1);
    //UAV
    ignition::math::Vector3d force4=ignition::math::Vector3d(fx1,fy1,fz1);
    ignition::math::Vector3d force5=ignition::math::Vector3d(fx1,fy1,fz1);
    ignition::math::Vector3d force6=ignition::math::Vector3d(fx1,fy1,fz1);
    ignition::math::Vector3d force7=ignition::math::Vector3d(fx1,fy1,fz1);


    //relative position of nodes on the toplink
    //末端执行器的各个固定点相对于自身的位置3ugv，4uav
    ignition::math::Vector3d const pos1= ignition::math::Vector3d(posx1,posy1,posz1);
    ignition::math::Vector3d const pos2= ignition::math::Vector3d(posx2,posy2,posz2);
    ignition::math::Vector3d const pos3= ignition::math::Vector3d(posx3,posy3,posz3);
    ignition::math::Vector3d const pos4= ignition::math::Vector3d(posx4,posy4,posz4);
    ignition::math::Vector3d const pos5= ignition::math::Vector3d(posx5,posy5,posz5);
    ignition::math::Vector3d const pos6= ignition::math::Vector3d(posx6,posy6,posz6);
    ignition::math::Vector3d const pos7= ignition::math::Vector3d(posx7,posy7,posz7);
        //无人机的绳子接触点相对于自身坐标系的位置
    ignition::math::Vector3d const pos_drone1= ignition::math::Vector3d(bot_posx1,bot_posy1,bot_posz1);
    ignition::math::Vector3d const pos_drone2= ignition::math::Vector3d(bot_posx2,bot_posy2,bot_posz2);
    ignition::math::Vector3d const pos_drone3= ignition::math::Vector3d(bot_posx3,bot_posy3,bot_posz3);
    ignition::math::Vector3d const pos_drone4= ignition::math::Vector3d(bot_posx4,bot_posy4,bot_posz4);
    //小车的绳子接触点相对于自身坐标系的位置
    ignition::math::Vector3d const pos_car1= ignition::math::Vector3d(bot_carposx1,bot_carposy1,bot_carposz1);
    ignition::math::Vector3d const pos_car2= ignition::math::Vector3d(bot_carposx2,bot_carposy2,bot_carposz2);
    ignition::math::Vector3d const pos_car3= ignition::math::Vector3d(bot_carposx3,bot_carposy3,bot_carposz3);

    ignition::math::Vector3d const test= ignition::math::Vector3d(0,0,1);

    //absolute position of nodes on the bottom link

    //temporaty variable for force calculation
    ignition::math::Vector3d force_dir;
    ignition::math::Pose3d toplinkpose;
    ignition::math::Pose3d bot_pos0;
    ignition::math::Pose3d bot_pos1;
    ignition::math::Pose3d bot_pos2;
    ignition::math::Vector3d bot_position0;
    ignition::math::Vector3d bot_position1;
    ignition::math::Vector3d bot_position2;
    //UAV
    ignition::math::Pose3d bot_pos4;
    ignition::math::Pose3d bot_pos5;
    ignition::math::Pose3d bot_pos6;
    ignition::math::Pose3d bot_pos7;
    ignition::math::Vector3d bot_position4;
    ignition::math::Vector3d bot_position5;
    ignition::math::Vector3d bot_position6;
    ignition::math::Vector3d bot_position7;


    ignition::math::Vector3d toplinkposition;
    ignition::math::Quaterniond toplinkattitude;


    common::Time timeinterval=common::Time(0, common::Time::SecToNano(0.001));
    common::Time prevtime;

    /// \brief A node use for ROS transport
private: std::unique_ptr<ros::NodeHandle> rosNode;

/// \brief A ROS subscriber
private: ros::Subscriber rosSub0;

/// \brief A ROS callbackqueue that helps process messages
private: ros::CallbackQueue rosQueue;

/// \brief A thread the keeps running the rosQueue
private: std::thread rosQueueThread;
private: int count_drone = 0;
private: int count_car = 0;
private: int count_drone_and_car = 0;

  };

  // Tell Gazebo about this plugin, so that Gazebo can call Load on this plugin.
  GZ_REGISTER_WORLD_PLUGIN(TestforcePlugin)
}
#endif