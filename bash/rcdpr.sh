#!/bin/bash
# 运行后会打开仿真，加载mavros
gnome-terminal --tab --title="killall gzclient" -- bash -c "killall gzclient;sleep 1; "
gnome-terminal --tab --title="killall gzserver" -- bash -c "killall gzserver;sleep 1; "
gnome-terminal --tab --title="conda" -- bash -c "conda activate tftorch;"
gnome-terminal --tab --title="source" -- bash -c "source /home/firefly/chh_ws/devel/setup.sh"
gnome-terminal --tab --title="SITL-gazebo" -- bash -c "roslaunch rcdpr rcdpr_env.launch;exec bash;"
gnome-terminal --tab -- bash -c "sim_vehicle.py -v ArduCopter -f gazebo-drone1 -I0 --out=tcpin:0.0.0.0:8000;sleep 1; "
gnome-terminal --tab -- bash -c "sim_vehicle.py -v ArduCopter -f gazebo-drone2 -I1 --out=tcpin:0.0.0.0:8100;sleep 1; "
gnome-terminal --tab -- bash -c "sim_vehicle.py -v ArduCopter -f gazebo-drone3 -I2 --out=tcpin:0.0.0.0:8200;sleep 1 "
gnome-terminal --tab -- bash -c "sim_vehicle.py -v ArduCopter -f gazebo-drone4 -I3 --out=tcpin:0.0.0.0:8300;sleep 1 "
gnome-terminal --tab --title="source" -- bash -c "source /home/firefly/chh_ws/devel/setup.sh"
gnome-terminal --tab --title="multiMavros" -- bash -c "roslaunch rcdpr rcdpr-apm.launch;exec bash;"

