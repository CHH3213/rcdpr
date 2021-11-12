#!/bin/bash
gnome-terminal --tab  --title="reset_params" -x bash -c "rosrun mavros mavparam load ~/intelligent_quads/default_mavros.parm;exec bash;"
gnome-terminal --tab  --title="reset_params" -x bash -c "sim_vehicle.py -w -v ArduCopter;exec bash;"
