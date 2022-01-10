# rcdpr

> 2021/12/30

## Environments
- 整个环境是在`apm`飞控下运行成功的。所以在运行本环境之前，需要安装好mavros和apm，[Intelligent Quads Tutorials](https://github.com/Intelligent-Quads/iq_tutorials)
- 安装好后，将本功能包从仓库上git clone下来：
```bash
git clone https://github.com/CHH3213/cdpr_uav_ddrive_endition2.git
```


- 将整个`rcdpr`功能包放到ros工作空间中，并编译和source

  ```bash
  cd ~/ros_ws/
  catkin_make
  source devel/setup.sh
  ```
- 编译插件

进入`rcdpr/worlds/plugins/build`文件夹下，将里面所有文件删除，并在终端依次执行:

```bash
cmake ../
make
```
正常情况下将编译成功.

- 将`rcdpr/worlds/plugins/build`的完整路径添加到系统环境变量中。

   首先打开系统环境变量

```bash
gedit ~/.bashrc
```

​		  在文件末尾添加以下两行：

```bash
export GAZEBO_PLUGIN_PATH=${GAZEBO_PLUGIN_PATH}:~/catkin_ws/src/rcdpr/worlds/plugins/build
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:~/catkin_ws/src/rcdpr/worlds/plugins/build
```

​			路径千万别写错，否则力的插件加载会失败。

添加完后关闭bashrc，并source：

```bash
source ~/.bashrc
```
- 进入bash文件夹，给bash脚本添加权限

  ```bash
  chmod a+x start_multi_drone.sh
  chmod a+x ardupilot.sh
  ```


- 运行bash脚本，正常可以打开环境：

  ```
  ./start_multi_drone.sh
  ```

- 打开环境，正常打开后应该会有如下界面：

  <img src="./worlds/fig/env3.png" alt="world" style="zoom:150%;" />

- 如果在运行bash脚本后仿真环境无法正常工作，则依次打开不同终端手动运行：
  ```bash
  roslaunch cdpr_uav_ddrive multi_drone.launch
  ./start_multi_drone.sh
  roslaunch cdpr_uav_ddrive multi-apm.launch
  ```


## launch文件夹说明
- `apm.launch`:单架无人机下的mavros开启
- `multi-apm.launch`:多架无人机下的mavros开启（默认开启三架）
- `multi_drone.launch`：仿真环境launch文件

## bash文件夹说明
- `reset_params.sh`和`rc_multi_drone.sh`是修改`rc/override`的，在这里未用到。
- `multi-ardupilot.sh`：开启软件在环仿真（SITL）
- `start_multi_drone.sh`：是整个项目运行的脚本
## 运行

在`scripts`文件夹下，主要的main函数有`experi_2dBA`,`experi_2dSA`,`experi_3drone`分别表示2架无人机大间距，两架无人机无间距，两架无人机小间距，3架无人机。需要分别打开对应的环境才可以运行(在`multi_drone.launch`文件中切换)。
- 首先运行`cmd_force.py`
- 而后运行相应的main函数：
  - `experi_2dBA.py`:2架无人机大间距下的实验主程序
  - `experi_2dSA.py`:2架无人机小间距下的实验主程序
  - `experi_3drone.py`:3架无人机下的实验主程序

