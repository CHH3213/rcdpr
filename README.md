# rcdpr

> 2021/12/15

## Environments
- 整个环境是在`apm`飞控下运行成功的。所以在运行本环境之前，需要安装好mavros和apm，[Intelligent Quads Tutorials](https://github.com/Intelligent-Quads/iq_tutorials)
- 安装好后，将本功能包从仓库上git clone下来：
  ```bash
  git clone https://github.com/CHH3213/rcdpr.git
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

  在文件末尾添加以下两行：

  ```bash
  export GAZEBO_PLUGIN_PATH=${GAZEBO_PLUGIN_PATH}:~/ros_ws/src/rcdpr/worlds/plugins/build
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:~/ros_ws/src/rcdpr/worlds/plugins/build
  ```
  路径千万别写错，否则力的插件加载会失败。

  添加完后关闭bashrc，并source：

  ```bash
  source ~/.bashrc
  ```
- 进入bash文件夹，给bash脚本添加权限

  ```bash
  chmod a+x rcdpr.sh
  ```


- 运行bash脚本，正常可以打开环境：

  ```
  ./rcdpr.sh
  ```



## launch文件夹说明
- `rcdpr-apm.launch `:mavros开启
- `rcdpr_env.launch`：仿真环境launch文件

## bash文件夹说明
- `reset_params.sh`和`rc_multi_drone.sh`是修改`rc/override`的，在这里未用到。
- `rcdpr.sh`：是整个项目运行的脚本
## 运行

在`scripts`文件夹下，主要的main函数有`online_phase_network.py`,`online_phase_optimizer.py`
- 首先运行`cmd_force.py`
- 而后运行相应的main函数：
  - `python online_phase_network.py`

