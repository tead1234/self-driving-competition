#关闭所有ros节点
~/.stop_ros.sh

cd ~/ros2_ws 
#colcon build --event-handlers  console_direct+  --cmake-args  -DCMAKE_BUILD_TYPE=Release --symlink-install
#单独编译某个包
colcon build --event-handlers  console_direct+  --cmake-args  -DCMAKE_BUILD_TYPE=Release --symlink-install --packages-select xxx

#线速度校准(ROSMentor_Mecanum, MentorPi_Acker)
ros2 launch calibration linear_calib.launch.py

#角速度校准(ROSMentor_Mecanum)
ros2 launch calibration angular_calib.launch.py

#imu校准
ros2 launch ros_robot_controller ros_robot_controller.launch.py
ros2 run imu_calib do_calib --ros-args -r imu:=/ros_robot_controller/imu_raw --param output_file:=/home/ubuntu/ros2_ws/src/calibration/config/imu_calib.yaml

#查看imu校准效果
ros2 launch peripherals imu_view.launch.py

#深度摄像头点云可视化(ascamera)
#深度摄像头RGB图像可视化(ascamera)
ros2 launch peripherals depth_camera.launch.py
rviz2

#单目摄像头可视化
ros2 launch peripherals usb_cam.launch.py
rviz2

#雷达数据可视化
ros2 launch peripherals lidar_view.launch.py

#雷达功能
ros2 launch app lidar_node.launch.py debug:=true
#雷达避障(ROSMentor_Mecanum, MentorPi_Acker)
ros2 service call /lidar_app/enter std_srvs/srv/Trigger {}
ros2 service call /lidar_app/set_running interfaces/srv/SetInt64 "{data: 1}"

#雷达跟随(ROSMentor_Mecanum, MentorPi_Acker)
ros2 service call /lidar_app/enter std_srvs/srv/Trigger {}
ros2 service call /lidar_app/set_running interfaces/srv/SetInt64 "{data: 2}"

#雷达警卫(ROSMentor_Mecanum)
ros2 service call /lidar_app/enter std_srvs/srv/Trigger {}
ros2 service call /lidar_app/set_running interfaces/srv/SetInt64 "{data: 3}"

#巡线
ros2 launch app line_following_node.launch.py debug:=true
ros2 service call /line_following/enter std_srvs/srv/Trigger {}
#鼠标左键点击画面取色
ros2 service call /line_following/set_running std_srvs/srv/SetBool "{data: True}"

#目标跟踪
ros2 launch app object_tracking_node.launch.py debug:=true
ros2 service call /object_tracking/enter std_srvs/srv/Trigger {}
#鼠标左键点击画面取色
ros2 service call /object_tracking/set_running std_srvs/srv/SetBool "{data: True}"


#hand_gesture
ros2 launch app hand_gesture_node.launch.py debug:=true
ros2 service call /hand_gesture/enter std_srvs/srv/Trigger {}
ros2 service call /hand_gesture/set_running std_srvs/srv/SetBool "{data: True}"

#二维码生成
cd ~/ros2_ws/src/example/example/qrcode && python3 qrcode_creater.py
#####################
ros2 launch peripherals depth_camera.launch.py

#二维码检测
cd ~/ros2_ws/src/example/example/qrcode && python3 qrcode_detecter.py

#人脸检测
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 face_detect.py

#人脸网格
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 face_mesh.py

#手关键点检测
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 hand.py

#肢体关键点检测
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 pose.py

#背景分割
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 self_segmentation.py

#整体检测
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 holistic.py

#3D物体检测
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 objectron.py

#指尖轨迹
cd ~/ros2_ws/src/example/example/mediapipe_example && python3 hand_gesture.py

#颜色识别
cd ~/ros2_ws/src/example/example/color_detect && python3 color_detect_demo.py

#无人驾驶
ros2 launch example self_driving.launch.py

#2D建图
ros2 launch slam slam.launch.py


#rviz查看建图效果
ros2 launch slam rviz_slam.launch.py


#键盘控制(可选)
ros2 launch peripherals teleop_key_control.launch.py


#保存地图
#/home/ubuntu/ros2_ws/src/slam/maps/map_01.yaml
cd ~/ros2_ws/src/slam/maps && ros2 run nav2_map_server map_saver_cli -f "map_01" --ros-args -p map_subscribe_transient_local:=true

#cd ~/ros2_ws/src/slam/maps && ros2 run nav2_map_server map_saver_cli -f "保存名称" --ros-args -p map_subscribe_transient_local:=true -r __ns:=/robot_1

#3D建图(ascamera)
ros2 launch slam rtabmap_slam.launch.py

#rviz查看建图效果
ros2 launch slam rviz_rtabmap.launch.py

#键盘控制(可选)
ros2 launch peripherals teleop_key_control.launch.py


#2D导航
##rviz发布导航目标
ros2 launch navigation rviz_navigation.launch.py
ros2 launch navigation navigation.launch.py map:=地图名称

#3D导航(Dabai)
ros2 launch navigation rtabmap_navigation.launch.py

#rviz发布导航目标
ros2 launch navigation rviz_rtabmap_navigation.launch.py

#######simulations#######
#urdf可视化
ros2 launch jetrover_description ack.launch.py

########使用上位机software在关闭APP自启范围的情况下需要先打开摄像头服务
ros2 launch peripherals depth_camera.launch.py

#lab_tool
python3 ~/software/lab_tool/main.py

#servo_tool（调整pwm舵机偏差）
python3 ~/software/servo_tool/main.py
