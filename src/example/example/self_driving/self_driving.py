#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/28
# @author:aiden
# autonomous driving
import os
import cv2
import math
import time
import queue
import rclpy
import threading
import numpy as np
import sdk.pid as pid
import sdk.fps as fps
from rclpy.node import Node
import sdk.common as common
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from interfaces.msg import ObjectsInfo
from std_srvs.srv import SetBool, Trigger
from sdk.common import colors, plot_one_box
from example.self_driving import lane_detect
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from ros_robot_controller_msgs.msg import BuzzerState, SetPWMServoState, PWMServoState


class SelfDrivingNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.is_running = True
        # P: 비례, I: 정상상태 오차 보정, D: 급격한 변화 억제
        self.pid = pid.PID(0.3, 0.01, 0.09)
        self.param_init()

        self.fps = fps.FPS()
        self.image_queue = queue.Queue(maxsize=2)
        self.classes = ['go', 'right', 'park', 'red', 'green', 'crosswalk']
        self.display = True
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.colors = common.Colors()
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.lane_detect = lane_detect.LaneDetector("yellow")

        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.servo_state_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)

        self.create_service(Trigger, '~/enter', self.enter_srv_callback)   # enter the game
        self.create_service(Trigger, '~/exit', self.exit_srv_callback)     # exit the game
        self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)

        timer_cb_group = ReentrantCallbackGroup()
        self.client = self.create_client(Trigger, '/yolov5_ros2/init_finish')
        self.client.wait_for_service()
        self.start_yolov5_client = self.create_client(Trigger, '/yolov5/start', callback_group=timer_cb_group)
        self.start_yolov5_client.wait_for_service()
        self.stop_yolov5_client = self.create_client(Trigger, '/yolov5/stop', callback_group=timer_cb_group)
        self.stop_yolov5_client.wait_for_service()

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def init_process(self):
        self.timer.cancel()

        self.mecanum_pub.publish(Twist())
        time.sleep(1)

        if 1:  # self.get_parameter('start').value:
            self.display = True
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())
            request = SetBool.Request()
            request.data = True
            self.set_running_srv_callback(request, SetBool.Response())

        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def param_init(self):
        self.start = False
        self.enter = False
        self.right = True

        self.have_turn_right = False
        self.detect_turn_right = False
        self.detect_far_lane = False
        self.park_x = -1  # 주차 표지 x 픽셀 좌표 (없으면 -1)

        self.start_turn_time_stamp = 0
        self.count_turn = 0
        self.start_turn = False

        self.count_right = 0
        self.count_right_miss = 0
        self.turn_right = False
        self.right_turn_time = 0

        self.last_park_detect = False
        self.count_park = 0
        self.stop = False
        self.start_park = False

        self.count_crosswalk = 0
        self.crosswalk_distance = 0  # 횡단보도 중심 y픽셀 (클수록 가까움)

        self.traffic_signs_status = None  # 신호등 상태 기록
        self.red_loss_count = 0

        self.object_sub = None
        self.image_sub = None
        self.objects_info = []

        # ===== 횡단보도/신호등 상수 (직접 조정) =====
        self.CROSSWALK_STOP_Y = 150        # 횡단보도 중심 y픽셀이 이 값보다 크면(=가까우면) 정지
        self.NO_LIGHT_TIMEOUT = 3.0        # 신호등을 한 번도 못 봤을 때 통과까지 대기(초)
        self.RED_LOSS_TOLERANCE = 5        # 신호등 깜빡임 허용 프레임 수
        self.CROSSWALK_GONE_CONFIRM = 5    # 횡단보도가 완전히 사라졌다고 볼 연속 프레임 수
        self.PARK_TRIGGER_Y = 220          # 주차 트리거용 횡단보도 y픽셀 임계
        self.PARK_CONFIRM = 13             # 주차 시작 전 연속 확인 횟수

        # 횡단보도 상태머신: 'NORMAL' -> 'STOPPED' -> 'PASSED' -> 'NORMAL'
        self.crosswalk_state = 'NORMAL'
        self.stop_enter_time = 0           # STOPPED 진입 시각
        self.crosswalk_gone_count = 0      # 횡단보도가 안 보인 연속 프레임 수

        self.drive_speed = 0.3

    def get_node_state(self, request, response):
        response.success = True
        return response

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "self driving enter")
        with self.lock:
            self.start = False
            # subscription 객체를 보관해야 exit에서 정리 가능
            self.image_sub = self.create_subscription(
                Image, '/ascamera/camera_publisher/rgb0/image', self.image_callback, 1)
            self.object_sub = self.create_subscription(
                ObjectsInfo, '/yolov5_ros2/object_detect', self.get_object_callback, 1)
            self.mecanum_pub.publish(Twist())
            self.enter = True
        response.success = True
        response.message = "enter"
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "self driving exit")
        with self.lock:
            try:
                # ROS2에서는 destroy_subscription 사용 (unregister 아님)
                if self.image_sub is not None:
                    self.destroy_subscription(self.image_sub)
                if self.object_sub is not None:
                    self.destroy_subscription(self.object_sub)
            except Exception as e:
                self.get_logger().info('\033[1;32m%s\033[0m' % str(e))
            self.mecanum_pub.publish(Twist())
        self.param_init()
        response.success = True
        response.message = "exit"
        return response

    def set_running_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "set_running")
        with self.lock:
            self.start = request.data
            if not self.start:
                self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "set_running"
        return response

    def shutdown(self, signum, frame):
        self.is_running = False

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            self.image_queue.get()
        self.image_queue.put(rgb_image)

    # ---- 주차 기동 (Ackerman 기준, 개루프) ----
    def park_action(self):
        # 오른쪽으로 꺾기
        twist = Twist()
        twist.linear.x = 0.15
        twist.angular.z = twist.linear.x * math.tan(-0.5061) / 0.145
        self.mecanum_pub.publish(twist)
        time.sleep(3)

        # 왼쪽으로 꺾기
        twist = Twist()
        twist.linear.x = 0.15
        twist.angular.z = -twist.linear.x * math.tan(-0.5061) / 0.145
        self.mecanum_pub.publish(twist)
        time.sleep(2)

        # 다시 각도 맞추기
        twist = Twist()
        twist.linear.x = 0.15
        twist.angular.z = twist.linear.x * math.tan(-0.5061) / 0.145
        self.mecanum_pub.publish(twist)
        time.sleep(1.5)

        self.mecanum_pub.publish(Twist())

    def main(self):
        self.get_logger().info('\033[1;33m%s\033[0m' % "self_driving main start")

        while self.is_running:
            time_start = time.time()
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.is_running:
                    break
                else:
                    continue

            result_image = image.copy()
            if self.start:
                h, w = image.shape[:2]

                # 차선 이진화
                binary_image = self.lane_detect.get_binary(image)

                twist = Twist()
                twist.linear.x = self.drive_speed

                # ===== 횡단보도 사라짐 카운트 =====
                if self.crosswalk_distance == 0:
                    self.crosswalk_gone_count += 1
                else:
                    self.crosswalk_gone_count = 0

                # ===== 횡단보도 + 신호등 상태머신 =====
                if self.crosswalk_state == 'NORMAL':
                    # 가장 가까운 횡단보도가 정지 임계를 넘으면 정지
                    if self.crosswalk_distance > self.CROSSWALK_STOP_Y:
                        self.crosswalk_state = 'STOPPED'
                        self.stop_enter_time = time.time()
                        self.stop = True
                        self.red_loss_count = 0

                elif self.crosswalk_state == 'STOPPED':
                    self.stop = True  # 기본은 정지 유지
                    sign = self.traffic_signs_status.class_name \
                        if self.traffic_signs_status is not None else None

                    if sign == 'green':
                        # 초록불 -> 출발, 이 정지 이벤트 종료
                        self.stop = False
                        self.crosswalk_state = 'PASSED'
                    elif sign == 'red':
                        # 빨간불 -> 정지 유지, 타임아웃 리셋(빨간불 오통과 방지)
                        self.stop = True
                        self.stop_enter_time = time.time()
                        self.red_loss_count = 0
                    else:
                        # 신호등이 안 잡힘: 깜빡임인지 진짜 없는지 구분
                        self.red_loss_count += 1
                        if self.red_loss_count <= self.RED_LOSS_TOLERANCE:
                            # 깜빡임으로 간주: 정지 유지, 타임아웃 리셋
                            self.stop_enter_time = time.time()
                        else:
                            # 신호등이 정말 없음: 타임아웃 경과하면 통과
                            if time.time() - self.stop_enter_time > self.NO_LIGHT_TIMEOUT:
                                self.stop = False
                                self.crosswalk_state = 'PASSED'

                elif self.crosswalk_state == 'PASSED':
                    # 통과 중: 두 번째 횡단보도가 보여도 멈추지 않음
                    self.stop = False
                    # 횡단보도 묶음이 완전히 사라지면 다음 횡단보도 대비해 잠금 해제
                    if self.crosswalk_gone_count >= self.CROSSWALK_GONE_CONFIRM:
                        self.crosswalk_state = 'NORMAL'
                        # 다음 횡단보도를 위해 신호등 상태도 리셋
                        self.traffic_signs_status = None
                        self.red_loss_count = 0

                # ===== 주차 판정 =====
                # (주의) 현재 주차는 횡단보도가 가까울 때만 트리거됨
                if 0 < self.park_x and self.crosswalk_distance > self.PARK_TRIGGER_Y:
                    if not self.start_park:
                        self.count_park += 1
                        if self.count_park >= self.PARK_CONFIRM:
                            self.mecanum_pub.publish(Twist())
                            self.start_park = True
                            self.stop = True
                            threading.Thread(target=self.park_action, daemon=True).start()
                else:
                    self.count_park = 0

                # ===== 우회전 (개루프) =====
                skip_lane = False
                if self.turn_right:
                    if time.time() - self.right_turn_time < 1:
                        twist.angular.z = -1.0
                        self.mecanum_pub.publish(twist)
                        skip_lane = True   # 차선 추종만 건너뜀 (루프 전체 X)
                    else:
                        self.turn_right = False

                # ===== 차선 추종 (정지/우회전 중이 아닐 때만) =====
                result_image, lane_angle, lane_x = self.lane_detect(binary_image, image.copy())

                if self.stop:
                    # 정지: 멈춤 명령 + PID 누적 제거
                    self.mecanum_pub.publish(Twist())
                    self.pid.clear()
                elif skip_lane:
                    # 우회전 중: 차선 추종 건너뜀 (twist는 이미 위에서 publish)
                    pass
                elif lane_x >= 0:
                    if lane_x > 120:
                        # 급커브
                        self.count_turn += 1
                        if self.count_turn > 5 and not self.start_turn:
                            self.start_turn = True
                            self.count_turn = 0
                            self.start_turn_time_stamp = time.time()
                        if self.machine_type != 'MentorPi_Acker':
                            twist.angular.z = -0.9
                        else:
                            twist.angular.z = twist.linear.x * math.tan(-0.9) / 0.145
                    else:
                        # 직선/완만한 보정: PID
                        self.count_turn = 0
                        if time.time() - self.start_turn_time_stamp > 2 and self.start_turn:
                            self.start_turn = False
                        if not self.start_turn:
                            self.pid.SetPoint = 100
                            self.pid.update(lane_x)
                            if self.machine_type != 'MentorPi_Acker':
                                twist.angular.z = common.set_range(self.pid.output, -0.1, 0.1)
                            else:
                                twist.angular.z = twist.linear.x * math.tan(
                                    common.set_range(self.pid.output, -0.1, 0.1)) / 0.145
                        else:
                            if self.machine_type == 'MentorPi_Acker':
                                twist.angular.z = 0.15 * math.tan(-0.5061) / 0.145
                    self.mecanum_pub.publish(twist)
                else:
                    # 차선을 못 찾음: PID 누적 제거 (직진 관성 방지)
                    self.pid.clear()

                # ===== 디버그 박스 =====
                if self.objects_info:
                    for i in self.objects_info:
                        box = i.box
                        class_name = i.class_name
                        cls_conf = i.score
                        cls_id = self.classes.index(class_name)
                        color = colors(cls_id, True)
                        plot_one_box(
                            box, result_image, color=color,
                            label="{}:{:.2f}".format(class_name, cls_conf))
            else:
                time.sleep(0.01)

            bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            if self.display:
                self.fps.update()
                bgr_image = self.fps.show_fps(bgr_image)

            self.result_publisher.publish(self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))

            time_d = 0.03 - (time.time() - time_start)
            if time_d > 0:
                time.sleep(time_d)

        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

    # ---- 객체 감지 콜백 ----
    def get_object_callback(self, msg):
        self.objects_info = msg.objects
        if self.objects_info == []:
            # 아무것도 안 보이면 리셋
            self.traffic_signs_status = None
            self.crosswalk_distance = 0
            self.park_x = -1
            self.count_right_miss += 1
            if self.count_right_miss >= 3:
                self.count_right = 0
                self.count_right_miss = 0
            return

        min_distance = 0
        seen_park = False
        seen_right = False
        seen_traffic = False
        last_class = None

        for i in self.objects_info:
            class_name = i.class_name
            last_class = class_name
            center = (int((i.box[0] + i.box[2]) / 2), int((i.box[1] + i.box[3]) / 2))

            if class_name == 'crosswalk':
                # 가장 가까운(=y가 가장 큰) 횡단보도 채택
                if center[1] > min_distance:
                    min_distance = center[1]
            elif class_name == 'right':
                seen_right = True
                self.count_right += 1
                self.count_right_miss = 0
                if self.count_right >= 4:
                    self.turn_right = True
                    self.right_turn_time = time.time()
                    self.count_right = 0
            elif class_name == 'park':
                seen_park = True
                self.park_x = center[0]
            elif class_name == 'red' or class_name == 'green':
                seen_traffic = True
                self.traffic_signs_status = i

        # 이 프레임에 안 보인 표지는 정리
        if not seen_park:
            self.park_x = -1
        if not seen_traffic:
            self.traffic_signs_status = None
        if not seen_right:
            self.count_right_miss += 1
            if self.count_right_miss >= 3:
                self.count_right = 0
                self.count_right_miss = 0

        self.crosswalk_distance = min_distance


def main():
    node = SelfDrivingNode('self_driving')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()


if __name__ == "__main__":
    main()