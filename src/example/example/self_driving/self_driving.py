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

from gpiozero import LED
from ros_robot_controller_msgs.msg import RGBStates, RGBState


class SelfDrivingNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(
            name,
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.name = name
        self.is_running = True
        # P: 비례, I: 정상상태 오차 보정, D: 급격한 변화 억제
        self.pid = pid.PID(0.3, 0.01, 0.09)
        self.param_init()

        self.fps = fps.FPS()
        self.image_queue = queue.Queue(maxsize=2)
        self.classes = ["go", "right", "park", "red", "green", "crosswalk"]
        self.display = True
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.colors = common.Colors()
        self.machine_type = os.environ.get("MACHINE_TYPE")
        self.lane_detect = lane_detect.LaneDetector("yellow")

        self.mecanum_pub = self.create_publisher(Twist, "/controller/cmd_vel", 1)
        self.servo_state_pub = self.create_publisher(
            SetPWMServoState, "ros_robot_controller/pwm_servo/set_state", 1
        )
        self.result_publisher = self.create_publisher(Image, "~/image_result", 1)

        self.create_service(Trigger, "~/enter", self.enter_srv_callback)
        self.create_service(Trigger, "~/exit", self.exit_srv_callback)
        self.create_service(SetBool, "~/set_running", self.set_running_srv_callback)

        timer_cb_group = ReentrantCallbackGroup()
        self.client = self.create_client(Trigger, "/yolov5_ros2/init_finish")
        self.client.wait_for_service()
        self.start_yolov5_client = self.create_client(
            Trigger, "/yolov5/start", callback_group=timer_cb_group
        )
        self.start_yolov5_client.wait_for_service()
        self.stop_yolov5_client = self.create_client(
            Trigger, "/yolov5/stop", callback_group=timer_cb_group
        )
        self.stop_yolov5_client.wait_for_service()

        self.timer = self.create_timer(
            0.0, self.init_process, callback_group=timer_cb_group
        )

        self.rgb_pub = self.create_publisher(
            RGBStates, "/ros_robot_controller/set_rgb", 10
        )

    def init_process(self):
        self.timer.cancel()

        self.mecanum_pub.publish(Twist())
        time.sleep(1)

        if 1:
            self.display = True
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())

            # 처음엔 시작 안함
            self.start = False

        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, "~/init_finish", self.get_node_state)
        self.get_logger().info("\033[1;32m%s\033[0m" % "start")

    def param_init(self):
        self.start = False
        self.enter = False
        self.right = True

        self.have_turn_right = False
        self.detect_turn_right = False
        self.detect_far_lane = False
        self.park_x = -1  # 주차 표지 x 픽셀 좌표 (없으면 -1)
        self.park_area = 0  # 주차 표지 bbox 면적

        self.start_turn_time_stamp = 0
        self.count_turn = 0
        self.start_turn = False
        self.MIN_STOP_TIME = 1.0   # 횡단보도에서 최소 정지 시간(초)
        # ===== 우회전 (단순화: 크기 조건 만족한 표지를 한 번이라도 보면 기억) =====
        self.right_seen = False  # 우회전 표지를 (크기 조건 만족하며) 본 적 있음
        self.turn_right = False  # 실제 회전 중
        self.right_turn_time = 0
        self.right_sign_y = 0
        self.right_sign_center_x = -1
        self.right_sign_area = 0

        self.last_park_detect = False
        self.count_park = 0
        self.stop = False
        self.start_park = False
        self.park_phase = None
        self.park_phase_start_time = 0.0
        self.park_completed = False

        self.count_crosswalk = 0

        self.traffic_signs_status = None  # 신호등 상태 기록
        self.red_loss_count = 0

        self.object_sub = None
        self.image_sub = None
        self.objects_info = []

        # ===== 횡단보도/신호등 상수 (직접 조정) =====
        self.CROSSWALK_GAP = (
            60  # 박스 y중심이 이 픽셀 이상 떨어지면 "별개 횡단보도"로 봄
        )
        self.CROSSWALK_STOP_COUNT = 2  # 별개 횡단보도가 이 개수 이상이면 정지 절차 시작
        self.APPROACH_DISTANCE = 0.07  # 2개 판단 후 더 전진할 거리(m)
        self.NO_LIGHT_TIMEOUT = 3.0  # 신호등을 한 번도 못 봤을 때 통과까지 대기(초)
        self.RED_LOSS_TOLERANCE = 5  # 신호등 깜빡임 허용 프레임 수
        self.CROSSWALK_GONE_CONFIRM = (
            5  # 횡단보도가 완전히 사라졌다고 볼 연속 프레임 수
        )
        self.PARK_CONFIRM = 1  # 주차 시작 전 연속 확인 횟수

        # ===== 우회전 크기 판정 상수 (느슨한 쪽 사용) =====
        self.RIGHT_TRIGGER_Y = 60  # 박스 하단 y가 이 값 이상
        self.RIGHT_TRIGGER_AREA = 100  # 박스 면적이 이 값 이상
        self.RIGHT_TURN_DURATION = 1.0  # 우회전 기동 시간(초), 개루프
        self.RIGHT_APPROACH_DISTANCE = 1.4  # 회전 전 전진 거리(m), 실측으로 조정
        self.right_approaching = False  # 회전 전 전진 중
        self.right_approach_start = 0
        # 횡단보도 상태머신: 'NORMAL' -> 'APPROACH' -> 'STOPPED' -> 'PASSED' -> 'NORMAL'
        self.crosswalk_state = "NORMAL"
        self.approach_enter_time = 0
        self.stop_enter_time = 0
        self.crosswalk_count = 0
        self.crosswalk_gone_count = 0

        self.drive_speed = 0.3

        # 빵판 led에 gpio 할당 및 초기 소등 처리
        self.blue_led = LED(23)
        self.red_led = LED(24)
        self.right_yellow_led = LED(18)
        self.left_yellow_led = LED(15)
        self.blue_led.off()
        self.red_led.off()
        self.right_yellow_led.off()
        self.left_yellow_led.off()

        self.wait_for_green = True   # 시작 전 신호 대기

    def get_node_state(self, request, response):
        response.success = True
        return response

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def enter_srv_callback(self, request, response):
        self.get_logger().info("\033[1;32m%s\033[0m" % "self driving enter")
        with self.lock:
            self.start = False
            self.image_sub = self.create_subscription(
                Image, "/ascamera/camera_publisher/rgb0/image", self.image_callback, 1
            )
            self.object_sub = self.create_subscription(
                ObjectsInfo, "/yolov5_ros2/object_detect", self.get_object_callback, 1
            )
            self.mecanum_pub.publish(Twist())
            self.enter = True
        response.success = True
        response.message = "enter"
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info("\033[1;32m%s\033[0m" % "self driving exit")
        with self.lock:
            try:
                if self.image_sub is not None:
                    self.destroy_subscription(self.image_sub)
                if self.object_sub is not None:
                    self.destroy_subscription(self.object_sub)
            except Exception as e:
                self.get_logger().info("\033[1;32m%s\033[0m" % str(e))
            self.mecanum_pub.publish(Twist())
        self.param_init()
        response.success = True
        response.message = "exit"
        return response

    def set_running_srv_callback(self, request, response):
        self.get_logger().info("\033[1;32m%s\033[0m" % "set_running")
        with self.lock:
            self.start = request.data
            if not self.start:
                self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "set_running"
        if self.start:
            self.led_control("move")
        return response

    def shutdown(self, signum, frame):
        self.is_running = False

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            self.image_queue.get()
        self.image_queue.put(rgb_image)

    def get_right_box_metrics(self, box):
        x1, y1, x2, y2 = box
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        return {
            "center_x": int((x1 + x2) / 2),
            "bottom_y": int(max(y1, y2)),
            "height": int(height),
            "area": int(width * height),
        }

    def start_right_turn(self):
        self.turn_right = True
        self.right_turn_time = time.time()
        self.get_logger().info(
            "right turn start: x=%d y=%d area=%d"
            % (self.right_sign_center_x, self.right_sign_y, self.right_sign_area)
        )

    # ---- 횡단보도 박스들의 y중심으로 "별개 횡단보도" 개수 세기 ----
    @staticmethod
    def count_distinct_crosswalks(y_centers, gap):
        if not y_centers:
            return 0
        ys = sorted(y_centers)
        groups = 1
        for prev, cur in zip(ys, ys[1:]):
            if cur - prev > gap:
                groups += 1
        return groups

    # ---- 주차 기동 (메인 루프에서 순차 실행) ----
    def park_action(self):
        self.get_logger().info("PARK ACTION 시작")
        self.start_park = True
        self.stop = True
        self.count_park = 0

        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = -0.2
        self.mecanum_pub.publish(twist)
        time.sleep(2.0)

        twist.linear.y = 0.0
        self.mecanum_pub.publish(twist)
        time.sleep(1.0)

        self.led_blink()
        self.get_logger().info("PARK ACTION 종료")
        return True

    def led_control(self, state):
        match state:
            case "move":
                self.red_led.off()
                self.blue_led.on()
                self.right_yellow_led.off()
                self.left_yellow_led.off()

                # self.set_rgb([[1, 0, 0, 255], [2, 0, 0, 255]])  # 기준 수정

            case "stop":
                self.red_led.on()
                self.blue_led.off()
                self.left_yellow_led.off()

                # self.set_rgb([[1, 255, 0, 0], [2, 255, 0, 0]])  # 기준 수정
            case "turn_start":
                for _ in range(3):
                    self.red_led.off()
                    self.blue_led.off()
                    self.right_yellow_led.on()
                    self.left_yellow_led.off()

                    self.set_rgb([[1, 0, 0, 0], [2, 255, 255, 0]])
                    time.sleep(0.1)

                    self.red_led.off()
                    self.blue_led.off()
                    self.right_yellow_led.off()
                    self.left_yellow_led.off()

                    self.set_rgb([[1, 0, 0, 0], [2, 0, 0, 0]])
                    time.sleep(0.1)  # 기준 수정
            case "turn_end":
                self.red_led.off()
                self.blue_led.on()
                self.right_yellow_led.off()
                self.left_yellow_led.off()

                # self.set_rgb([[1, 0, 0, 255], [2, 0, 0, 255]])  # 기준 수정

    def led_blink(self):
        for _ in range(2):
            self.red_led.off()
            self.blue_led.off()
            self.right_yellow_led.off()
            self.left_yellow_led.off()
            self.set_rgb([[1, 0, 0, 0], [2, 0, 0, 0]])
            time.sleep(0.2)
            self.red_led.on()
            self.blue_led.on()
            self.right_yellow_led.on()
            self.left_yellow_led.on()
            self.set_rgb([[1, 255, 255, 0], [2, 255, 255, 0]])
            time.sleep(0.2)
        self.red_led.off()
        self.blue_led.off()
        self.right_yellow_led.off()
        self.left_yellow_led.off()
        self.set_rgb([[1, 0, 0, 0], [2, 0, 0, 0]])  # 기준 수정

    def set_rgb(self, pixels):
        msg = RGBStates()
        msg.states = [
            RGBState(index=idx, red=r, green=g, blue=b) for idx, r, g, b in pixels
        ]
        self.rgb_pub.publish(msg)

    def main(self):
        self.get_logger().info("\033[1;33m%s\033[0m" % "self_driving main start")

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

                binary_image = self.lane_detect.get_binary(image)

                twist = Twist()
                twist.linear.x = self.drive_speed

                # ===== 횡단보도 사라짐 카운트 (개수 기반) =====
                if self.crosswalk_count == 0:
                    self.crosswalk_gone_count += 1
                else:
                    self.crosswalk_gone_count = 0

                # ===== 횡단보도 + 신호등 상태머신 =====
                if self.crosswalk_state == "NORMAL":
                    if self.crosswalk_count >= self.CROSSWALK_STOP_COUNT:
                        self.crosswalk_state = "APPROACH"
                        self.approach_enter_time = time.time()
                        self.stop = False
                        self.led_control("move")

                elif self.crosswalk_state == "APPROACH":
                    approach_time = self.APPROACH_DISTANCE / self.drive_speed
                    if time.time() - self.approach_enter_time < approach_time:
                        self.stop = False
                        self.led_control("move")
                    else:
                        self.crosswalk_state = "STOPPED"
                        self.stop_enter_time = time.time()
                        self.stop = True
                        self.led_control("stop")
                        self.red_loss_count = 0

                elif self.crosswalk_state == "STOPPED":
                    self.stop = True
                    self.led_control("stop")
                    sign = (
                        self.traffic_signs_status.class_name
                        if self.traffic_signs_status is not None
                        else None
                    )
                    # 최소 정지 시간이 지났는지
                    stopped_long_enough = (time.time() - self.stop_enter_time) >= self.MIN_STOP_TIME

                    if sign == "green":
                        # 초록불이어도 최소 정지 시간은 채운 뒤 출발
                        if stopped_long_enough:
                            self.stop = False
                            self.led_control("move")
                            self.crosswalk_state = "PASSED"
                        # 아직 1초 안 됐으면 계속 정지 (다음 프레임에 다시 확인)
                    elif sign == "red":
                        self.stop = True
                        self.led_control("stop")
                        self.stop_enter_time = time.time()   # 빨간불은 타임아웃 리셋
                        self.red_loss_count = 0
                    else:
                        self.red_loss_count += 1
                        if self.red_loss_count <= self.RED_LOSS_TOLERANCE:
                            self.stop_enter_time = time.time()
                        else:
                            if time.time() - self.stop_enter_time > self.NO_LIGHT_TIMEOUT:
                                self.stop = False
                                self.led_control("move")
                                self.crosswalk_state = "PASSED"

                elif self.crosswalk_state == "PASSED":
                    self.stop = False
                    self.led_control("move")
                    if self.crosswalk_gone_count >= self.CROSSWALK_GONE_CONFIRM:
                        self.crosswalk_state = "NORMAL"
                        self.traffic_signs_status = None
                        self.red_loss_count = 0

                # ===== 주차 판정 =====
                if 0 < self.park_x and self.park_area > 1800:
                    self.count_park += 1
                    if self.count_park >= self.PARK_CONFIRM:
                        self.mecanum_pub.publish(Twist())
                        self.start_park = True
                        self.stop = True
                        self.park_action()
                        self.start_park = False
                        time.sleep(1)
                        self.is_running = False
                        break
                else:
                    self.count_park = 0

                # ===== 우회전 (단순화) =====
                # 우회전 표지를 (크기 조건 만족하며) 한 번이라도 봤고(right_seen),
                # 횡단보도 정지가 풀리면(stop=False) 그때 회전 시작.
                skip_lane = False
                # 정지가 풀렸고 우회전 표지를 봤으면 → 먼저 전진 단계 시작
                if (
                    self.right_seen
                    and not self.stop
                    and not self.turn_right
                    and not self.right_approaching
                ):
                    self.right_approaching = True
                    self.right_approach_start = time.time()

                # 전진 단계: 정해진 거리만큼 차선 추종으로 전진 후 회전 시작
                if self.right_approaching:
                    self.get_logger().info("approaching=%s turn_right=%s lane_x=%s stop=%s" % (self.right_approaching, self.turn_right, str(lane_x), self.stop))
                    approach_time = self.RIGHT_APPROACH_DISTANCE / self.drive_speed
                    if time.time() - self.right_approach_start >= approach_time:
                        self.right_approaching = False
                        self.start_right_turn()
                    # 전진 중에는 skip_lane을 걸지 않음 → 아래 차선 추종이 전진 담당

                if self.turn_right:
                    if time.time() - self.right_turn_time < self.RIGHT_TURN_DURATION:
                        twist.angular.z = -1.0
                        self.mecanum_pub.publish(twist)
                        skip_lane = True
                    else:
                        self.turn_right = False
                        self.right_seen = False
                        self.right_approaching = False
                        self.led_control("turn_end")

                self.get_logger().info(
                    "turn_right=%s right_seen=%s" % (self.turn_right, self.right_seen)
                )

                # ===== 차선 추종 (정지/우회전 중이 아닐 때만) =====
                result_image, lane_angle, lane_x = self.lane_detect(
                    binary_image, image.copy()
                )

                if skip_lane:
                    self.get_logger().info("SKIP LANE")
                    pass
                elif self.stop:
                    self.mecanum_pub.publish(Twist())
                    self.pid.clear()
                elif lane_x >= 0:
                    if lane_x > 120:
                        self.count_turn += 1
                        if self.count_turn > 8 and not self.start_turn:
                            self.get_logger().info("LANE DETECT - TURN RIGHT")
                            self.start_turn = True
                            self.led_control("turn_start")
                            self.count_turn = 0
                            self.start_turn_time_stamp = time.time()
                        if self.machine_type != "MentorPi_Acker":
                            twist.angular.z = -0.9

                    else:
                        self.count_turn = 0
                        if (
                            time.time() - self.start_turn_time_stamp > 2
                            and self.start_turn
                        ):
                            self.start_turn = False
                            self.led_control("turn_end")
                        if not self.start_turn:
                            self.pid.SetPoint = 100
                            self.pid.update(lane_x)
                            if self.machine_type != "MentorPi_Acker":
                                twist.angular.z = common.set_range(
                                    self.pid.output, -0.1, 0.1
                                )

                        else:
                            if self.machine_type == "MentorPi_Acker":
                                twist.angular.z = 0.15 * math.tan(-0.5061) / 0.145
                    self.mecanum_pub.publish(twist)
                else:
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
                            box,
                            result_image,
                            color=color,
                            label="{}:{:.2f}".format(class_name, cls_conf),
                        )
            else:
                self.mecanum_pub.publish(Twist())   # 계속 정지 명령
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
            self.traffic_signs_status = None
            self.crosswalk_count = 0
            self.park_x = -1
            self.park_area = 0
            return

        crosswalk_ys = []
        seen_park = False
        seen_traffic = False

        for i in self.objects_info:
            class_name = i.class_name
            center = (int((i.box[0] + i.box[2]) / 2), int((i.box[1] + i.box[3]) / 2))

            if class_name == "crosswalk":
                crosswalk_ys.append(center[1])
                self.get_logger().info("CROSSWALK 탐지")
            elif class_name == "right":
                self.get_logger().info("RIGHT 탐지")
                metrics = self.get_right_box_metrics(i.box)
                # 크기 조건 유지: 충분히 크고 가까운 표지만 인정, 한 번 보면 계속 기억
                if (
                    metrics["bottom_y"] >= self.RIGHT_TRIGGER_Y
                    and metrics["area"] >= self.RIGHT_TRIGGER_AREA
                ):
                    self.right_seen = True
                    self.right_sign_center_x = metrics["center_x"]
                    self.right_sign_y = metrics["bottom_y"]
                    self.right_sign_area = metrics["area"]
            elif class_name == "park":
                self.get_logger().info("PARK 탐지")
                seen_park = True
                self.park_x = center[0]
                self.park_area = int(abs((i.box[2] - i.box[0]) * (i.box[3] - i.box[1])))
            elif class_name == "red" or class_name == "green":
                self.get_logger().info("신호등 탐지")
                seen_traffic = True
                self.traffic_signs_status = i

                # 출발 시작
                if self.wait_for_green:
                    if class_name == "red":
                        self.get_logger().info("RED - 대기중")

                    elif class_name == "green":
                        self.get_logger().info("GREEN - 주행 시작")

                        self.wait_for_green = False
                        self.start = True
                        self.led_control("move")

        # 횡단보도 개수는 신호등 유무와 무관하게 항상 계산 (루프 밖)
        self.crosswalk_count = self.count_distinct_crosswalks(
            crosswalk_ys, self.CROSSWALK_GAP
        )

        if crosswalk_ys:
            self.get_logger().info(
                "crosswalk distinct=%d ys=%s"
                % (self.crosswalk_count, str(sorted(crosswalk_ys)))
            )
        if not seen_traffic:
            self.traffic_signs_status = None


def main():
    node = SelfDrivingNode("self_driving")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()


if __name__ == "__main__":
    main()
