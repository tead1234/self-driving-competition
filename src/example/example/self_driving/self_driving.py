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
        self.park_area = 0  # 주차 표지 bbox 면적

        self.start_turn_time_stamp = 0
        self.count_turn = 0
        self.start_turn = False

        self.count_right = 0
        self.count_right_miss = 0
        self.turn_right = False
        self.right_turn_time = 0
        self.right_sign_y = 0
        self.right_sign_center_x = -1
        self.right_sign_area = 0
        self.right_turn_state = 'IDLE'
        self.right_seen_close = False
        self.right_ready_seen = False
        self.right_lost_count = 0
        self.right_pending = False        # 회전 준비 완료, 횡단보도 정지 해제를 기다리는 중

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
        self.CROSSWALK_GAP = 60            # 박스 y중심이 이 픽셀 이상 떨어지면 "별개 횡단보도"로 봄 (실측 후 튜닝)
        self.CROSSWALK_STOP_COUNT = 2      # 별개 횡단보도가 이 개수 이상이면 정지 절차 시작
        self.APPROACH_DISTANCE = 0.2       # 2개 판단 후 더 전진할 거리(m) ≈ 20cm
        self.NO_LIGHT_TIMEOUT = 3.0        # 신호등을 한 번도 못 봤을 때 통과까지 대기(초)
        self.RED_LOSS_TOLERANCE = 5        # 신호등 깜빡임 허용 프레임 수
        self.CROSSWALK_GONE_CONFIRM = 5    # 횡단보도가 완전히 사라졌다고 볼 연속 프레임 수
        self.PARK_CONFIRM = 5              # 주차 시작 전 연속 확인 횟수

        # ===== 우회전 anchor 상수 (카메라 bbox 기반) =====
        self.RIGHT_APPROACH_Y = 160         # 이 값부터 표지판을 가까운 anchor로 추적
        self.RIGHT_TURN_TRIGGER_Y = 210     # 박스 하단 y가 이 값 이상이면 회전 준비 완료
        self.RIGHT_MIN_HEIGHT = 16          # 너무 작은 원거리/오검출 bbox 제외
        self.RIGHT_MIN_AREA = 350           # APPROACH 진입 최소 bbox 면적
        self.RIGHT_TURN_MIN_AREA = 700      # 회전 준비 최소 bbox 면적
        self.RIGHT_CENTER_EXIT_X = 260      # 표지판 중심이 우측 경계에 닿으면 회전 준비 완료
        self.RIGHT_CONFIRM = 3              # 회전 준비 조건 연속 확인 횟수
        self.RIGHT_PASS_LOST_CONFIRM = 3    # 회전 준비 후 표지판이 사라진 연속 프레임 수
        self.RIGHT_COOLDOWN_LOST_CONFIRM = 5
        self.RIGHT_TURN_DURATION = 1.0      # 실제 우회전 기동 시간(초), 개루프

        # 횡단보도 상태머신: 'NORMAL' -> 'APPROACH' -> 'STOPPED' -> 'PASSED' -> 'NORMAL'
        self.crosswalk_state = 'NORMAL'
        self.approach_enter_time = 0       # APPROACH 진입 시각
        self.stop_enter_time = 0           # STOPPED 진입 시각
        self.crosswalk_count = 0           # 현재 프레임의 "별개 횡단보도" 개수
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

    def get_right_box_metrics(self, box):
        x1, y1, x2, y2 = box
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        return {
            'center_x': int((x1 + x2) / 2),
            'bottom_y': int(max(y1, y2)),
            'height': int(height),
            'area': int(width * height),
        }

    def start_right_turn(self):
        self.turn_right = True
        self.right_turn_time = time.time()
        self.right_turn_state = 'TURNING'
        self.right_pending = False
        self.count_right = 0
        self.count_right_miss = 0
        self.right_lost_count = 0
        self.get_logger().info(
            'right turn start: x=%d y=%d area=%d' % (
                self.right_sign_center_x, self.right_sign_y, self.right_sign_area))

    # STOPPING 단계 제거: 회전 준비가 끝나면 즉시 대기 상태로 전환.
    # 실제 회전 시작은 main에서 횡단보도 정지가 풀린(stop=False) 순간에 이뤄진다.
    def prepare_right_turn(self):
        if self.right_turn_state in ('TURNING', 'COOLDOWN') or self.right_pending:
            return
        self.right_pending = True     # 준비 완료, 횡단보도 정지 해제를 기다림
        self.count_right = 0
        self.count_right_miss = 0
        self.right_lost_count = 0
        self.get_logger().info(
            'right turn ready (pending): x=%d y=%d area=%d' % (
                self.right_sign_center_x, self.right_sign_y, self.right_sign_area))

    def reset_right_anchor(self):
        self.right_turn_state = 'IDLE'
        self.right_seen_close = False
        self.right_ready_seen = False
        self.right_lost_count = 0
        self.right_pending = False
        self.count_right = 0
        self.right_sign_y = 0
        self.right_sign_center_x = -1
        self.right_sign_area = 0

    def update_right_turn_anchor(self, right_metrics):
        if self.right_turn_state == 'TURNING':
            return

        if right_metrics is None:
            self.right_sign_y = 0
            self.right_sign_center_x = -1
            self.right_sign_area = 0
            self.count_right_miss += 1

            if self.right_turn_state == 'APPROACH':
                self.right_lost_count += 1
                if self.right_ready_seen and self.right_lost_count >= self.RIGHT_PASS_LOST_CONFIRM:
                    self.prepare_right_turn()
                    return
                if self.right_lost_count >= self.RIGHT_PASS_LOST_CONFIRM:
                    self.reset_right_anchor()
                    return

            if self.right_turn_state == 'COOLDOWN':
                if self.count_right_miss >= self.RIGHT_COOLDOWN_LOST_CONFIRM:
                    self.reset_right_anchor()
                    self.count_right_miss = 0
                return

            if self.count_right_miss >= 3:
                self.count_right = 0
                self.count_right_miss = 0
            return

        self.right_sign_y = right_metrics['bottom_y']
        self.right_sign_center_x = right_metrics['center_x']
        self.right_sign_area = right_metrics['area']
        self.count_right_miss = 0
        self.right_lost_count = 0

        if self.right_turn_state == 'COOLDOWN':
            return

        close_enough = (
            right_metrics['bottom_y'] >= self.RIGHT_APPROACH_Y and
            right_metrics['height'] >= self.RIGHT_MIN_HEIGHT and
            right_metrics['area'] >= self.RIGHT_MIN_AREA
        ) or right_metrics['center_x'] >= self.RIGHT_CENTER_EXIT_X
        ready_to_turn = (
            right_metrics['bottom_y'] >= self.RIGHT_TURN_TRIGGER_Y and
            right_metrics['area'] >= self.RIGHT_TURN_MIN_AREA
        ) or right_metrics['center_x'] >= self.RIGHT_CENTER_EXIT_X

        if not close_enough:
            if self.right_turn_state != 'APPROACH':
                self.count_right = 0
            return

        self.right_seen_close = True
        if ready_to_turn:
            self.right_ready_seen = True
        if self.right_turn_state == 'IDLE':
            self.right_turn_state = 'APPROACH'
            self.count_right = 1
            return

        if self.right_turn_state == 'APPROACH':
            self.count_right += 1
            if self.count_right >= self.RIGHT_CONFIRM:
                self.prepare_right_turn()

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
    def start_parking_sequence(self):
        self.park_phase = 0
        self.park_phase_start_time = time.time()
        self.park_completed = False
        self.start_park = True
        self.stop = True
        self.count_park = 0
        self.get_logger().info('\033[1;31m%s\033[0m' % 'PARK ACTION START')

    def update_parking_sequence(self, twist):
        if not self.start_park or self.park_completed:
            return False

        if self.park_phase is None:
            self.mecanum_pub.publish(Twist())
            self.park_completed = True
            return False

        durations = [3.0, 2.0, 1.5]
        if time.time() - self.park_phase_start_time >= durations[self.park_phase]:
            self.park_phase += 1
            self.park_phase_start_time = time.time()
            if self.park_phase >= len(durations):
                self.mecanum_pub.publish(Twist())
                self.park_completed = True
                return False

        twist.linear.x = 0
        twist.linear.y = 0.2
        self.mecanum_pub.publish(twist)
        return True

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

                binary_image = self.lane_detect.get_binary(image)

                twist = Twist()
                twist.linear.x = self.drive_speed

                # ===== 횡단보도 사라짐 카운트 (개수 기반) =====
                if self.crosswalk_count == 0:
                    self.crosswalk_gone_count += 1
                else:
                    self.crosswalk_gone_count = 0

                # ===== 횡단보도 + 신호등 상태머신 =====
                if self.crosswalk_state == 'NORMAL':
                    if self.crosswalk_count >= self.CROSSWALK_STOP_COUNT:
                        self.crosswalk_state = 'APPROACH'
                        self.approach_enter_time = time.time()
                        self.stop = False

                elif self.crosswalk_state == 'APPROACH':
                    approach_time = self.APPROACH_DISTANCE / self.drive_speed
                    if time.time() - self.approach_enter_time < approach_time:
                        self.stop = False
                    else:
                        self.crosswalk_state = 'STOPPED'
                        self.stop_enter_time = time.time()
                        self.stop = True
                        self.red_loss_count = 0

                elif self.crosswalk_state == 'STOPPED':
                    self.stop = True
                    sign = self.traffic_signs_status.class_name \
                        if self.traffic_signs_status is not None else None

                    if sign == 'green':
                        self.stop = False
                        self.crosswalk_state = 'PASSED'
                    elif sign == 'red':
                        self.stop = True
                        self.stop_enter_time = time.time()
                        self.red_loss_count = 0
                    else:
                        self.red_loss_count += 1
                        if self.red_loss_count <= self.RED_LOSS_TOLERANCE:
                            self.stop_enter_time = time.time()
                        else:
                            if time.time() - self.stop_enter_time > self.NO_LIGHT_TIMEOUT:
                                self.stop = False
                                self.crosswalk_state = 'PASSED'

                elif self.crosswalk_state == 'PASSED':
                    self.stop = False
                    if self.crosswalk_gone_count >= self.CROSSWALK_GONE_CONFIRM:
                        self.crosswalk_state = 'NORMAL'
                        self.traffic_signs_status = None
                        self.red_loss_count = 0

                # ===== 주차 판정 =====
                if 0 < self.park_x and self.park_area >= 1500:
                    if not self.start_park and not self.park_completed:
                        self.count_park += 1
                        if self.count_park >= self.PARK_CONFIRM:
                            self.start_parking_sequence()
                        else:
                            self.get_logger().info(
                                'park detected: count=%d/%d x=%d area=%d' % (
                                    self.count_park, self.PARK_CONFIRM, self.park_x, self.park_area))
                else:
                    if self.count_park > 0:
                        self.get_logger().info(
                            'park lost: count reset from %d to 0' % self.count_park)
                    self.count_park = 0

                # ===== 주차 기동 =====
                skip_lane = False
                if self.start_park and not self.park_completed:
                    self.stop = True
                    skip_lane = self.update_parking_sequence(twist)

                # ===== 우회전 (bbox anchor + 개루프 회전) =====
                # 회전 준비(right_pending)가 됐고 횡단보도 정지가 풀리면(stop=False) 그때 회전 시작.
                # STOPPING 단계 없음: 횡단보도에서 이미 한 번 정지했으므로 추가 정지 안 함.
                if not skip_lane and self.right_pending and not self.stop and not self.turn_right:
                    self.start_right_turn()

                if self.turn_right:
                    if time.time() - self.right_turn_time < self.RIGHT_TURN_DURATION:
                        twist.angular.z = -1.0
                        self.mecanum_pub.publish(twist)
                        skip_lane = True
                    else:
                        self.turn_right = False
                        self.right_turn_state = 'COOLDOWN'
                        self.right_lost_count = 0

                self.get_logger().info(
                    'state=%s stop=%s turn_right=%s pending=%s skip=%s cw_count=%d gone=%d '
                    'right_state=%s right_x=%d right_y=%d right_area=%d' % (
                        self.crosswalk_state, self.stop, self.turn_right, self.right_pending,
                        skip_lane, self.crosswalk_count, self.crosswalk_gone_count,
                        self.right_turn_state, self.right_sign_center_x,
                        self.right_sign_y, self.right_sign_area))

                # ===== 차선 추종 (정지/우회전 중이 아닐 때만) =====
                result_image, lane_angle, lane_x = self.lane_detect(binary_image, image.copy())

                if skip_lane:
                    # 우회전 중: 회전 명령을 정지 명령으로 덮지 않음
                    pass
                elif self.stop:
                    self.mecanum_pub.publish(Twist())
                    self.pid.clear()
                elif lane_x >= 0:
                    if lane_x > 120:
                        self.count_turn += 1
                        if self.count_turn > 8 and not self.start_turn:
                            self.start_turn = True
                            self.count_turn = 0
                            self.start_turn_time_stamp = time.time()
                        if self.machine_type != 'MentorPi_Acker':
                            twist.angular.z = -0.9
                        else:
                            twist.angular.z = twist.linear.x * math.tan(-0.9) / 0.145
                    else:
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
            self.traffic_signs_status = None
            self.crosswalk_count = 0
            self.park_x = -1
            self.park_area = 0
            self.update_right_turn_anchor(None)
            return

        crosswalk_ys = []
        seen_park = False
        seen_traffic = False
        right_metrics = None

        for i in self.objects_info:
            class_name = i.class_name
            center = (int((i.box[0] + i.box[2]) / 2), int((i.box[1] + i.box[3]) / 2))

            if class_name == 'crosswalk':
                crosswalk_ys.append(center[1])
            elif class_name == 'right':
                metrics = self.get_right_box_metrics(i.box)
                if right_metrics is None or metrics['area'] > right_metrics['area']:
                    right_metrics = metrics
            elif class_name == 'park':
                seen_park = True
                self.park_x = center[0]
                self.park_area = int(abs((i.box[2] - i.box[0]) * (i.box[3] - i.box[1])))
                self.get_logger().info(
                    '\033[1;34m%s\033[0m' % ('park detected: x=%d area=%d box=' % (center[0], self.park_area) + str(i.box)))
            elif class_name == 'red' or class_name == 'green':
                seen_traffic = True
                self.traffic_signs_status = i

        self.crosswalk_count = self.count_distinct_crosswalks(crosswalk_ys, self.CROSSWALK_GAP)

        if crosswalk_ys:
            self.get_logger().info('crosswalk distinct=%d ys=%s'
                                   % (self.crosswalk_count, str(sorted(crosswalk_ys))))

        if not seen_park:
            self.park_x = -1
            self.park_area = 0
        if not seen_traffic:
            self.traffic_signs_status = None
        self.update_right_turn_anchor(right_metrics)


def main():
    node = SelfDrivingNode('self_driving')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()


if __name__ == "__main__":
    main()