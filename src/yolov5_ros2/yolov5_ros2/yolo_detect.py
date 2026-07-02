#!/usr/bin/env python3
# encoding: utf-8
# yolov5 target detection (큐 + 전용 추론 스레드 버전)

# ===== PyTorch 스레드 제한 (반드시 torch import보다 먼저) =====
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)

import cv2
import yaml
import time
import queue
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from yolov5 import YOLOv5
import yolov5_ros2.fps as fps
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
from interfaces.msg import ObjectInfo, ObjectsInfo
from yolov5_ros2.cv_tool import px2xy

ros_distribution = os.environ.get("ROS_DISTRO")
package_share_directory = get_package_share_directory('yolov5_ros2')


class YoloV5Ros2(Node):
    def __init__(self):
        super().__init__('yolov5_ros2')
        self.get_logger().info(f"Current ROS 2 distribution: {ros_distribution}")
        self.fps = fps.FPS()

        # ---- 파라미터 ----
        self.declare_parameter("device", "cuda", ParameterDescriptor(
            name="device", description="Compute device, default: cpu, options: cuda:0"))
        self.declare_parameter("model", "yolov5s", ParameterDescriptor(
            name="model", description="Model selection"))
        self.declare_parameter("image_topic", "/ascamera/camera_publisher/rgb0/image", ParameterDescriptor(
            name="image_topic", description="Image topic"))
        self.declare_parameter("show_result", False, ParameterDescriptor(
            name="show_result", description="Display detection results"))
        self.declare_parameter("pub_result_img", False, ParameterDescriptor(
            name="pub_result_img", description="Publish detection result images"))

        # ---- 서비스 ----
        self.create_service(Trigger, '/yolov5/start', self.start_srv_callback)
        self.create_service(Trigger, '/yolov5/stop', self.stop_srv_callback)
        self.create_service(Trigger, '~/init_finish', self.get_node_state)

        # ---- 모델 로드 ----
        model_path = package_share_directory + "/config/" + self.get_parameter('model').value + ".pt"
        device = self.get_parameter('device').value
        self.yolov5 = YOLOv5(model_path=model_path, device=device)

        # ---- 퍼블리셔 ----
        self.yolo_result_pub = self.create_publisher(Detection2DArray, "yolo_result", 10)
        self.result_msg = Detection2DArray()
        self.object_pub = self.create_publisher(ObjectsInfo, '~/object_detect', 1)
        self.result_img_pub = self.create_publisher(Image, "result_img", 10)

        # ---- 옵션 ----
        self.bridge = CvBridge()
        self.show_result = self.get_parameter('show_result').value
        self.pub_result_img = self.get_parameter('pub_result_img').value

        # ===== 큐 + 전용 추론 스레드 =====
        self.running = True
        self.frame_skip = 3            # 들어오는 프레임 중 N장에 1번만 큐에 적재
        self._frame_count = 0
        # maxsize=1: 항상 "가장 최신" 프레임만 추론 (밀린 프레임은 버림 → 지연 최소화)
        self.image_queue = queue.Queue(maxsize=1)

        # ---- 이미지 구독 (콜백은 적재만) ----
        image_topic = self.get_parameter('image_topic').value
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 1)

        # ---- 추론 전용 스레드 시작 ----
        threading.Thread(target=self.inference_loop, daemon=True).start()

    def get_node_state(self, request, response):
        response.success = True
        return response

    def start_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "start yolov5 detect")
        response.success = True
        response.message = "start"
        return response

    def stop_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "stop yolov5 detect")
        response.success = True
        response.message = "stop"
        return response

    # ===== 콜백: 무거운 추론을 하지 않고 큐에 최신 프레임만 적재 =====
    def image_callback(self, msg: Image):
        # 프레임 솎기
        self._frame_count += 1
        if self._frame_count % self.frame_skip != 0:
            return

        image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

        # 큐가 차 있으면 오래된 프레임을 버리고 최신으로 교체 (지연 누적 방지)
        if self.image_queue.full():
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            # 헤더도 함께 넘겨 결과 이미지 타임스탬프 유지
            self.image_queue.put_nowait((image, msg.header))
        except queue.Full:
            pass

    # ===== 전용 스레드: 큐에서 꺼내 추론 + publish =====
    def inference_loop(self):
        while self.running and rclpy.ok():
            try:
                image, header = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            try:
                self.process_one(image, header)
            except BaseException as e:
                self.get_logger().warn('inference error: %s' % str(e))

    def process_one(self, image, header):
        detect_result = self.yolov5.predict(image)

        self.result_msg.detections.clear()
        self.result_msg.header.frame_id = "camera"
        self.result_msg.header.stamp = self.get_clock().now().to_msg()

        predictions = detect_result.pred[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        h, w = image.shape[:2]
        objects_info = []   # 한 프레임의 모든 객체를 모아 한 번에 publish

        for index in range(len(categories)):
            name = detect_result.names[int(categories[index])]
            x1, y1, x2, y2 = boxes[index]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            detection2d = Detection2D()
            detection2d.id = name
            if ros_distribution == 'galactic':
                detection2d.bbox.center.x = center_x
                detection2d.bbox.center.y = center_y
            else:
                detection2d.bbox.center.position.x = center_x
                detection2d.bbox.center.position.y = center_y
            detection2d.bbox.size_x = float(x2 - x1)
            detection2d.bbox.size_y = float(y2 - y1)

            obj_pose = ObjectHypothesisWithPose()
            obj_pose.hypothesis.class_id = name
            obj_pose.hypothesis.score = float(scores[index])
            detection2d.results.append(obj_pose)
            self.result_msg.detections.append(detection2d)

            object_info = ObjectInfo()
            object_info.class_name = name
            object_info.box = [x1, y1, x2, y2]
            object_info.score = round(float(scores[index]), 2)
            object_info.width = w
            object_info.height = h
            objects_info.append(object_info)

            if self.show_result or self.pub_result_img:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{name}:{obj_pose.hypothesis.score:.2f}", (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ===== 루프 밖에서 한 번만 publish (프레임당 1개 메시지) =====
        object_msg = ObjectsInfo()
        object_msg.objects = objects_info
        self.object_pub.publish(object_msg)

        if len(self.result_msg.detections) > 0:
            self.yolo_result_pub.publish(self.result_msg)

        if self.show_result:
            self.fps.update()
            image = self.fps.show_fps(image)
            cv2.imshow('result', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if self.pub_result_img:
            result_img_msg = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
            result_img_msg.header = header
            self.result_img_pub.publish(result_img_msg)


def main():
    rclpy.init()
    node = YoloV5Ros2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()