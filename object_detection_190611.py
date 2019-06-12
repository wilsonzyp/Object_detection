#!/usr/bin/env python
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!
# This notebook will walk you step by step through the process of using a pre-trained model
# to detect objects in an image.

import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util  # 这里一定要记住：
# cannot import name 'string_int_label_map_pb2' from 'object_detection.protos'
# 要运行protoc object_detection/protos/string_int_label_map.proto --python_out=. 这一句生成.py 文件
from object_detection.utils import visualization_utils as vis_util
import time

# model 存放位置  文件名称
MODEL_NAME = r'trained_models/ship_detection_by_f_rcnn_inception_20190612'
# MODEL_NAME = r'trained_models/detection_on_sea_by_ssd_20190611'
# MODEL_NAME = r'trained_models/faster_rcnn_inception_v2_coco_2018_01_28'
# MODEL_NAME = r'trained_models/faster_rcnn_nas_coco_2018_01_28'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
# frozen_inference_graph.pb 这个名字不用改，训练好的模型
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
# 标签路径、名称
# PATH_TO_LABELS = os.path.join('trained_models', 'mscoco_label_map.pbtxt'
PATH_TO_LABELS = os.path.join('trained_models', 'ship_label.pbtxt')
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# 测试文件路径
PATH_TO_TEST_IMAGES_DIR = r'imageset/test_images'
# 列出所有测试图片
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, name) for name in os.listdir(PATH_TO_TEST_IMAGES_DIR)]
# Size, in inches, of the output images.
IMAGE_NAMES = os.listdir(PATH_TO_TEST_IMAGES_DIR)
IMAGE_SIZE = (12, 8)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Detection
def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


if __name__ == '__main__':
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Loading label map
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    print("开始进行图像识别")
    times = []
    for i,image_path in enumerate(TEST_IMAGE_PATHS):
        temp = time.time()
        # print("正在处理",image_path)
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
        plt.savefig(r'imageset/output/' + IMAGE_NAMES[i])
        print("成功保存 "+IMAGE_NAMES[i])
        times.append(time.time()-temp)
    print("平均识别一张图片用时：", sum(times)/len(times))
