import numpy as np
import os
import sys
import tensorflow as tf
import glob
import cv2


from utils import visualization_utils as vis_util
from utils import label_map_util

MAX_NUMBER_OF_BOXES = 10
MINIMUM_CONFIDENCE = 0.9

PATH_TO_LABELS = 'annotations/label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'test_images'

# Path to frozen detection graph. This is the actual model that is used
# for the object detection.
MODEL_NAME = 'custom_model'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=sys.maxsize, use_display_name=True)
CATEGORY_INDEX = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def detect_objects(image_path):
    image_np = cv2.imread(image_path)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        CATEGORY_INDEX,
        min_score_thresh=MINIMUM_CONFIDENCE,
        use_normalized_coordinates=True,
        line_thickness=8)
    cv2.imwrite("output/" + image_path, image_np)

TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*'))

# Load model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

print("Model loaded")
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name(
            'detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name(
            'detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name(
            'detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        for image_path in TEST_IMAGE_PATHS:
            detect_objects(image_path)

print("Done")
