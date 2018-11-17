import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import time
import list_file as lf
import test_vgg
import json

from req import request
from multiprocessing import Process
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops


from utils import label_map_util

from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'ssd_inception_v2_coco_11_06_2017'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image, im_width, im_height):
#   (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (24, 16)

def swap_RB (frame, a, b):
    for i in range(len(frame)):
        for j in range(len(frame[i])):
            tp_channel = frame[i][j][a]
            frame[i][j][a] = frame[i][j][b]
            frame[i][j][b] = tp_channel

    return frame

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

#cap = cv2.VideoCapture(0)
key = 0
padding = 10
lists = lf.Lists()

def rcnn():
    while(1):
      img = Image.open('frame.jpg')
      #ret, frame = cap.read()
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      frame = load_image_into_numpy_array(img, 1280, 720)
      # img = cv2.resize(, (640, 480), interpolation=cv2.INTER_CUBIC)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      #image_np_expanded = np.expand_dims(frame, axis=0)
      # Actual detection.
      output_dict = run_inference_for_single_image(frame, detection_graph)
      # Visualization of the results of a detection.

      for i in range(3):
          #Get the boxes and labels for the top 5 object in the frame
          box = output_dict['detection_boxes'][i]
          label = category_index[output_dict['detection_classes'][i]]['name']
          print("label{}:".format(i)+str(label))

          #Transform relative coordinates to absolute coordinates
          y_min = box[0]*480-padding
          x_min = box[1]*640-padding
          y_max = box[2]*480+padding
          x_max = box[3]*640+padding

          box_tu = (x_min,y_min,x_max,y_max)

          #Check if it is a target or not. If so, store it in target_pos dic
          #And then put it in CNN

          img = Image.fromarray(frame)
          if (lists.is_target(label)):
               region = img.crop(box_tu)
               tempimg = np.array(region)

               target = test_vgg.recognize(tempimg, 'sky.pb')
              #TODO use the pic crop by the box to feed CNN and get new label
               lists.target_pos[target] = box

          if (lists.is_reference(label)):
               lists.references_pos[label] = box

      lists.connect_pairs()
      lists.target_pos.clear()
      lists.references_pos.clear()

      print('\n\n')
      print('The pairs are:')
      print(lists.pair)

      vis_util.visualize_boxes_and_labels_on_image_array(
          frame,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=10)
      frame = swap_RB(frame, 0, 2)

      cv2.imshow("capture", frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

      with open('data.txt', 'w') as f:
          json.dump(lists.pair, f)
          f.write('\n')

if __name__ == '__main__':
    p1 = Process(target = request)
    p2 = Process(target = rcnn)

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    cv2.destroyAllWindows()
