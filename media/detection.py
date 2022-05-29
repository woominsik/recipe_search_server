# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 21:54:27 2021

@author: X5967T
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import time
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings

# -----프로젝트에 맞게 수정할 디렉터리 ---
model_paths = {
    'SSD' : '/workspace/Main_Server/detection/trained_model/SSD_resnet152',
    'Efficient' : '/workspace/Main_Server/detection/trained_model/efficientDet_d0_512',
    'faster_R-CNN' : '/workspace/Main_Server/detection/trained_model/faster_rcnn_resnet101_epochs_20000'
}

PATH_TO_MODEL_DIR = model_paths['Efficient']
PATH_TO_LABELS = PATH_TO_MODEL_DIR + '/annotation/label_map.pbtxt'
image_path = '/workspace/Main_Server/recipe_search_server/media/unnamed.jpg'
#---------------------------------------

img = []
PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"
PATH_TO_CKPT = PATH_TO_MODEL_DIR + "/checkpoint"

      

@tf.function
def detect_fn(image, detection_model):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    print('preprocessed')
    prediction_dict = detection_model.predict(image, shapes)
    print('predicted')
    detections = detection_model.postprocess(prediction_dict, shapes)
    print('postprocessed')
    
    print('return detect_fn')
    return detections

def get_ingredient_names(scores, classes, category_index, threshold=0.30):
    indicies = []
    names = []
    for i in range(len(scores)):
        if scores[i] >= threshold:
            indicies.append(i)
    for i in indicies:
        names.append(category_index[classes[i]]['name'])
    return names

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))
     
# mode, checkpoint를 서버 실행 시 만들어놓고 이 함수에 전달해주는 방식?
def get_ingredients(img_path = None):
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs['model']
    print('building model')
    detection_model = model_builder.build(model_config=model_config, is_training=False)
    print("model built!")
    
    #배포시 수정
#    if img_path != None:
#        image_path = img_path
    
    # Restore checkpoint
    print('getting checkpoint')
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()
    
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    
    print('checkpoint loaded!')
    
    warnings.filterwarnings('ignore')
    

    print('Running inference for {}... '.format(image_path))
    start = time.time()
    image_np = load_image_into_numpy_array(image_path)

    print('image loaded')
    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    print('image converted')
    
    detections = detect_fn(input_tensor, detection_model)
    print('detections')

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    min_score_threshold = 0.40
    label_id_offset = 1
    ingredients = get_ingredient_names(detections['detection_scores'],
                                 detections['detection_classes'] + label_id_offset,
                                 category_index,
                                 threshold=min_score_threshold)
    print(ingredients)
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=min_score_threshold,
            agnostic_mode=False)
    #plt.figure()pyt
    img.append(image_np_with_detections)
    #plt.imshow(image_np_with_detections)
    print('Done, took {}seconds'.format(time.time() - start))
    #plt.show()
    return ingredients

get_ingredients()
