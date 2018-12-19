from styx_msgs.msg import TrafficLight
from keras.models import Model, load_model, model_from_json
from keras.layers import Activation
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from PIL import Image as PIL_Image
import numpy as np

import time
import json
import os


class TLClassifier(object):
    def __init__(self):
        #load classifier
        t = time.time()
        with open('../../models/mobilenet_sim_model_v1.json', 'r') as f:
            json_arch = f.read()
        self.model = model_from_json(json_arch, custom_objects={'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D})
        self.model.load_weights('../../models/mobilenet_sim_model_v1.weights')
        print 'load time of json arch model: {}'.format(time.time() - t)
        self.model._make_predict_function()


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #implement light color prediction
        image = image.resize((224, 224))
        image_array = np.asarray(image)
        image_array = (image_array - 127.5) / 128.0
        prediction = self.model.predict(image_array[None, ...])

        max_index = np.argmax(prediction)
        if max_index == 0:
            return TrafficLight.RED
        elif max_index == 1:
            return TrafficLight.YELLOW
        elif max_index == 2:
            return TrafficLight.GREEN
        else:
            return TrafficLight.UNKNOWN