#!/usr/bin/env python
# coding: utf-8
from keras.backend.tensorflow_backend import set_session
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
# import miscellaneous modules
from numpy import expand_dims

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

data_path_base = './data/'
#data_path = './data/restaurant/'

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
set_session(get_session())

# ## Load RetinaNet model
#model_path = os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')
model_path="./final_models/resnet50_coco_best_v2.1.0.h5"

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet101')

# print(model.summary())

labels_to_names = {0: 'person'}

#返回指定下标的图像的人数
def returnNum(category,index):
    # load image
    image_name=data_path_base+category+'/'+str(index)+'.png'
    image = read_image_bgr(image_name)
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
    boxes, scores, labels = model.predict_on_batch(expand_dims(image, axis=0))
    boxes /= scale
    count_person=0
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.4:
            break
        if label==0:
            count_person+=1
    return count_person

