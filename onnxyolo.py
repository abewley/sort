import onnxruntime as rt
import random
import numpy as np
from PIL import Image

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data

def get_sess():
    sess = rt.InferenceSession("./weights/yolov3-10.onnx")
    # sess = rt.InferenceSession("./weights/yolov5s.onnx")
    # sess = rt.InferenceSession("weights/yolov3-spp-ultralytics.onnx")
    input_name = sess.get_inputs()[0].name
    print("input name", input_name)
    input_shape = sess.get_inputs()[0].shape
    print("input shape", input_shape)
    input_type = sess.get_inputs()[0].type
    print("input type", input_type)
    output_name = sess.get_outputs()[0].name
    print("output name", output_name)
    output_shape = sess.get_outputs()[0].shape
    print("output shape", output_shape)
    output_type = sess.get_outputs()[0].type
    print("output type", output_type)
    return sess

def get_dets(p, sess):
    image = Image.open(p)
    # input
    image_data = preprocess(image)
    image_size = np.array([image.size[1], image.size[0]], dtype=np.float32).reshape(1, 2)
    preds = sess.run(
        ["yolonms_layer_1/ExpandDims_1:0",
         "yolonms_layer_1/ExpandDims_3:0",
         "yolonms_layer_1/concat_2:0"], {"input_1": image_data, "image_shape": image_size})
    dets = []
    for _, cls, idx in preds[2]:
        a, b, c, d = preds[0][0][idx]
        score = preds[1][0][cls][idx]
        dets.append([b,a,d,c,score])
    return np.array(dets)
