import onnxruntime as rt
import numpy as np
import torch
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords, plot_one_box
import random
import cv2
from pathlib import Path

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

sess = rt.InferenceSession("./weights/yolov5s.onnx")
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
# img_path = "/Users/samuelchin/Downloads/2DMOT2015/test/ADL-Rundle-1/img1/000001.jpg"
# img = cv2.imread(img_path)

dataset = LoadImages("inference/images", img_size=640)
for path, img, im0s, vid_cap in dataset:
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    img = img / 255.0
    print("wtf", img.shape)
    pred = sess.run(["output", "424", "444"], {input_name: img})
    pred0 = pred[0]
    pred0 = torch.from_numpy(pred0)
    pred0 = torch.reshape(pred0, (1, 11520, 85))
    pred1 = pred[1]
    pred1 = torch.from_numpy(pred1)
    pred1 = torch.reshape(pred1, (1, 2880, 85))
    pred2 = pred[2]
    pred2 = torch.from_numpy(pred2)
    pred2 = torch.reshape(pred2, (1, 720, 85))
    pred = torch.cat((pred0, pred1, pred2), 1)
    pred = non_max_suppression(pred, 0.4, 0.5)
    print(pred[0].shape)
    # pred = non_max_suppression(pred, 0.8, 0.6)
    for i, det in enumerate(pred):
        if det is not None and len(det):
            print(img.shape)
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            print(det)
            s = ""
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string
            print("%s" %(s))

            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (names[int(cls)], conf)
                print(label)
                plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=3)

    save_path = str(Path("inference/outputs") / Path(path).name)
    cv2.imwrite(save_path, im0s)
    break

# Padded resize
# img = letterbox(img, new_shape=640, auto=False)[0]
# cv2.imwrite("test.jpg", img)

# Convert
# img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#
# print(img.shape)
# res = sess.run(["output"], {input_name: img})
# res = torch.from_numpy(res[0])
# pred = non_max_suppression(res, 0.8, 0.6)
#
