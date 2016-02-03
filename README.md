SORT
=====

A simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences.

By Alex Bewley
DynamicDetection.com

### Introduction

SORT is a barebones implementation of a visual multiple object tracking framework based on rudimentary data association and state estimation techniques. It is designed for online tracking applications where only past and current frames are available and the method produces object identities on the fly. While this minimalistic tracker doesn't handle occlusion or re-entering objects its purpose is to serve as a baseline and testbed for the development of future trackers.

SORT was initially described in an [arXiv tech report](http://arxiv.org/abs/1602.00763). At the time of the initial publication, SORT was ranked the best *open source* multiple object tracker on the [MOT benchmark](https://motchallenge.net/results/2D_MOT_2015/).

This code has been tested on Mac OSX 10.10, and Ubuntu 14.04, with Python 2.7 (anaconda).

**Note:** A significant proportion of SORT's accuracy is attributed to the detections.
For your convenience, this repo also contains *Faster* RCNN detections for the MOT benchmark sequences. To run the detector yourself please see the original [*Faster* RCNN project](https://github.com/ShaoqingRen/faster_rcnn) or the python reimplementation of [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) by Ross Girshick.

### License

SORT is released under the GPL License (refer to the LICENSE file for details) to promote the open use of the tracker and future improvements. If you require a permissive license contact Alex (alex@dynamicdetection.com).

### Citing SORT

If you find this repo useful in your research, please consider citing:

    @article{bewley2015sort,
        Author = {Alex Bewley and Zonguan Ge and Lionel Ott and Fabio Ramos and Ben Upcroft},
        Title = {Simple Online and Realtime Tracking},
        Journal = {arXiv preprint arXiv:1602.00763},
        Year = {2016}
    }



### Dependencies:

0. [`scikit-learn`](http://scikit-learn.org/stable/)
0. [`scikit-image`](http://scikit-image.org/download)
0. [`FilterPy`](https://github.com/rlabbe/filterpy)
```
$ pip search filterpy
```


### Demo:

To run the tracker with the provided detections:

```
$ cd path/to/sort
$ python sort.py
```

To display the results you need to:

0. Download the [2D MOT 2015 benchmark dataset](https://motchallenge.net/data/2D_MOT_2015/#download)
0. Create a symbolic link to the dataset
  ```
  $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark
  ```
0. Run the demo with the ```--display``` flag
  ```
  $ python sort.py --display
  ```


### Main Results

Using the [MOT challenge deckit](https://motchallenge.net/devkit/) the method produces the following results (as described in the paper).

 Sequence       | Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL
--------------- |:---------------:|:-------------:|:-------------------:|:------------------:
 TUD-Campus     | 68.5  94.3  0.21|  8   6   2   0|   15   113    6    9|  62.7  73.7  64.1
 ETH-Sunnyday   | 77.5  81.9  0.90| 30  11  16   3|  319   418   22   54|  59.1  74.4  60.3
 ETH-Pedcross2  | 51.9  90.8  0.39|133  17  60  56|  330  3014   77  103|  45.4  74.8  46.6
 ADL-Rundle-8   | 44.3  75.8  1.47| 28   6  16   6|  959  3781  103  211|  28.6  71.1  30.1
 Venice-2       | 42.5  64.8  2.75| 26   7   9  10| 1650  4109   57  106|  18.6  73.4  19.3
 KITTI-17       | 67.1  92.3  0.26|  9   1   8   0|   38   225    9   16|  60.2  72.3  61.3
 *Overall*      | 49.5  77.5  1.24|234  48 111  75| 3311 11660  274  499|  34.0  73.3  35.1
