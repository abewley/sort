SORT
=====

A simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences.
See an example [video here](https://alex.bewley.ai/misc/SORT-MOT17-06-FRCNN.webm).

By Alex Bewley  

### Introduction

SORT is a barebones implementation of a visual multiple object tracking framework based on rudimentary data association and state estimation techniques. It is designed for online tracking applications where only past and current frames are available and the method produces object identities on the fly. While this minimalistic tracker doesn't handle occlusion or re-entering objects its purpose is to serve as a baseline and testbed for the development of future trackers.

SORT was initially described in [this paper](http://arxiv.org/abs/1602.00763). At the time of the initial publication, SORT was ranked the best *open source* multiple object tracker on the [MOT benchmark](https://motchallenge.net/results/2D_MOT_2015/).

**Note:** A significant proportion of SORT's accuracy is attributed to the detections.
For your convenience, this repo also contains *Faster* RCNN detections for the MOT benchmark sequences in the [benchmark format](https://motchallenge.net/instructions/). To run the detector yourself please see the original [*Faster* RCNN project](https://github.com/ShaoqingRen/faster_rcnn) or the python reimplementation of [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) by Ross Girshick.

**Also see:**
A new and improved version of SORT with a Deep Association Metric implemented in tensorflow is available at [https://github.com/nwojke/deep_sort](https://github.com/nwojke/deep_sort) .

### License

SORT is released under the GPL License (refer to the LICENSE file for details) to promote the open use of the tracker and future improvements. If you require a permissive license contact Alex (alex@bewley.ai).

### Citing SORT

If you find this repo useful in your research, please consider citing:

    @inproceedings{Bewley2016_sort,
      author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
      booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
      title={Simple online and realtime tracking},
      year={2016},
      pages={3464-3468},
      keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
      doi={10.1109/ICIP.2016.7533003}
    }


### Dependencies:

To install required dependencies run:
```
$ pip install -r requirements.txt
```


### Demo:

To run the tracker with the provided detections:

```
$ cd path/to/sort
$ python sort.py
```

To display the results you need to:

1. Download the [2D MOT 2015 benchmark dataset](https://motchallenge.net/data/2D_MOT_2015/#download)
0. Create a symbolic link to the dataset
  ```
  $ ln -s /path/to/MOT2015_challenge/data/2DMOT2015 mot_benchmark
  ```
0. Run the demo with the ```--display``` flag
  ```
  $ python sort.py --display
  ```


### Main Results

Using the [MOT challenge devkit](https://motchallenge.net/devkit/) the method produces the following results (as described in the paper).

 Sequence       | Rcll | Prcn |  FAR | GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL
--------------- |:----:|:----:|:----:|:-------------:|:-------------------:|:------------------:
 TUD-Campus     | 68.5 | 94.3 | 0.21 |  8   6   2   0|   15   113    6    9|  62.7  73.7  64.1
 ETH-Sunnyday   | 77.5 | 81.9 | 0.90 | 30  11  16   3|  319   418   22   54|  59.1  74.4  60.3
 ETH-Pedcross2  | 51.9 | 90.8 | 0.39 | 133  17  60  56|  330  3014   77  103|  45.4  74.8  46.6
 ADL-Rundle-8   | 44.3 | 75.8 | 1.47 | 28   6  16   6|  959  3781  103  211|  28.6  71.1  30.1
 Venice-2       | 42.5 | 64.8 | 2.75 | 26   7   9  10| 1650  4109   57  106|  18.6  73.4  19.3
 KITTI-17       | 67.1 | 92.3 | 0.26 |  9   1   8   0|   38   225    9   16|  60.2  72.3  61.3
 *Overall*      | 49.5 | 77.5 | 1.24 | 234  48 111  75| 3311 11660  274  499|  34.0  73.3  35.1


### Using SORT in your own project

Below is the gist of how to instantiate and update SORT. See the ['__main__'](https://github.com/abewley/sort/blob/master/sort.py#L239) section of [sort.py](https://github.com/abewley/sort/blob/master/sort.py#L239) for a complete example.
    
    from sort import *
    
    #create instance of SORT
    mot_tracker = Sort() 
    
    # get detections
    ...
    
    # update SORT
    track_bbs_ids = mot_tracker.update(detections)

    # track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
    ...
    
 
