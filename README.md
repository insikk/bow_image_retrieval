# bow_image_retrieval
Bag-of-words Image Retrieval

Let's implement traditional way of image search system

Any contributions are welcome :)

# Requirements

Tested on Ubuntu 14.04

```
# OpenCV Contrib
pip install opencv-contrib-python
```

FLANN-1.9.1 (https://github.com/mariusmuja/flann/releases/tag/1.9.1)

# Step by Step TODO

## Data preparation 
[Oxford 5k](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/)

## Feature Generation
Image Feature Generator
Image Feature Descriptor

## Visual Words Generation
Generate Visual Words: Vector quantization with several methods
* flat k-means clustering: simple, but failed to scale
* approximate k-menas 
    * For nearest negihbor finding, we use randomized kd tree forest. We use FLANN(https://github.com/mariusmuja/flann) implementation
     

## Search Engine
Approximage nearest neighbor

## Evlauator


# Acknowledgement

## Related Paper

## Related Project
https://github.com/deviantony/docker-elk

