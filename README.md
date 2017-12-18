# bow_image_retrieval
Bag-of-words Image Retrieval

Let's implement traditional way of image search system

Any contributions are welcome :)

# Requirements

Tested on Ubuntu 14.04

```
# OpenCV Contrib
pip install opencv-contrib-python

# Datasketch for minHash
pip install datasketch -U
```

## Flann and python binding (pyflann)

Download https://github.com/mariusmuja/flann/releases/tag/1.9.1 for python binding. 
build with cmake
```
# goto ./lib/flann-1.8.4
mkdir build
cd build
cmake ..
make
sudo make install
# Make sure it builds python binding. 
python
>>> import pyflann
>>> # shows no error
```


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

[ ] Enable CUDA support for FLANN during FLANN compilation. 

## Search Engine
Approximage nearest neighbor

## Evlauator


# Acknowledgement

## Related Paper

## Related Project
https://github.com/deviantony/docker-elk

