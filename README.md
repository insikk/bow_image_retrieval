# bow_image_retrieval
Bag-of-words Image Retrieval

Let's implement traditional way of image search system

Any contributions are welcome :)

# Requirements

Tested on Ubuntu 14.04

```
# OpenCV Contrib
pip install opencv-contrib-python
           
# Scikit for RANSAC
pip install scikit-image

# Datasketch for minHash
pip install datasketch -U

# Config Parser
pip install pyyaml
```

## Modified Datasketch for minHash
 
 You need modified version of datasketch package. https://github.com/insikk/datasketch/tree/update_with_int
 You can simply install from submodule added in this directory. 
 ```
 git clone --recursive <this repo>
 cd datasketch
 pip install -e .
 ```

## Modified pqkmeans

```
cd pqkmeans
git submodule update --init --recursive # get pybind 
pip install -e .
```

## FLANN and python binding (pyflann)

Download https://github.com/mariusmuja/flann/releases/tag/1.9.1 for python binding. 
build with cmake
```
# Install required packages 
sudo apt-get install libhdf5-dev libgtest-dev

# Finish google test install following https://www.eriksmistad.no/getting-started-with-google-test-on-ubuntu/
cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make
 
# copy or symlink libgtest.a and libgtest_main.a to your /usr/lib folder
sudo cp *.a /usr/lib

# goto ./lib/flann-1.8.4
mkdir build
cd build
cmake -DBUILD_CUDA_LIB=ON ..
make
sudo make install
# Make sure it builds python binding. 
python
>>> import pyflann
>>> # shows no error
```

## [PQk-means](https://github.com/DwangoMediaVillage/pqkmeans)




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

## (Optional) MinHash Image Clustering
```
cd .
export PYTHONPATH=$PYTHONPATH:$(pwd)
python ./src/mhic_seed_generation.py

```

## Search Engine
Approximage nearest neighbor

## Evlauator


# Acknowledgement

## Related Paper

## Related Project
https://github.com/deviantony/docker-elk

