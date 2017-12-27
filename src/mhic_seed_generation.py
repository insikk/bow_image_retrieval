"""
# minHash Image Clustering (MHIC) algorithm (Seed Generation Only)

Implementation based on 
* [Large-Scale Discovery of Spatially Related Images](ieeexplore.ieee.org/iel5/34/4359286/05235143.pdf) by Ondrej Chum and Jiri Matas
* [Scalable Near Identical Image and Shot Detection - Microsoft](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/civr2007.pdf) by Ondrej Chum, James Philbin, Michael Isard, Andrew Zisserman

## Purpose

If we see a similar image cluster as a connected compoenent, images are vertex. 
We have to find edges to get image cluster. minHash can be used to find subset of the edges quickly. 

Afterward, you may use image retrieval system to complete the connected component. 


## Requirements

* Visual words index list for each image

Author: Insik Kim (insik92@gmail.com)
"""
import argparse
import os
import pickle

import sys
sys.path.insert(0,'..')

from utils.bow_utils import get_idf_word_weights
# from utils.minhash_utils import VisualMinHashWithDataSketch
from utils.minhash_utils import SketchCollisionTester, VisualMinHashWithLookupTable
from utils.minhash_utils import get_collision_pairs
from utils.geo_verif_utils import get_ransac_inlier, draw_ransac

from multiprocessing import Pool

import cv2
from tqdm import tqdm
import numpy as np

import random


def get_keypoints(image_name):
    # Oxford 5k dataset provides already converted visual words. We could use this one
    oxf5k_visualword_dir = './data/word_oxc1_hesaff_sift_16M_1M'
    filepath = os.path.join(oxf5k_visualword_dir, image_name + ".txt")
    kp = []
    with open(filepath) as f:
        lines = list(map(lambda x: x.strip(), f.readlines()[2:])) # ignore first two lines
        for l in lines:
            val = l.split(" ")
            visual_word_index = int(val[0])-1 # This data use 1 to 1,000,000. convert to zero-based so 0 to 999,999
            x = float(val[1])
            y = float(val[2])
            a = float(val[3])
            b = float(val[4])
            c = float(val[5])
            # TODO: generate ellipse shaped key point
            # Refer: https://math.stackexchange.com/questions/1447730/drawing-ellipse-from-eigenvalue-eigenvector
            # Refer: http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/display_features.m
            # Refer: http://www.robots.ox.ac.uk/~vgg/research/affine/detectors.html
            key_point = cv2.KeyPoint(x, y, 1)
            kp.append(key_point)
    return kp

def print_similar_paris_statistics(similar_pairs, th_sim_min, th_sim_max):
    import pprint
    similar_pairs = list(similar_pairs)
    print("----------SIMILAR PAIRS STATISTICS------------")
    print('len similar_pairs:', len(similar_pairs))
    similar_pairs.sort(key=lambda x: x[1])
    print('similar_pairs[:5]:')
    pprint.pprint(similar_pairs[:5])

    count_irr = 0
    count_sim = 0
    count_dup = 0

    for image_cluster, score in similar_pairs:
        if score < th_sim_min:
            count_irr += 1
        elif score >= th_sim_min and score < th_sim_max:
            count_sim += 1
        else:
            count_dup += 1
    print("count (irr, sim, dup) : ({}, {}, {})".format(count_irr, count_sim, count_dup))
    print()


def print_ransac_result_statistics(ransac_result):
    import pprint
    print("----------RANSAC RESULT STATISTICS------------")
    print('len ransac_result:', len(ransac_result))
    ransac_result.sort(key=lambda x: x[2], reverse=True)
    print('ransac_result[:5]:')
    pprint.pprint(ransac_result[:5])
    print()

def show_image_cluster(config, image_names):
    """
    show image cluster for oxford 5k dataset
    """
    # Visualize images assigned to this cluster
    from PIL import Image
    import matplotlib.pyplot as plt
    image_dir = config["image_dir"]
    imgs = []
    for image_name in image_names:
        image_name = image_name.replace("oxc1_", "") + ".jpg"
        image_path = os.path.join(image_dir, image_name)
        img = Image.open(image_path)
        imgs.append(img)
    cols = 5
    imgs = imgs[:cols]
    plt.figure(figsize=(20, 5))
    for i, img in enumerate(imgs):
        plt.subplot(1, cols, i + 1)
        plt.imshow(img)
    plt.show()

def show_image_pair_ransac(config, image_names):
    """
    show image cluster for oxford 5k dataset
    """
    # Visualize images assigned to this cluster
    from PIL import Image
    import matplotlib.pyplot as plt

    with open(config["image_descriptor_dict_path"], 'rb') as f:
        # key: image_name, value: 2d numpy array of shape (num_descriptor, dim_descriptor)
        image_descriptor_dict = pickle.load(f)
    image_dir = config["image_dir"]

    imgs = []
    for image_name in image_names:
        image_name = image_name.replace("oxc1_", "") + ".jpg"
        image_path = os.path.join(image_dir, image_name)
        img = cv2.imread(image_path)
        imgs.append(img)

    kp1 = get_keypoints(image_names[0])
    des1 = np.array(image_descriptor_dict[image_names[0]], dtype=np.uint8)
    kp2 = get_keypoints(image_names[1])
    des2 = np.array(image_descriptor_dict[image_names[1]], dtype=np.uint8)

    return draw_ransac(imgs[0], imgs[1], kp1, kp2, des1, des2, False, 'BRUTE_FORCE', None)


def parallel_task(val):
    image_cluster, score = val
    kp1 = get_keypoints(image_cluster[0])
    des1 = np.array(image_descriptor_dict[image_cluster[0]], dtype=np.uint8)
    kp2 = get_keypoints(image_cluster[1])
    des2 = np.array(image_descriptor_dict[image_cluster[1]], dtype=np.uint8)
    num_inlier = get_ransac_inlier(kp1, kp2, des1, des2)
    return (image_cluster, score, num_inlier)

def filter_keep_similar_only(score, min_th, max_th):
    return score >= min_th and score < max_th


def similarity_estimation(bow_dict, vocab_size):
    """
    MHIC algorithm. Step 2. Similarity Estimation.
    """
    # To use idf word weighthing.
    # inverse document frequency. It is importance of the words.
    word_weights = get_idf_word_weights(bow_dict, vocab_size=vocab_size)

    collision_tester = SketchCollisionTester(minHash_param_k=512)
    # vmh = VisualMinHashWithDataSketch(minHash_hash_num=512, minHash_param_k=512, minHash_param_s=3)

    # The word weighted minHash gives more probability to have minHash value for important words.
    vmh = VisualMinHashWithLookupTable(minHash_hash_num=512, vocab_size=vocab_size, word_weights=word_weights, minHash_param_k=512, minHash_param_s=3)

    similar_pairs = get_collision_pairs(bow_dict, vmh, collision_tester)

    # Timing: 5062 images took 2 min 30 sec.
    # History:
    # Oxf5k, 16M features, 1M cluster(visual_words), count (irr, sim, dup) : (194, 150, 4)
    # Oxf5k, 16M features, codebook train size 100k with 4 subspaces, 2^17 cluster(visual_words), count (irr, sim, dup) : (15348, 1141, 2)
    # Oxf5k, 16M features, codebook train size 1M with 4 subspaces, 2^17 cluster(visual_words), count (irr, sim, dup) : (15587, 1155, 2)
    # Oxf5k, 16M features, codebook train size 1M with 8 subspaces, 2^17 cluster(visual_words), count (irr, sim, dup) : (12775, 724, 2)
    return similar_pairs

class Config:
    pass
def read_config(config_path):
    """
    From configuration json file, read config and return the object.
    """
    import yaml
    import pprint
    with open(config_path, "r") as f:
        config = yaml.load(f)

    config.update(config["general"])
    config.pop("general")
    config.update(config["mhic_config"])
    config.pop("mhic_config")

    pprint.pprint(config)

    return config

def main():
    parser = argparse.ArgumentParser(description='MinHash Image Clustering Seed Generation')
    parser.add_argument('--config', default="./config/mhic_seed_gen.config", help='config file path')
    args = parser.parse_args()
    print("use config path:", args.config)
    config = read_config(args.config)

    # -------- Logic Start Here ----------------
    if not os.path.exists(config["work_dir"]):
        os.mkdir(config["work_dir"])

    # Read Bag-of-visual-words of target dataset from file.
    with open(config["bow_dict_save_path"], 'rb') as f:
        # key: image_name, value: list of visual word index
        bow_dict = pickle.load(f)
    if not os.path.exists(os.path.join(config["work_dir"], config["output_similar_pair_result"])):
        print("sweep dataset with minHash to find collisions. It will take 3 min for D=5000 images. time complexity O(D).")
        similar_pairs = similarity_estimation(bow_dict, config["vocab_size"])
        pickle.dump(similar_pairs, open(os.path.join(config["work_dir"], config["output_similar_pair_result"]), 'wb'))
    else:
        similar_pairs = pickle.load(open(os.path.join(config["work_dir"], config["output_similar_pair_result"]), 'rb'))
        print("skip minHash sweep, because there is file: {}. We use it.".format(os.path.join(config["work_dir"], config["output_similar_pair_result"])))

    print_similar_paris_statistics(similar_pairs, config["THRESHOLD_DATAMINING_SIMILARITY_MIN"], config["THRESHOLD_DATAMINING_SIMILARITY_MAX"])

    if not os.path.exists(os.path.join(config["work_dir"], config["output_ransac_result"])):
        def initializer():
            global image_descriptor_dict
            with open(config["image_descriptor_dict_path"], 'rb') as f:
                # key: image_name, value: 2d numpy array of shape (num_descriptor, dim_descriptor)
                image_descriptor_dict = pickle.load(f)
        pool = Pool(config["num_processes"], initializer)
        ransac_result = []
        similar_pairs = list(filter(lambda x: filter_keep_similar_only(x[1], config["THRESHOLD_DATAMINING_SIMILARITY_MIN"], config["THRESHOLD_DATAMINING_SIMILARITY_MAX"]), similar_pairs))
        for result in tqdm(pool.imap_unordered(parallel_task, similar_pairs), total=len(similar_pairs)):
            if result is not None:
                ransac_result.append(result)

        pickle.dump(ransac_result, open(os.path.join(config["work_dir"], config["output_ransac_result"]), 'wb'))
    else:
        ransac_result = pickle.load(open(os.path.join(config["work_dir"], config["output_ransac_result"]), 'rb'))
        print("skip RANSAC geometric verification, because there is file: {}. We use it.".format(os.path.join(config["work_dir"], config["output_ransac_result"])))

    print_ransac_result_statistics(ransac_result)

if __name__ == "__main__":
    main()