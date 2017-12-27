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

import os
import pickle
from utils.bow_utils import get_idf_word_weights

# from utils.minhash_utils import VisualMinHashWithDataSketch
from utils.minhash_utils import SketchCollisionTester, VisualMinHashWithLookupTable
from utils.minhash_utils import get_collision_pairs


from multiprocessing import Pool
from utils.geo_verif_utils import get_ransac_inlier, draw_ransac
import cv2
from tqdm import tqdm
import numpy as np
import pickle

import random


def show_image_cluster(image_dir, image_names):
    """
    show image cluster for oxford 5k dataset
    """
    # Visualize images assigned to this cluster    
    from PIL import Image
    import matplotlib.pyplot as plt
    
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
    similar_pairs = list(similar_pairs)
    print('len similar_pairs:', len(similar_pairs))
    similar_pairs.sort(key=lambda x: x[1])
    print('similar_pairs[:5]:', similar_pairs[:5])

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
    

    
def show_image_pair_ransac(image_dir, image_names, kp1, kp2, des1, des2):
    """
    show image cluster for oxford 5k dataset
    """
    # Visualize images assigned to this cluster    
    from PIL import Image
    import matplotlib.pyplot as plt

    imgs = []
    for image_name in image_names:
        image_name = image_name.replace("oxc1_", "") + ".jpg"
        image_path = os.path.join(image_dir, image_name)
        img = cv2.imread(image_path)
        imgs.append(img)

    return draw_ransac(imgs[0], imgs[1], kp1, kp2, des1, des2, False, 'BRUTE_FORCE', None)


def parallel_task(val):
    image_cluster, score = val
    kp1 = get_keypoints(image_cluster[0])
    des1 = np.array(image_descriptor_dict[image_cluster[0]], dtype=np.uint8)
    kp2 = get_keypoints(image_cluster[1])
    des2 = np.array(image_descriptor_dict[image_cluster[1]], dtype=np.uint8)
    num_inlier = get_ransac_inlier(kp1, kp2, des1, des2)
    # num_inlier = show_image_pair_ransac(IMAGE_DIR, image_cluster, kp1, kp2, des1, des2)
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
    # Oxf5k, 16M features, 1M cluster(visual_words), count (irr, sim, dup) : ???
    # Oxf5k, 16M features, codebook train size 100k with 4 subspaces, 2^17 cluster(visual_words), count (irr, sim, dup) : (15348, 1141, 2)
    # Oxf5k, 16M features, codebook train size 1M with 4 subspaces, 2^17 cluster(visual_words), count (irr, sim, dup) : (15587, 1155, 2)
    # Oxf5k, 16M features, codebook train size 1M with 8 subspaces, 2^17 cluster(visual_words), count (irr, sim, dup) : (12775, 724, 2)
    return similar_pairs
def main(args=None):
    # TODO: read configuration from args or configuration file.

    num_processes = 2

    bow_dict_save_path = 'bow_dict_word_oxc1_hesaff_sift_16M_1M_pretrained.pkl'
    vocab_size=1000000

    # bow_dict_save_path = 'bow_dict_word_oxc1_hesaff_sift_16M_100k_handmade.pkl'
    # vocab_size = 2**17

    # bow_dict_save_path = 'bow_dict_word_oxc1_hesaff_sift_16M_1M_4_sub_handmade.pkl'
    # vocab_size = 2**17

    # bow_dict_save_path = 'bow_dict_word_oxc1_hesaff_sift_16M_1M_8_sub_handmade.pkl'
    # vocab_size = 2**17

    # For datamining purpose, we want to get less simlar but the same scene.
    # So we are interested in similiarity in [THRESHOLD_DATAMINING_SIMILARITY_MIN, THRESHOLD_DATAMINING_SIMILARITY_MAX]
    # See Large-Scale Discovery of Spatially Related Images. Sec 3.2 for THRESHOLD_DATAMINING_SIMILARITY_MIN
    # See Scaleable Near Identical Image and Shot Detection. Sec 4.3 for THRESHOLD_DATAMINING_SIMILARITY_MAX
    THRESHOLD_DATAMINING_SIMILARITY_MIN = 0.045
    THRESHOLD_DATAMINING_SIMILARITY_MAX = 0.35

    work_dir = './output'

    output_similar_pair_result = 'similar_pair.pkl'
    output_ransac_result = 'similar_pair_ransac.pkl'

    image_descriptor_dict_path = 'image_descriptor_dict_oxc1_hesaff_sift_16M.pkl'

    IMAGE_DIR = "./data/oxford/oxford5k/images"

    # -------- Logic Start Here ----------------
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    # Read Bag-of-visual-words of target dataset from file.
    with open(bow_dict_save_path, 'rb') as f:
        # key: image_name, value: list of visual word index
        bow_dict = pickle.load(f)
    if not os.path.exists(os.path.join(work_dir, output_similar_pair_result)):
        print("sweep dataset with minHash to find collisions. It will take 3 min for D=5000 images. time complexity O(D).")
        similar_pairs = similarity_estimation(bow_dict, vocab_size)
        pickle.dump(similar_pairs, open(os.path.join(work_dir, output_similar_pair_result), 'wb'))
    else:
        similar_pairs = pickle.load(open(os.path.join(work_dir, output_similar_pair_result), 'rb'))
        print("skip minHash sweep, because there is file: {}. We use it.".format(os.path.join(work_dir, output_similar_pair_result)))

    print_similar_paris_statistics(similar_pairs, THRESHOLD_DATAMINING_SIMILARITY_MIN, THRESHOLD_DATAMINING_SIMILARITY_MAX)

    def initializer():
        global image_descriptor_dict
        with open(image_descriptor_dict_path, 'rb') as f:
            # key: image_name, value: 2d numpy array of shape (num_descriptor, dim_descriptor)
            image_descriptor_dict = pickle.load(f)
    pool = Pool(num_processes, initializer)
    rasac_result = []
    similar_pairs = list(filter(lambda x: filter_keep_similar_only(x[1], THRESHOLD_DATAMINING_SIMILARITY_MIN, THRESHOLD_DATAMINING_SIMILARITY_MAX), similar_pairs))
    for result in tqdm(pool.imap_unordered(parallel_task, similar_pairs), total=len(similar_pairs)):
        if result is not None:
            rasac_result.append(result)

    pickle.dump(rasac_result, open(os.path.join(work_dir, output_ransac_result), 'wb'))

    print(rasac_result[:5])