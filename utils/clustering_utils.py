import numpy as np
from tqdm import tqdm
import gc


def _matrix_closest_centroid(points, centroids):
    """
    returns an array containing the index to the nearest centroid for each point

    NOTE: this impelementation may work fast, but not scalable due to large memory usage. 
    With 16GB RAM, we cannot run 8000 points with 2000 centroids. 
    """
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def _naive_assignment(data_points, center_points):
    num_point, dim = data_points.shape # 128 for SIFT descriptor
    k, _ = center_points.shape
    associated_center_index = np.ones((num_point,), dtype=np.int32) * -1
    
    def get_distance(vector1, vector2):
        return np.linalg.norm(vector1-vector2)
    
    for i in tqdm(range(num_point)):
        min_index = -1
        min_val = np.iinfo(np.int64).max
        for j in range(k):
            dist = get_distance(data_points[i, :], center_points[j, :])
            if min_val > dist:
                min_val = dist
                min_index = j

        assert(min_index >= 0)
        associated_center_index[i] = min_index    
    
    return associated_center_index
    # return closest_centroid(data_points, center_points)

def _approximate_assignment(data_points, center_points):
    """
    Use k-d trees for assignment step in k-means clustering. 
    It change O(NK) to O(N log(K)). 
    See [Object retrieval with large vocabularies and fast spatial matching] paper, Section 3.1. 
    """
    import pyflann
    flann = pyflann.FLANN(algorithm='kdtree', trees=8)
    # flann = pyflann.FLANN(algorithm='kdtree_single')
    params = flann.build_index(center_points)
    # print("flann params:", params)

    ids, dists = flann.nn_index(data_points) # By default, only the most nearest point (centroid)
    
    return ids

def run_kmeans_clustering(data_points, k=2**17, max_iter=None, init_center_points=None, knn_method="naive"):
    """
    Args:
        data_points: 2d numpy array with shape of (num_point, dim)
        knn_method: "naive" or "kdtree"
    """
    print('k means clustering with k={K}, knn_method:{KNN_METHOD}'.format(K=k, KNN_METHOD=knn_method))
    
    # Init Step
    num_point, dim = data_points.shape # 128 for SIFT descriptor    
    
    if init_center_points is None:
        max_val = 2**8-1 # 255. uint8 max. for SIFT descriptor
        print('max_val:', max_val)
        center_points = np.random.randint(max_val, size=(k, dim), dtype=np.uint8)         
    else:
        center_points = init_center_points
    # center_points = center_points.astype(float)
    # print('init center points:', center_points)
        
    
    prev_val = np.ones((num_point,), dtype=np.uint8) * -1
    step = 0
    while True:
        if max_iter is not None and step == max_iter:
            break
        print('k-means clustering. start step:{}'.format(step))
        
        # Assignment Step        
        if knn_method == "naive":            
            associated_center_index = _naive_assignment(data_points, center_points)

        elif knn_method == "kdtree":
            associated_center_index = _approximate_assignment(data_points, center_points)
        else:
            raise Exception("no valid knn method")

        # print('index:', associated_center_index)
        # print('center points:', center_points)
        num_same_centroid = np.sum(np.equal(prev_val, associated_center_index))
        print("num_same_centroid:", num_same_centroid)
        if num_same_centroid == num_point:
            print("cluster assignment is not changed. It means convergence!")
            break
        
        prev_val = associated_center_index

        # Update Step         
        # print('associated_center_index:', associated_center_index)
        for center_idx in range(k):
            assigned_points = data_points[associated_center_index == center_idx]
            if assigned_points.shape[0] == 0:
                # print('no center')
                continue # do not update when no points are assigned to this cluster center
            # print('center_idx:', center_idx)
            # print('assigned_points:', assigned_points)
            assigned_points = assigned_points.astype(float)
            center_points[center_idx, :] = assigned_points.mean(axis = 0)
            # print(center_points[center_idx, :])
       
        # associated_center_count = [0] * k
        # associated_center_sum = np.zeros((k, dim), dtype=np.uint32)
        # # Improvement idea. change loop to matrix multiplication. select operator. 
        # for i in range(num_point):                
        #     assigned_index = associated_center_index[i]
        #     associated_center_sum[assigned_index, :] = associated_center_sum[assigned_index, :] + data_points[i, :]
        #     associated_center_count[assigned_index] = associated_center_count[assigned_index] + 1
            
        # # print('sum:', associated_center_sum)
        # # print('count:', associated_center_count)
            
        # for j in range(k):
        #     if associated_center_count[j] == 0:
        #         continue
        #     else:
        #         center_points[j, :] = associated_center_sum[j, :] / associated_center_count[j]

        gc.collect()

        step += 1
    return center_points
