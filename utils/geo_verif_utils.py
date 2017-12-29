import cv2
import numpy as np
from skimage.measure import ransac as _ransac
from skimage.transform import AffineTransform
import matplotlib.pyplot as plt


def get_matches(des1, des2, method='BRUTE_FORCE', is_binary_descriptor=False):
    # create BFMatcher object
    if method == 'BRUTE_FORCE':
        # BEWARE that, distance is different for binary descriptor and float descriptor. See http://answers.opencv.org/question/59996/flann-error-in-opencv-3/
        if is_binary_descriptor: # ORB, BRIEF, BRISK
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else: # SIFT, SURF
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
    elif method == 'FLANN':
        if is_binary_descriptor:
            raise "Not supported yet"
        else:
            # FLANN parameters
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params,search_params)

        matches = flann.knnMatch(des1.astype(np.float32),des2.astype(np.float32),k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        good = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                good.append(m)
                # matchesMask[i]=[1,0]
        matches = good
    return matches

def sift_to_rootsift(descs):
        if descs.dtype != np.float:
            descs = descs.astype(np.float32)
        # apply the Hellinger kernel by first L1-normalizing and taking the
        # square-root
        eps = 1e-10
        l1_norm = np.linalg.norm(descs, 1)
        descs /= (l1_norm + eps)
        descs = np.sqrt(descs)
        return descs

def get_ransac_inlier(kp1, kp2, des1, des2,
                      is_binary_descriptor=False,
                      matcher_method="FLANN",
                      min_samples=3,
                      residual_threshold=5,
                      max_trials=1000):
    """
    """
    des1 = sift_to_rootsift(des1)
    des2 = sift_to_rootsift(des2)
    matches = get_matches(des1, des2, matcher_method, is_binary_descriptor=False)
    if len(matches) < 5:
        # print("not enough matches: {}".format(len(matches)))
        return []

    # Perform geometric verification using RANSAC.
    # ransac_lib = "scikit"
    ransac_lib = "opencv"
    if ransac_lib == "opencv":
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        # print(matchesMask)
        inlier_idxs = np.nonzero(matchesMask)[0]
    elif ransac_lib == "scikit":
        locations_1_to_use = []
        locations_2_to_use = []
        for match in matches:
            locations_1_to_use.append((kp1[match.queryIdx].pt[1], kp1[match.queryIdx].pt[0])) # (row, col)
            locations_2_to_use.append((kp2[match.trainIdx].pt[1], kp2[match.trainIdx].pt[0]))

        locations_1_to_use = np.array(locations_1_to_use)
        locations_2_to_use = np.array(locations_2_to_use)
        _, inliers = _ransac(
        (locations_1_to_use, locations_2_to_use),
        AffineTransform,
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        max_trials=max_trials)

        inlier_idxs = np.nonzero(inliers)[0]

    inlier_match = []
    for idx in inlier_idxs:
        inlier_match.append(matches[idx])

    print("num kp1: {}, kp2: {}, match: {}, inlier: {}".format(len(kp1), len(kp2), len(matches), len(inlier_match)))
    return inlier_match

def draw_ransac(img1, img2, kp1, kp2, des1, kes2,
                is_binary_descriptor=False,
                match_method='BRTUE_FORCE',
                min_samples=3,
                residual_threshold=5,
                max_trials=1000,
                description=None):
    inlier_match = get_ransac_inlier(kp1, kp2, des1, kes2, is_binary_descriptor, match_method, min_samples, residual_threshold, max_trials)
    # Does ransac consider size and orientation of keypoints?
    ransac_img = cv2.drawMatches(img1, kp1, img2, kp2, inlier_match, None, flags=0)

    ransac_img = cv2.cvtColor(ransac_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(16, 12))
    plt.title(description)
    plt.imshow(ransac_img)
    plt.show()
    return len(inlier_match)


