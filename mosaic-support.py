#####################################################################

# Example : real-time mosaicking - supporting functionality

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Acknowledgements: bmhr46@durham.ac.uk (2016/17);
# Marc Pare, code taken from:
# https://github.com/marcpare/stitch/blob/master/crichardt/stitch.py

# no claims are made that these functions are bug free

# Copyright (c) 2017 Dept. Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

# Takes an image and a Hessian threshold value and
# returns the SURF features points (kp) and descriptors (des) of image
# (for SURF features - Hessian threshold of typically 400-1000 can be used)

def getFeatures(img, thres):
    surf = cv2.xfeatures2d.SURF_create(thresh)
    kp, des = surf.detectAndCompute(img,None)
    return kp, des

#####################################################################

# Performs FLANN-based feature matching of descriptor from 2 images
# returns 'good matches' based on their distance
# typically number_of_checks = 50, match_ratio = 0.7

def matchFeatures(des1, des2, number_of_checks, match_ratio):
    index_params = dict(algorithm = 0, trees = 1) #FLANN_INDEX_KDTREE = 0
    search_params = dict(checks = number_of_checks)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    matchesMask = [[0,0] for i in range(len(matches))]
    good_matches = [];
    for i,(m,n) in enumerate(matches):
        if m.distance < match_ratio*n.distance:   #filter out 'bad' matches
            matchesMask[i]=[1,0];
            good_matches.append(m);
    return good_matches

#####################################################################

# Computes and returns the homography matrix relating the two images

def computeHomography(kp1, kp2, good_matches):
    #compute the transformation
    pts1 = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2);
    pts2 = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2);
    homography, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)      #using RANSAC to find homography
    return homography, mask

#####################################################################

# Calculates the required size for the mosaic

def calculate_size(size_image1, size_image2, homography):

    (h1, w1) = size_image1[:2]
    (h2, w2) = size_image2[:2]

    #remap the coordinates of the projected image onto the panorama image space
    top_left = np.dot(homography,np.asarray([0,0,1]))
    top_right = np.dot(homography,np.asarray([w2,0,1]))
    bottom_left = np.dot(homography,np.asarray([0,h2,1]))
    bottom_right = np.dot(homography,np.asarray([w2,h2,1]))

    #normalize
    top_left = top_left/top_left[2]
    top_right = top_right/top_right[2]
    bottom_left = bottom_left/bottom_left[2]
    bottom_right = bottom_right/bottom_right[2]

    pano_left = int(min(top_left[0], bottom_left[0], 0))
    pano_right = int(max(top_right[0], bottom_right[0], w1))
    W = pano_right - pano_left

    pano_top = int(min(top_left[1], top_right[1], 0))
    pano_bottom = int(max(bottom_left[1], bottom_right[1], h1))
    H = pano_bottom - pano_top

    size = (W, H)

    # offset of first image relative to panorama
    X = int(min(top_left[0], bottom_left[0], 0))
    Y = int(min(top_left[1], top_right[1], 0))
    offset = (-X, -Y)

    return (size, offset)

#####################################################################

# Merges two images given the homography and translation offset between them

def merge_images(image1, image2, homography, size, offset, keypoints):
    (h1, w1) = image1.shape[:2]
    (h2, w2) = image2.shape[:2]

    panorama = np.zeros((size[1], size[0], 3), np.uint8)

    (ox, oy) = offset

    translation = np.matrix([[1.0, 0.0, ox],
                             [0,   1.0, oy],
                             [0.0, 0.0, 1.0]])

    homography = translation * homography

    # draw the transformed image2
    cv2.warpPerspective(image2, homography, size, panorama)

    #masking
    A = cv2.cvtColor(panorama[oy:h1+oy, ox:ox+w1], cv2.COLOR_RGB2GRAY)
    B = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    AandB = cv2.bitwise_and(A, B)
    overlap_area_mask = cv2.threshold(AandB, 1, 255, cv2.THRESH_BINARY)[1]

    As_nonoverlap_area_mask = cv2.threshold(A, 1, 255, cv2.THRESH_BINARY)[1] - overlap_area_mask
    Bs_nonoverlap_area_mask = cv2.threshold(B, 1, 255, cv2.THRESH_BINARY)[1] - overlap_area_mask

    ored = cv2.bitwise_or(panorama[oy:h1+oy, ox:ox+w1], image1, mask=(Bs_nonoverlap_area_mask-As_nonoverlap_area_mask))
    oredcorrect = cv2.subtract(ored, panorama[oy:h1+oy, ox:ox+w1])

    panorama[oy:h1+oy, ox:ox+w1] = cv2.add(panorama[oy:h1+oy, ox:ox+w1], oredcorrect)

    return panorama

#####################################################################
