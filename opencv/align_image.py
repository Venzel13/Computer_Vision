import numpy as np
import cv2 as cv


def align_image(
    query_path, train_path, colormap=cv.COLOR_RGB2YUV, num_features=300, n_best=30
):
    """
    Align (e.g. rotated, squeezed) distorted/train image correspond to original/query one.
    
    Parameters
    ----------
    @ query_path: str
        The path to the original image.
    @ train_path: str
        The path to the image that needs to be transformed
    @ colormap: int, optional
        The converter from RGB to the another format. Default is YUV
    @ num_features: int, optional
        The quantity of keypoints, which should be found be descriptors. Default is 300.
    @ n_best: int, optional
        number of the most closest features (keypoints) that will be used for transformation. Default is 30.
    
    Return
    ------
    aligned_image: ndarray
        The transformed image
    """

    query = read_image(query_path)
    train = read_image(train_path)
    gray_query = extract_gray_channel(query, colormap)
    gray_train = extract_gray_channel(train, colormap)

    kp_query, desc_query = find_keypoints(gray_query, num_features)
    kp_train, desc_train = find_keypoints(gray_train, num_features)

    matches = match_keypoints(desc_query, desc_train, n_best)
    coord_query, coord_train = get_coord(matches, kp_query, kp_train, n_best)
    M = find_transform(coord_query, coord_train)

    h, w, _ = query.shape
    aligned_image = cv.warpPerspective(train, M, (w, h))

    return aligned_image


def read_image(image_path):
    image = cv.imread(image_path)
    return image


def extract_gray_channel(image, converter):
    converted_image = cv.cvtColor(image, converter)
    gray_channel = converted_image[:, :, 0]
    return gray_channel


def find_keypoints(image, num_features):
    "image=gray_channel"
    descriptor = cv.ORB_create(num_features)
    keypoint, descriptor = descriptor.detectAndCompute(image, None)
    return keypoint, descriptor


def match_keypoints(desc_query, desc_train, n_best):
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_query, desc_train)
    matches = sorted(matches, key=lambda x: x.distance)  # sort keypoints by a distance
    matches = matches[:n_best]  # get a subset of the closest keypoints

    return matches


def get_coord(matches, kp_query, kp_train, n_best):
    coord_query = np.zeros(
        [n_best, 2], dtype=np.float32
    )  # placeholder for keypoint's coordinates
    coord_train = np.zeros([n_best, 2], dtype=np.float32)

    for i, match in enumerate(matches):
        coord_query[i, :] = kp_query[
            match.queryIdx
        ].pt  # extract coordinates of the keypoints
        coord_train[i, :] = kp_train[match.trainIdx].pt

    return coord_query, coord_train


def find_transform(coord_query, coord_train):
    M, _ = cv.findHomography(coord_train, coord_query, cv.RANSAC)
    return M