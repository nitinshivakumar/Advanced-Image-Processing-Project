# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 3. Not following the project guidelines will result in a 10% reduction in grades
# 4 . If you want to show an image for debugging, please use show_image() function in helper.py.
# 5. Please do NOT save any intermediate files in your final submission.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import array as arr


def parse_args():
    parser = argparse.ArgumentParser(description="cse 573 homework 4.")
    parser.add_argument(
        "--input_path", type=str, default="data/images_panaroma",
        help="path to images for panaroma construction")
    parser.add_argument(
        "--output_overlap", type=str, default="./task2_overlap.txt",
        help="path to the overlap result")
    parser.add_argument(
        "--output_panaroma", type=str, default="./task2_result.png",
        help="path to final panaroma image ")

    args = parser.parse_args()
    return args

def stitch(inp_path, imgmark, N=4, savepath=''): 
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgs = []
    imgpath = [f'{inp_path}/{imgmark}_{n}.png' for n in range(1,N+1)]
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"

    import imutils

    over_lap = overlap_arr(imgs)

    sift = cv2.SIFT_create()

    while len(imgs) > 1:
        
        match_dict = {'M' : [], 'image_1' : [], 'image_2' : [],'length_match' : [], 'side' : []}
        for i in range(len(imgs)):
            i = 0
            match_dict = {'M' : [], 'image_1' : [], 'image_2' : [],'length_match' : [], 'side' : []}
            for j in range(i+1, len(imgs)):
                keypoints1, descriptors1 = sift.detectAndCompute(imgs[i], None)
                keypoints2, descriptors2 = sift.detectAndCompute(imgs[j], None)
                x1 = compute_coordinates(keypoints1)
                x2 = compute_coordinates(keypoints2)
                match_1 = feature_matching(descriptors1, descriptors2)
                match_2 = feature_matching(descriptors2, descriptors1)
                new1 = len(match_1)/len(descriptors2)
                new2 = len(match_2)/len(descriptors1)
                src_pts_1, dst_pts_1 = compute_source(match_1, keypoints1, keypoints2)
                src_pts_2, dst_pts_2 = compute_source(match_2, keypoints2, keypoints1)
                M1, rem1 = cv2.findHomography(src_pts_1, dst_pts_1, cv2.RANSAC, 9.0)
                M2, rem2 = cv2.findHomography(src_pts_2, dst_pts_2, cv2.RANSAC, 9.0)
                
                if (x1 < x2):
                    match_dict['M'].append(M1)
                    match_dict['image_1'].append(i)
                    match_dict['image_2'].append(j)
                    match_dict['length_match'].append(new1)
                else:
                    match_dict['M'].append(M2)
                    match_dict['image_1'].append(j)
                    match_dict['image_2'].append(i)
                    match_dict['length_match'].append(new2)

            length_match = match_dict['length_match']
            max_value = max(length_match)
            max_index = length_match.index(max_value)
            
            first_image = match_dict['image_1'][max_index]
            second_image = match_dict['image_2'][max_index]

            width = imgs[second_image].shape[1] + imgs[first_image].shape[1]
            height = imgs[second_image].shape[0] + imgs[first_image].shape[0]


            result = cv2.warpPerspective(imgs[first_image], match_dict['M'][max_index], (width, height))

            height, width, breadth = imgs[second_image].shape
            for i in range(0, height):
                for j in range(0, width):
                    if sum(list(imgs[second_image][i, j])) and not sum(list(result[i, j])) :
                        result[i, j] = imgs[second_image][i, j]
            
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(c)
            result = result[y:y+h, x:x + w]
            import utils
            if first_image < second_image:
                imgs.pop(first_image)
                imgs.pop(second_image-1)
                imgs.insert(0, result)
            else:
                imgs.pop(second_image)
                imgs.pop(first_image-1)
                imgs.insert(0, result)
            
            if len(imgs) == 1:
                cv2.imwrite('task2_result.png',result) 
                return over_lap
            i = 0
             
    return over_lap

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def feature_matching(f1, f2):
    list_match = []
    for i, j in enumerate(f1):
        distances = np.linalg.norm(f2 - j, axis=1)
        val_sort = np.argsort(distances)
        output1 = val_sort[0]
        output2 = val_sort[1]
        if distances[output1] < 0.75 * distances[output2]:
            list_match.append((i, output1, 0, distances[output1]))
    return list_match

def overlap_arr(imgs):
    no_image = len(imgs)
    overlap_array = np.eye(no_image, dtype=int)
    sift = cv2.SIFT_create()
    for i in range(no_image):
        for j in range(i+1,no_image):
            keypoints1, descriptors1 = sift.detectAndCompute(imgs[i], None)
            keypoints2, descriptors2 = sift.detectAndCompute(imgs[j], None)
            match = feature_matching(descriptors1, descriptors2)
            denom = min(len(descriptors1), len(descriptors2))
            percentage_overlap = len(match)/denom
            if percentage_overlap > 0.2:
                overlap_array[i, j] = 1
                overlap_array[j, i] = 1
            else:
                pass
    return overlap_array


def compute_source(match, keypoints1, keypoints2):
    if len(match)%2 != 0:
        match.pop()
    src_pts = np.float32([keypoints1[k[0]].pt for k in match])
    dst_pts = np.float32([keypoints2[k[1]].pt for k in match])
    return src_pts, dst_pts

def compute_coordinates(keypoints):
    keypoint_coordinates = [(keypoint.pt[0], keypoint.pt[1]) for keypoint in keypoints]
    mean_x = sum(x for x, y in keypoint_coordinates) / len(keypoint_coordinates)
    mean_y = sum(y for x, y in keypoint_coordinates) / len(keypoint_coordinates)
    return mean_x

    
if __name__ == "__main__":
    #task2
    args = parse_args()
    overlap_arr = stitch(args.input_path, 't2', N=4, savepath=f'{args.output_panaroma}')
    with open(f'{args.output_overlap}', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    
