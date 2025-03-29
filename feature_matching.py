import cv2
import numpy as np
from tkinter import Tk, filedialog, Button, Label, OptionMenu, StringVar
from PIL import Image, ImageOps


def select_image():
    global img1, file_path1
    file_path1 = filedialog.askopenfilename()
    if file_path1:
        img1 = Image.fromarray(cv2.imread(file_path1, cv2.IMREAD_GRAYSCALE))
        img1 = np.array(ImageOps.pad(img1, size=(512, 512), color=0))
        label_img1.config(text="Image 1: {}".format(file_path1.split('/')[-1]))


def select_image_2():
    global img2, file_path2
    file_path2 = filedialog.askopenfilename()
    if file_path2:
        img2 = Image.fromarray(cv2.imread(file_path2, cv2.IMREAD_GRAYSCALE))
        img2 = np.array(ImageOps.pad(img2, size=(512, 512), color=0))
        label_img2.config(text="Image 2: {}".format(file_path2.split('/')[-1]))


def feature_matching():
    if img1 is None or img2 is None:
        return

    # Get the selected feature matching method from the dropdown menu
    method = selected_method.get()

    # Initialize the feature detector
    if method == "ORB":
        detector = cv2.ORB_create()
    elif method == "SIFT":
        detector = cv2.SIFT_create()
    elif method == "SURF":
        detector = cv2.xfeatures2d.SURF_create()

    # Detect keypoints and descriptors
    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
    
    descriptors1 = descriptors1.astype(np.uint8)
    descriptors2 = descriptors2.astype(np.uint8)
    
    # Brute force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # print(descriptors1, descriptors1.dtype)
    # print(descriptors2, descriptors2.dtype)
    # if method == "ORB":
        # matches = bf.match(descriptors1, descriptors2)
        # # Sort matches by distance
        # matches = sorted(matches, key=lambda x: x.distance)
        # # Draw top 50 matches
        # img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None,
                                  # flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # elif method == "SIFT":
        # descriptors1 = descriptors1.astype(np.uint8)
        # descriptors2 = descriptors2.astype(np.uint8)

        # matches = bf.knnMatch(descriptors1,descriptors2,k=2)
        # # Apply ratio test
        # good = []
        # for m,n in matches:
            # if m.distance < 0.75*n.distance:
                # good.append([m])

        # # cv.drawMatchesKnn expects list of lists as matches.
        # img3 = cv.drawMatchesKnn(img1,keypoints1,img2,keypoints2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # FLANN_INDEX_LSH = 6
    # index_params= dict(algorithm = FLANN_INDEX_LSH,
                       # table_number = 6, # 12
                       # key_size = 12,     # 20
                       # multi_probe_level = 1) #2
                   
    # # FLANN parameters
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50)   # or pass empty dictionary
     
    # flann = cv.FlannBasedMatcher(index_params,search_params)
     
    # matches = flann.knnMatch(des1,des2,k=2)
     
    # # Need to draw only good matches, so create a mask
    # matchesMask = [[0,0] for i in range(len(matches))]
     
    # # ratio test as per Lowe's paper
    # for i,(m,n) in enumerate(matches):
        # if m.distance < 0.7*n.distance:
            # matchesMask[i]=[1,0]
     
    # draw_params = dict(matchColor = (0,255,0),
                       # singlePointColor = (255,0,0),
                       # matchesMask = matchesMask,
                       # flags = cv.DrawMatchesFlags_DEFAULT)
     
    # img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

    matches = bf.match(descriptors1, descriptors2)
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw top 50 matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    # Show the result
    cv2.imshow('Feature Matching', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Tkinter GUI setup
root = Tk()
root.title('Feature Matching')

img1 = None
img2 = None

# Buttons to select images
btn_select_image1 = Button(root, text="Select Image 1", command=select_image)
btn_select_image1.pack()

label_img1 = Label(root, text="Image 1: Not selected")
label_img1.pack()

btn_select_image2 = Button(root, text="Select Image 2", command=select_image_2)
btn_select_image2.pack()

label_img2 = Label(root, text="Image 2: Not selected")
label_img2.pack()

# Dropdown menu for feature matching method
selected_method = StringVar(root)
selected_method.set("ORB")  # Default value

methods = ["ORB", "SIFT", "SURF"]
method_menu = OptionMenu(root, selected_method, *methods)
method_menu.pack()

# Button to start feature matching
btn_match_features = Button(root, text="Match Features", command=feature_matching)
btn_match_features.pack()

root.mainloop()
