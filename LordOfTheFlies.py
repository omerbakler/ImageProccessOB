# -*- coding: utf-8 -*-

# function to calculate area with aruco of segmented image

# import libs
from skimage import measure, io, img_as_ubyte, morphology, util, color
from skimage.color import label2rgb, rgb2gray
import numpy as np
import pandas as pd
import cv2
import imutils
import streamlit as st

def getPxCmRatio(image):
  
  # Load Aruco detector
  parameters = cv2.aruco.DetectorParameters_create()
  aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

  # Setting the corners list of the aruco marker to global to use it later
  global corners

  # Get Aruco marker
  corners, _, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

  # Aruco Area
  aruco_area = cv2.contourArea (corners[0])

  # Get Pixel to cm ratio
  pixel_cm_ratio = 5*5 / aruco_area # since the AruCo is 5*5 cm, so we devide 25 cm*cm by the number of pixels
  
  return pixel_cm_ratio

def removeAruco(image):
  no_bg = remove(image)
  # Finding the values of the aruco corners(because it's not an exact square) and asigning
  if corners[0][0][3][1] < corners[0][0][0][1]:
    Up = int(corners[0][0][3][1])
  else:
    Up = int(corners[0][0][0][1])
  
  if corners[0][0][1][1] > corners[0][0][2][1]:
    Down = int(corners[0][0][1][1])
  else:
    Down = int(corners[0][0][2][1])

  if corners[0][0][2][0] < corners[0][0][3][0]:
    Left = int(corners[0][0][2][0])
  else:
    Left = int(corners[0][0][3][0])

  if corners[0][0][0][0] > corners[0][0][1][0]:
    Right = int(corners[0][0][0][0])
  else:
    Right = int(corners[0][0][1][0])

  # Turning the pixels values of the square to white 
  image_nA = no_bg.copy()
  image_nA[Up : Down , Left : Right] = no_bg[0,0]

  # Creating an image out of the previously modified array
  img_nA = Image.fromarray(image_nA)

  return image_nA

def segment_image_kmeans(img, k=3, attempts=10): 
  
  # Convert MxNx3 image into Kx3 where K=MxN
  pixel_values  = img.reshape((-1,3))  #-1 reshape means, in this case MxN
  
  #We convert the unit8 values to float as it is a requirement of the k-means method of OpenCV
  pixel_values = np.float32(pixel_values)
  
  # define stopping criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
  _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    
  # convert back to 8 bit values
  centers = np.uint8(centers)
  
  # flatten the labels array
  labels = labels.flatten()
  
  # convert all pixels to the color of the centroids
  segmented_image = centers[labels.flatten()]
    
  # reshape back to the original image dimension
  segmented_image = segmented_image.reshape(img.shape)
    
  return segmented_image, labels, centers

def aruco_calc(img,index,k,attempts):
  # Load Aruco detector
  parameters = cv2.aruco.DetectorParameters_create()
  aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

  # Get Aruco marker
  corners, _, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

  # Aruco Area
  aruco_area = cv2.contourArea(corners[0])

  # Pixel to cm ratio
  pixel_cm_ratio = 5*5 / aruco_area # since the AruCo is 5*5 cm, so we devide 25 cm*cm by the number of pixels
  
  # segment the image using k-means
  segmented_kmeans, labels, centers = segment_image_kmeans(image, k, attempts)
  
  # copy source img
  img = image.copy()
  masked_image = img.copy()
  
  # convert to the shape of a vector of pixel values (like suits for kmeans)
  masked_image = masked_image.reshape((-1, 3))
  
  index_to_remove = index
  
  # color (i.e cluster) to exclude
  list_of_cluster_numbers_to_exclude = list(range(k)) # create a list that has the number from 0 to k-1
  list_of_cluster_numbers_to_exclude.remove(index_to_remove) # remove the cluster of leaf that we want to keep, and not black out
  for cluster in list_of_cluster_numbers_to_exclude:
    masked_image[labels == cluster] = [0, 0, 0] # black all clusters except cluster leaf_center_index
  
  # convert back to original shape
  masked_image = masked_image.reshape(img.shape)
  masked_image_grayscale = rgb2gray(masked_image)
  
  # count how many pixels are in the foreground and bg
  area_px_count = np.sum(np.array(masked_image_grayscale) >0)

  area_in_cm = area_px_count * pixel_cm_ratio

  return area_in_cm

# function to segment using k-means

def segment_image_kmeans(img, k=3, attempts=10): 

    # Convert MxNx3 image into Kx3 where K=MxN
    pixel_values  = img.reshape((-1,3))  #-1 reshape means, in this case MxN

    #We convert the unit8 values to float as it is a requirement of the k-means method of OpenCV
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    
    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()
    
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(img.shape)
    
    return segmented_image, labels, centers

# vars
DEMO_IMAGE = 'demo.jpg' # a demo image for the segmentation page, if none is uploaded

# main page
st.set_page_config(page_title='AruCo area calculator', layout = 'wide', initial_sidebar_state = 'auto')
st.title('Lord Of The Flies')

# side bar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] . div:first-child{
        width: 350px
    }
    
    [data-testid="stSidebar"][aria-expanded="false"] . div:first-child{
        width: 350px
        margin-left: -350px
    }    
    </style>
    
    """,
    unsafe_allow_html=True,


)

st.sidebar.title('Options')

# using st.cache so streamlit runs the following function only once, and stores in chache (until changed)
@st.cache()

# take an image, and return a resized that fits our page
def image_resize(image, width=None, height=None, inter = cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)
    
    else:
        r = width/float(w)
        dim = (width, int(h*r))
        
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    
    return resized

# add dropdown to select pages on left
app_mode = st.sidebar.selectbox('Navigate',
                                  ['About App', 'Calculate'])

# About page
if app_mode == 'About App':
    st.markdown('Welcome! \n
                  In this app you can calculate the horizontal area of your BSF larvas and (hopefully) get an estimation of their weight. \n
                  In order to insure the accuracy of the results, make sure you upload an image that meets the following requirements (example below): \n
                  1. The larvas are placed in a Petri dish on a blank white page with Aruco marker 5X5-50. \n
                  2. The larvas are seperated from one another and the Petri dish does not cover the Aruco marker. \n
                  3. If necessary, use a flash to light the image and make larvas different from the background. \n
                  4. The image is in jpg/jpeg format. \n
                  \n
                  Please notice: Due to time limitations, the app is yet to be in its final version. For now, it works well with prepupals and pupas. \n
                  In the future, we hope to make every it relevant to any larva stage.')
    
    
    # side bar
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] . div:first-child{
            width: 350px
        }

        [data-testid="stSidebar"][aria-expanded="false"] . div:first-child{
            width: 350px
            margin-left: -350px
        }    
        </style>

        """,
        unsafe_allow_html=True,


    )


# Run image
if app_mode == 'Calculate':
    
    st.sidebar.markdown('---') # adds a devider (a line)
    
    # side bar
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] . div:first-child{
            width: 350px
        }

        [data-testid="stSidebar"][aria-expanded="false"] . div:first-child{
            width: 350px
            margin-left: -350px
        }    
        </style>

        """,
        unsafe_allow_html=True,


    )

    # choosing a k value (either with +- or with a slider)
    k_value = st.sidebar.selectbox('Larva stage',
                                  ['5 days old', '7 days old', '13 days old', 'Pre-pupal', 'Pupa'])
    st.sidebar.markdown('---') # adds a devider (a line)
    
    attempts_value_slider = st.sidebar.slider('Number of attempts for k-means segmentation', value = 7, min_value = 1, max_value = 10) # slider example
    st.sidebar.markdown('---') # adds a devider (a line)
    
    # index_value = st.sidebar.slider('Index of object', value = 2, min_value = 1, max_value = 3) # slider example
    # st.sidebar.markdown('---') # adds a devider (a line)

    # read an image from the user
    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    # assign the uplodaed image from the buffer, by reading it in
    if img_file_buffer is not None:
        image = io.imread(img_file_buffer)
    else: # if no image was uploaded, then segment the demo image
        demo_image = DEMO_IMAGE
        image = io.imread(demo_image)

    # display on the sidebar the uploaded image
    st.sidebar.text('Original Image')
    st.sidebar.image(image)
    
    # call the function to calculate the area
    index_area = aruco_calc(image, index = index_value-1, k = k_value, attempts = attempts_value_slider)

    st.markdown('''
          ##  The area of your object in cm\N{SUPERSCRIPT TWO}: 
                ''')
    st.text(index_area)
