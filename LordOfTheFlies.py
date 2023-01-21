# -*- coding: utf-8 -*-

# function to calculate area with aruco of segmented image

# import libs
import streamlit as st
import skimage.io as io
import skimage
import cv2

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import filters
from rembg import remove
from skimage import img_as_ubyte, data, filters, measure, morphology

from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
import plotly
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import imutils
from skimage.color import label2rgb, rgb2gray

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

def calc_area(image):
  
  #segment the image using k-means:
  segmented_image, labels, centers = segment_image_kmeans(image, k=3, attempts=10)
  #claculate AruCo ratio:
  pixel_cm_ratio = getPxCmRatio(image) 
  #remove AruCo an background:
  img_na = removeAruco(segmented_image)
  #converting to grayscale:
  img = skimage.color.rgb2gray(img_na[:,:,:3])

  # Binary image, post-process the binary mask and compute labels
  smooth = skimage.filters.gaussian(img, sigma=8, mode='constant', cval=0.0) 
  threshold = filters.threshold_otsu(smooth)
  mask = smooth < threshold
  mask = morphology.remove_small_objects(mask, min_size = 3000)
  mask = morphology.remove_small_holes(mask, 20)
  labels = measure.label(mask)
  
  #Calculate area for each label:
  props = measure.regionprops(labels, img)
  properties = ['area']
  total_area = 0

  for index in range(1, labels.max()):
    current_area = (getattr(props[index], 'area')*pixel_cm_ratio) #get area in cm^2 of each label
    total_area += current_area #add the area to the total area
  
  return (total_area/(labels.max()-1))

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
    st.markdown('Welcome!\n In this app you can calculate the horizontal area of your BSF larvas and (hopefully) get an estimation of their weight. \nIn order to insure the accuracy of the results, make sure you upload an image that meets the following requirements (example below): \n1. The larvas are placed in a Petri dish on a blank white page with Aruco marker 5X5-50. \n2. The larvas are seperated from one another and the Petri dish does not cover the Aruco marker. \n3. If necessary, use a flash to light the image and make larvas different from the background. \n4. The image is in jpg/jpeg format. \nPlease notice: Due to time limitations, the app is yet to be in its final version. For now, it works well with prepupals and pupas. \nIn the future, we hope to make every it relevant to any larva stage.')
    
    
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
    stage = st.sidebar.selectbox('Larva stage',
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
    
    #segment the image using k-means:
    segmented_image, labels, centers = segment_image_kmeans(image, k=3, attempts=10)
    
    #claculate AruCo ratio:
    pixel_cm_ratio = getPxCmRatio(image) 
    
    #remove AruCo an background:
    img_na = removeAruco(segmented_image)
    
    #converting to grayscale:
    img = skimage.color.rgb2gray(img_na[:,:,:3])
    
    # Binary image, post-process the binary mask and compute labels
    smooth = skimage.filters.gaussian(img, sigma=8, mode='constant', cval=0.0) 
    threshold = filters.threshold_otsu(smooth)
    mask = smooth < threshold
    mask = morphology.remove_small_objects(mask, min_size = 3000)
    mask = morphology.remove_small_holes(mask, 100)
    labels = measure.label(mask)
    
    fig = px.imshow(img, binary_string=True)
    fig.update_traces(hoverinfo='skip') # hover is only for label info
    
    #Calculate area for each label:
    props = measure.regionprops(labels, img)
    properties = ['area']
    total_area = 0
    
    # For each label, add a filled scatter trace for its contour,
    # and display the properties of the label in the hover of this trace.
    for index in range(1, labels.max()):
      label_i = props[index].label
      contour = measure.find_contours(labels == label_i, 0.5)[0]
      y, x = contour.T
      hoverinfo = ''
      current_area = (getattr(props[index], 'area')*pixel_cm_ratio) #get area in cm^2 of each label
      total_area += current_area #add the area to the total area
    
      for prop_name in properties:
        hoverinfo += f'<b>{prop_name}: {(getattr(props[index], prop_name)*pixel_cm_ratio):.2f}</b><br>'
      fig.add_trace(go.Scatter(
        x=x, y=y, name=label_i,
        mode='lines', fill='toself', showlegend=False,
        hovertemplate=hoverinfo, hoveron='points+fills'))
    average_area = total_area/(labels.max()-1) #calculate the average area of the larva by divide the total area by the number of labels
    plotly.io.show(fig)
    # print('Average area of Larva:{a} cm^2'.format(a = average_area))

    st.markdown('''
          ##  The average area of the larvas in cm\N{SUPERSCRIPT TWO}: 
                ''')
    st.text(average_area)
    
    if stage == 'Pre-pupal':
      average_weight = average_area*0.06637753102
    elif stage == 'Pupa':
      average_weight = average_area*0.06628851182
    
    st.markdown('''
          ##  The average weight of the karvas in cm\N{SUPERSCRIPT THREE}: 
                ''')
    st.text(average_weight)
