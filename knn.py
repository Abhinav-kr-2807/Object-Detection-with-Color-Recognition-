# Load required libraries and image
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import cv2
import numpy as np
import pandas as pd
from skimage.segmentation import slic
from colordetect import ColorDetect
import ast
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

csv = pd.read_csv(r"https://raw.githubusercontent.com/Yadav-Roshan/Color_Detection_YOLO/main/colors_final.csv")

def getColorName(RGB):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(int(RGB[0])- int(csv.loc[i,"R"])) + abs(int(RGB[1])- int(csv.loc[i,"G"]))+ abs(int(RGB[2])- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"color_name"]
    return cname

def crop_image(img):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    
    h, w, e = img.shape

    # GrayScale image
    
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale_img = cv2.convertScaleAbs(grayscale_img)
    
    cropped_product = img.copy()
    # Erosion and contour detection
    border = cv2.dilate(grayscale_img, None, iterations=10)
    border = border - cv2.erode(border, None)
    contours, hierarchy = cv2.findContours(border,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        rect = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        
        # Consider countors covering at least 50% of the whole image
        if area > 0.5 * h * w:
            cropped_product = cropped_product[   
                rect[1]:rect[1]+rect[3],
                rect[0]:rect[0]+rect[2]
            ]
    
    return cropped_product

def color_detector(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image(img)
    my_image = ColorDetect(img)
    colors = my_image.get_color_count(color_format="rgb")

    keys = list(colors.keys())
    values = list(colors.values())
    sorted_value_index = np.argsort(values)[::-1]
    sorted_colours = {keys[i]: values[i] for i in sorted_value_index}

    keys_new = []
    for key in sorted_colours.keys():
        keys_new.append(ast.literal_eval(key))

    keys_new = tuple(keys_new)
    values_new = tuple(sorted_colours.values())

    keys_color = []
    for i in keys_new:
        keys_color.append(getColorName(i))
    
    R = [key[0] for key in keys_new]
    G = [key[1] for key in keys_new]
    B = [key[2] for key in keys_new]
    
    keys_color = tuple(keys_color)

    df = pd.DataFrame()
    df['Color'] = keys_color
    df['Percentage'] = values_new
    df['R'] = R
    df['G'] = G
    df['B'] = B
    
    return df

def plotting(img):
    plt.imshow(img)
    ax = plt.gca()
    plt.axis('off')
    st.pyplot(plt.show())
    plt.clf()

    df = color_detector(img)

    details, plots = st.columns([2, 1.5])
    details.dataframe(df)
    
    with plots:
        # Create a new figure and axis
        fig, ax = plt.subplots()

        # Set the axis limits
        ax.set_xlim([0, 0.5])
        ax.set_ylim([0, len(df)])

        # Add a rectangle for each row in the DataFrame
        for i, row in df.iterrows():
            color = (row['R']/255, row['G']/255, row['B']/255)
            rect = plt.Rectangle((0, len(df)-1-i), 1, 1, color=color)
            ax.add_patch(rect)

        # Show the plot
        plt.axis('off')
        st.pyplot(plt.show())

# uploaded_img = st.file_uploader("Upload Image", type = ['jpeg', 'jpg'], accept_multiple_files=False)
# if st.button(label = "Predict"):
#     if uploaded_img is not None:
#         image = Image.open(uploaded_img)
#         img_array = np.array(image)
#         plotting(img_array)