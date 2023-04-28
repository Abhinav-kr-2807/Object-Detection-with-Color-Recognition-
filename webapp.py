import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
import numpy as np
from numpy import expand_dims
from keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from kmeans import plotting, color_detector
import time
import urllib

st.set_page_config(
    page_title="Color Detection with YOLO",
    page_icon="🎨"
)

st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://github.com/aadarsh1810/Color_Detection_YOLO3/blob/main/black-background.gif?raw=true");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


# This class returns the R, G, B values of the dominant colours
def relu(x):
	return max(0, x)
    

# This class predicts the bounding box in an image
class BoundBox():
    
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1
        
    def get_label(self):
        
        if self.label == -1:
            self.label = np.argmax(self.classes)
            
        return self.label
    
    def get_score(self):
        
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score
  
# Sigmoid function
        
def sigmoid(x):
    
    return 1. / (1. + np.exp(-x))

# Docoding the net output of the model

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    boxes = []
    
    netout[..., :2] = sigmoid(netout[..., :2])
    netout[..., 4:] = sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh
    
    for i in range(grid_h * grid_w):
        
        row = i / grid_w
        col = i % grid_w
        
        for b in range(nb_box):
            
            objectness = netout[int(row)][int(col)][b][4]
            
            if objectness.all() <= obj_thresh:
                continue
            
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w
            y = (row + y) / grid_h
            w = anchors[2 * b + 0] * np.exp(w) / net_w
            h = anchors[2 * b + 1] * np.exp(h) / net_h
            classes = netout[int(row)][int(col)][b][5:]
            box = BoundBox(x - w/2, y - h/2, x + w/2, y + h/2, objectness, classes)
            boxes.append(box)
            
    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    
    new_w, new_h = net_w, net_h
    
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
        
def interval_overlap(interval_a, interval_b):
    
    x1, x2 = interval_a
    x3, x4 = interval_b
    
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3
        
def bbox_iou(box1, box2):
    
    intersect_w = interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_h * intersect_w
    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    union = w1 * h1 + w2 * h2 - intersect
    return float(intersect) / union

def do_nms(boxes, nms_thresh):
    
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    
    for c in range(nb_class):
        
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        
        for i in range(len(sorted_indices)):
            
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0:
                continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
                    
def load_image_pixels(image_array, shape):
    
    width, height = image_array.shape[1], image_array.shape[0]
    image_array = cv2.resize(image_array, (shape[0], shape[1]))
    # image = load_img(filename)
    # width, height = image.size
    # image = load_img(filename, target_size = shape)
    # image = img_to_array(image)
    image = image_array.astype('float32')
    image = image / 255.0
    image = expand_dims(image, 0)
    return image, width, height

def get_boxes(boxes, labels, thresh):
    
    v_boxes, v_labels, v_scores = list(), list(), list()
    
    for box in boxes:
        
        for i in range(len(labels)):
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i] * 100)
                
    return v_boxes, v_labels, v_scores

def draw_boxes(img_array, v_boxes, v_labels, v_scores):

    data = img_array
    st.success("Here it is")
    pyplot.imshow(data)
    ax = pyplot.gca()
    
    for i in range(len(v_boxes)):
        
        box = v_boxes[i]
        y1, x1, y2, x2, = relu(box.ymin), relu(box.xmin), relu(box.ymax), relu(box.xmax)
        print(y1, x1, y2, x2)
        width, height = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), width, height, fill = False, color = 'black')
        ax.add_patch(rect)
        label = "%s (%.3f)" % (v_labels[i], v_scores[i]) + " "
        
        name = color_detector(data[y1:y1+height, x1:x1+width])
        name = name.loc[0, 'Color']
        
        label = "Color: " + name + " " + label

        pyplot.text(x1, y1-1, label, color = 'black', fontsize = 6)

    pyplot.axis('off')
    st.pyplot(pyplot.show())
    pyplot.clf()

    for i in range(len(v_boxes)):
        
        box = v_boxes[i]
        y1, x1, y2, x2, = relu(box.ymin), relu(box.xmin), relu(box.ymax), relu(box.xmax)
        print(y1, x1, y2, x2)
        width, height = x2 - x1, y2 - y1

        label = "%s (%.3f)" % (v_labels[i], v_scores[i]) + " "
        st.header(label.upper())
        plotting(data[y1:y1+height, x1:x1+width])
 
def merge_functions(img_array):

    model = load_model("model.h5")
    model.compile(optimizer='adam', loss={'yolo_loss': lambda y_true, y_pred: y_pred})

    
    input_w, input_h = 416, 416
    image, image_w, image_h = load_image_pixels(img_array, (input_w, input_h))
    yhat = model.predict(image)

    print([a.shape for a in yhat])
    anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

    class_threshold = 0.6

    boxes = list()

    for i in range(len(yhat)):
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
    
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

    do_nms(boxes, 0.5)

    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

    for i in range(len(v_boxes)):
        print(v_labels[i], v_scores[i])

    draw_boxes(img_array, v_boxes, v_labels, v_scores)


tab1, tab2, tab3 = st.tabs(["Color Detector", "Model Details", "About"])

with tab1:
    st.title("Object & Color Detection")
    st.header("Upload Image to detect object and its color")
    uploaded_img = st.file_uploader("Upload Image", type = ['jpeg', 'jpg'], accept_multiple_files=False)
    if st.button(label = "Predict"):
        if uploaded_img is None:
            st.warning("Please upload image first.")

        elif uploaded_img is not None:
            image = Image.open(uploaded_img)
            img_array = np.array(image)
            
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)

            for percent_complete in range(100):
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1, text=progress_text)

            merge_functions(img_array)


with tab2:
    st.title("Object Labels used for training model:")
    st.write("person, bicycle, car, motorbike, aeroplane, bus, train, truck,boat, traffic light, fire hydrant, stop sign, parking meter, bench,\n bird, cat, dog, horse, sheep, cow, elephant, bear, zebra,giraffe,\n backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard,\n sports ball, kite, baseball bat, baseball glove, skateboard, surfboard,\n tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana,\n apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake,\n chair, sofa, pottedplant, bed, diningtable, toilet, tvmonitor, laptop, mouse,\n remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator,\n book, clock, vase, scissors, teddy bear, hair drier, toothbrush")
    st.title("Project Framework")
    st.image("https://github.com/aadarsh1810/Color_Detection_YOLO3/blob/main/WORKFLOW.png?raw=true", use_column_width=True)
    st.title("YOLO Architecture")
    st.image("https://github.com/aadarsh1810/Color_Detection_YOLO3/blob/main/yolov3.png?raw=true")
    st.title("Color Detection Flowchart")
    st.image("https://github.com/aadarsh1810/Color_Detection_YOLO3/blob/main/Kmeans.jpg?raw=true", use_column_width=True)

with tab3:
    st.title("Our GitHub Page")
    st.write("https://github.com/aadarsh1810/Color_Detection_YOLO3", unsafe_allow_html=True)
    st.title("Developer")
    col1, col2, col3 = st.columns(3)
    with col1:
        #st.image("https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/roshan.png?raw=true", use_column_width=True)
        st.markdown("### Aadarsh Nayyer \n II year BTech AIML, SIT")

    with col2:
        #st.image("https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/roshan.png?raw=true", use_column_width=True)
        st.markdown("### Abhinav Kumar \n II year BTech AIML, SIT")


    with col3:
        #st.image("https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/roshan.png?raw=true", use_column_width=True)
        st.markdown("### Aayush Rajput \n II year BTech AIML, SIT")

   
