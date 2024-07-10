# Color Detection YOLOv3

## Project Overview

This project focuses on color detection of objects detected in images using the YOLOv3 model. After detecting objects in the frame, clustering techniques are applied to predict the color shades of these objects within bounding boxes.

### Key Features

- **Object Detection:** Implemented YOLOv3 model for accurate object detection in images.
- **Color Recognition:** Utilized optimized K-means clustering algorithm for predicting color shades of detected objects.
- **Accuracy Metrics:**
  - Achieved 97% accuracy with YOLOv3 for object detection.
  - Attained 94% accuracy with K-means clustering for color recognition.

### Project Details

- Developed a model for detecting objects and identifying color shades in images using deep learning and machine learning techniques.
- The project's detailed research findings are available in [research_paper.pdf](link_to_paper.pdf), providing a deep dive into the methodologies, experimental setup, and conclusions drawn. [Read the research paper](https://publications.eai.eu/index.php/IoT/article/view/5495).


## Website

The project is also available as a web application. Visit the [Color Detection Web App](https://color-detection.streamlit.app/) (currently under sleep mode).

## Repository Structure

- `code/`: Contains Python scripts for YOLOv3 model and K-means clustering.
- `data/`: Sample images and datasets used for training and testing.
- `research_paper.pdf`: Detailed research paper documenting methods and results.

## Installation Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/Color_Detection_YOLO3.git
   cd Color_Detection_YOLO3
