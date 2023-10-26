# Real Time Face Mask Detection
 
Real-Time Face Mask Detection is a computer vision project that leverages machine learning techniques to detect whether a person is wearing a face mask in real-time. This project uses popular deep learning frameworks like TensorFlow and OpenCV, making it a valuable resource for applications related to health and safety, such as monitoring mask compliance in public spaces. Transfer learning was applied by using a pre-trained model `MobileNetV2` to achieve reliable results.

## Table of Contents
- [Project Structure](#ProjectStructure)
- [Features](#Features)
- [Usage](#Usage)

## Project Structure
The project consists of the following key files:

   - `augmentation.py`: Given that the dataset is not large enough, more data was created using the `albumentation` library in this script.

   - `validate_images.py`: After data was created, this script was used to make sure that all the images are healthy and not defected.
     
   - `create_data.ipynb`: This notebook is used to convert the images into numpy arrays and normalize them, then save the data as a `.pkl` file.
     
   - `train_model.ipynb`: This notebook is used to train the face mask detector model.
     
   - `utils/model.png`: This is an image file that shows architecture of the model.

   - `utils/MobileNetV2.h5`: The model used to make predictions.

   - `video_cap.py`: This Python script uses your webcam to capture video and uses the trained model to detect and highlight faces with and without masks in real-time


## Features

   - Real-time face mask detection: Detects whether a person is wearing a face mask or not in real-time video streams.
     
   - High accuracy: Utilizes a pre-trained deep learning model to achieve reliable results.
     
   - Easy-to-use: Provides an easy way for running the detection and integrating the model into other projects.

## Usage

Once the project is set up, you can use the face mask detection feature. Here are the main usage instructions:

- Run the real-time face mask detection script:
   `python video_cap.py`

- A live video stream will open, and the application will start detecting faces and their mask-wearing status.

- To quit the application, press the 'q' key.
