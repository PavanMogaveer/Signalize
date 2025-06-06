# Signalize

Signalize is an intelligent traffic sign recognition system that uses computer vision and machine learning techniques to detect and classify traffic signs from images and live video streams. The project aims to assist in improving road safety and enabling smart transportation systems by accurately identifying traffic signs in real-time.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Live Webcam Prediction](#live-webcam-prediction)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

The Signalize project leverages OpenCV for image processing and a Convolutional Neural Network (CNN) model built with Keras/TensorFlow to recognize traffic signs from images or video frames. It provides functionality for both static image prediction and live webcam-based real-time detection.

Key functionalities include:
- Preprocessing traffic sign images for model input
- Training a CNN classifier on the German Traffic Sign Recognition Benchmark (GTSRB) dataset
- Building a Flask web app to upload images and display predictions
- Extending the app to support live webcam streaming and detection

---

## Features

- Accurate traffic sign recognition using deep learning
- Supports image upload and prediction through a web interface
- Live webcam feed prediction mode with start/stop functionality
- Modular and scalable code architecture for easy extension
- Model saved as `model.h5` for easy loading and inference

---

## Technologies Used

- Python 3.x
- OpenCV
- TensorFlow / Keras
- Flask (for web app)
- NumPy
- Pandas (for data handling)
- Git & GitHub for version control

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/PavanMogaveer/Signalize.git 
    cd Signalize
    ```



2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset from the official source:

    [GTSRB Dataset Download Link](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

    After downloading, extract the dataset and place the files into the `dataset/` folder (or the appropriate folder in your project).

4. Download the pretrained **model.h5** file from:

    [Download model.h5](https://your-model-hosting-link.com/model.h5)

    Place the `model.h5` file in the root directory of the project.

---

## Usage

### Running the Flask Web App

To start the web app and use the image upload or live webcam prediction features:

```bash
python app.py


