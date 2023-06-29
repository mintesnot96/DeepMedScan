# DeepMedScan: Enhancing Medical Image Prediction with CNN and Computer Vision

DeepMedScan is an advanced deep learning project developed by ***Mintesnot, Migbar, and Ermias*** as part of their Deep Learning class at ***Addis Ababa University, Addis Ababa Institute of Technology***. The project leverages Convolutional Neural Networks (CNN) and Computer Vision techniques for accurate medical image prediction. The focus of the project is on analyzing X-ray and MRI scan images to provide predictions and diagnoses for various medical conditions.

## Features

- **X-ray Image Prediction**: DeepMedScan utilizes a trained CNN model to predict medical conditions based on X-ray scan images.
- **MRI Image Prediction**: The project also includes a CNN model specifically designed for predicting medical conditions from MRI scan images.
- **Computer Vision Techniques**: DeepMedScan incorporates computer vision techniques to preprocess and enhance medical images, improving the accuracy of predictions.
- **User-Friendly Web Interface**: The project provides a user-friendly web interface where users can upload medical images and obtain predictions for various medical conditions.
- **Supported Medical Conditions**: DeepMedScan currently supports prediction and diagnosis for pneumonia, COVID-19, brain tumors, and Alzheimer's disease.

## Prerequisites

Before running DeepMedScan, ensure you have the following dependencies installed:

- Python 3.7 or higher
- TensorFlow
- OpenCV
- Flask
- NumPy
- scikit-learn
- Other required libraries (specified in `requirements.txt`)

## Installation

1. Clone this GitHub repository:

 ```
git clone https://github.com/mintesnot96/DeepMedScan.git
```
2. Navigate to the project directory:
  ```
cd DeepMedScan
```
Install the required dependencies using pip:

```
pip install -r requirements.txt
```
Start the DeepMedScan web application:

```
python app.py
```
Access the web application in your browser by visiting http://localhost:5000.

Usage
Upload a medical image (X-ray or MRI scan) through the web interface.
Click the corresponding prediction button to obtain the prediction for the uploaded image.
The prediction result will be displayed on the screen, indicating the presence or absence of the medical condition.
Contribution
Contributions to DeepMedScan are welcome! If you find any issues or have suggestions for improvements, please submit an issue or a pull request. Let's work together to enhance the capabilities of medical image prediction.
 

# Screenshots
<img width="960" alt="image" src="https://github.com/mintesnot96/DeepMedScan/assets/96992238/1b26041b-8f5a-4fc8-abe0-a3c706e06c47">



License
This project is licensed under the MIT License.

Acknowledgments
The CNN models used in this project were trained on publicly available medical image datasets. Special thanks to the contributors and organizations who curated and made those datasets accessible.
We would also like to express our gratitude to the open-source community for their valuable contributions in the field of deep learning and computer vision.
