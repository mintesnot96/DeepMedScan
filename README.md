# DeepMedScan: Enhancing Medical Image Prediction with CNN and Computer Vision

DeepMedScan is an advanced deep learning project developed as part of their Deep Learning class at ***Addis Ababa University, Addis Ababa Institute of Technology***. The project leverages Convolutional Neural Networks (CNN) and Computer Vision techniques for accurate medical image prediction. The focus of the project is on analyzing X-ray and MRI scan images to provide predictions and diagnoses for various medical conditions.

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
<img width="960" alt="image" src="https://github.com/mintesnot96/DeepMedScan/assets/96992238/2b4a92f0-7fd1-4774-bcac-459f259f3005">



Check out https://deepmedscan-by-mintesnot-ermias-migbar.onrender.com/


License
This project is licensed under the MIT License.

### Acknowledgments
The CNN models utilized in this project underwent training using publicly available medical image datasets. We want to say a big thank you to the people and organizations that collected and shared the medical image datasets we used in this project. We are also grateful to the open-source community for their valuable contributions to deep learning and computer vision. Lastly, we would like to express our appreciation to our teacher for giving us the chance to work on this project.
