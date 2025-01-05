# A Comparative Study of Deep Learning Models for early Alzheimer's Disease using MRI Images 

## Description
This project implements deep learning models to classify Alzheimer’s Disease stages based on MRI images. Three different architectures—Custom CNN, ResNet50, and VGG18—are explored to evaluate their performance in distinguishing four stages: Non-Demented, Mild Demented, Moderate Demented, and Very Mild Demented. The models are trained and tested on a dataset sourced from Kaggle, leveraging techniques such as data augmentation, oversampling, and transfer learning to achieve high classification accuracy.

---

## Repository Structure
- **`code_CNN.ipynb`**: Contains the implementation of the Custom CNN model. It includes the architecture design, data preprocessing, training, evaluation, and visualization of results.
- **`code_Resnet.ipynb`**: Implements the ResNet50 model with transfer learning. This notebook focuses on feature extraction using pre-trained ImageNet weights and fine-tuning for Alzheimer’s classification.
- **`code_VGG.ipynb`**: Implements the VGG18 model, leveraging pre-trained weights and transfer learning to achieve state-of-the-art performance.

---

## Dataset
The dataset used in this project is the **Augmented Alzheimer’s MRI Dataset**, available on Kaggle. The dataset includes MRI scans categorized into four classes. Preprocessing steps include resizing, normalization, and oversampling to handle class imbalance.  
[Link to Dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)

---

## Requirements
Install the necessary Python libraries before running the code. The key dependencies are:
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- imbalanced-learn
