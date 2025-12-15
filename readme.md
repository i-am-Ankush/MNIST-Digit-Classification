# MNIST Digit Classification

In this project, I implemented a machine learning pipeline to classify handwritten digits (0–9) using the MNIST dataset.

The main goal of this project was not to achieve state-of-the-art performance, but to understand and correctly implement the complete machine learning workflow — from data loading to evaluation.

---

## Task Statement

Given an image of a handwritten digit, I built a machine learning model that predicts which digit (0–9) the image represents.

---

## Dataset

I used the **MNIST dataset**, which contains:
- 70,000 grayscale images of handwritten digits
- Each image is of size **28 × 28 pixels**
- Each image was flattened into a **784-dimensional feature vector** before training

---

## Approach

I followed these steps to build the pipeline:

1. Loaded the MNIST dataset using scikit-learn  
2. Visualized sample digit images to understand the data  
3. Normalized pixel values to the range 0–1  
4. Split the dataset into training (80%) and testing (20%) sets  
5. Trained a **Logistic Regression** model for multiclass digit classification  
6. Evaluated the model using test accuracy and a confusion matrix  

---

## Results

- **Test Accuracy:** ~92%
- The model correctly classifies most handwritten digits.
- From the confusion matrix and misclassified samples, I observed that the model sometimes struggles with visually similar digits such as:
  - 5 and 3
  - 8 and 9

This behavior is expected since Logistic Regression is a linear model and does not capture complex spatial patterns in images.

---

## Confusion Matrix

The confusion matrix below shows strong diagonal dominance, indicating correct classification for most digits.  
Misclassifications mainly occur between visually similar digits such as 5 and 3, and 8 and 9.

![Confusion Matrix](confusion_matrix.png)

---


## Optional Experiment

As an extension, I also experimented with a Random Forest classifier using the same train–test split.

While the Random Forest achieved higher accuracy, I didnt used it as the primary model in order to keep the pipeline simple, fast, and interpretable. The main focus of this project was correctness and understanding rather than maximizing accuracy.
 
---

## Observations

- Normalizing pixel values significantly improves training stability and performance  
- Solver choice plays an important role when working with large datasets like MNIST  
- Analyzing misclassified samples provides better insight than accuracy alone  

---

## Why Logistic Regression?

I chose Logistic Regression deliberately to focus on:
- understanding data preprocessing
- building a correct and clean ML pipeline
- interpreting model behavior and limitations  

instead of directly using deep learning models.

---

## How to Run

```bash
pip install -r requirements.txt
python mnist_pipeline.py
