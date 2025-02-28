# Handling Imbalance in Fraudulent Transaction Data in Finance Using the CT-GAN Model

This is my research project on imbalanced data in financial fraud for the seminar course. It focuses on tabular credit card data and proposes using a Generative Model to address this issue. The goal is to augment minority class data, thereby improving the performance of classification models, which are typically sensitive to imbalanced data.

## Datasets Used:

### 1️⃣ [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Description**: This dataset contains credit card transactions made by European cardholders in September 2013. The dataset is highly imbalanced, with only **0.172% fraudulent transactions** out of **284,807 total transactions**.
- **Features**: 
  - Includes **28 anonymized numerical features (V1-V28)** obtained through PCA transformation.
  - Contains two additional features:  
    - `Time`: Seconds elapsed between each transaction and the first transaction in the dataset.  
    - `Amount`: The transaction amount.  
  - `Class` label:  
    - **0** → Legitimate transaction  
    - **1** → Fraudulent transaction  
- **Challenge**: Due to the extreme class imbalance, traditional machine learning models may struggle to correctly identify fraudulent transactions.  

### 2️⃣ [Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data)
- **Description**: This dataset provides a variety of transaction records labeled as fraudulent or non-fraudulent. It is designed for training and evaluating fraud detection models.
- **Features**:  
  - Includes categorical and numerical features such as transaction types, timestamps, and amount values.  
  - Provides class labels indicating whether a transaction is fraudulent or legitimate.  
- **Challenge**: Requires handling categorical encoding, feature selection, and balancing techniques for effective fraud detection.  

## Project Goal:
By leveraging **CT-GAN (Conditional Tabular GAN)**, this project aims to generate synthetic fraud samples to balance the dataset, thereby improving fraud detection model performance.

## Results and Findings

As observed in previous research, **credit card fraud detection** has gained significant attention in recent years, and multiple studies have been conducted to achieve optimal prediction performance using machine learning. As discussed earlier, **class imbalance is a major challenge** in credit card fraud detection.

### 🔹 Dataset 1 (Credit Card Fraud Detection)
- We encountered **overfitting** when attempting to generate fraudulent samples in large numbers (up to 1 million rows).  
- **Precision was too low**, leading to ineffective F1-score results.  
- Thus, **GMean and Recall were found to be the most optimal evaluation metrics** in this scenario.  

### 🔹 Dataset 2 (Fraud Detection Dataset)
- We used a **hybrid approach combining CT-GAN and SelectKBest feature selection** to address class imbalance in machine learning models.  
- **Three models were developed** based on this method.  
- **Random Forest combined with SelectKBest and CT-GAN outperformed all other classifiers (LR, XGB)**, achieving **100% Recall and F1-score**.  

### 🔹 Overall Model Performance:
- After training on **CT-GAN augmented data**, all classifiers showed **significant improvements in overall prediction performance** while reducing errors.  
- This study used datasets with entirely **numerical features**.  
- In the future, this approach could be tested on **categorical data** to verify its reliability.  
- The proposed method could also be applied to **other classification problems across different domains**.  


## 📖 Reference:
This project was conducted under the guidance of **Dr. Trần Anh Tuấn**.  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/tuantran83/)  

---
📌 **Next Steps**:  
- Train the **CT-GAN model** on the imbalanced dataset.  
- Generate synthetic fraud transactions.  
- Evaluate classification performance before and after augmentation.  
