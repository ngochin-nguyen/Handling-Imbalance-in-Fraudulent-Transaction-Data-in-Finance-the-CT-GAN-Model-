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

---
📌 **Next Steps**:  
- Train the **CT-GAN model** on the imbalanced dataset.  
- Generate synthetic fraud transactions.  
- Evaluate classification performance before and after augmentation.  
