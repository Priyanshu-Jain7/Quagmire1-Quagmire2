# Quagmire1-Quagmire2
Cipher Classification System
Overview
This project implements a machine learning-based system for classifying encryption algorithms,
focusing on Quagmire1 and Quagmire2 ciphers. The system generates encrypted datasets,
trains multiple machine learning models, and provides an interactive web interface for cipher
algorithm prediction.
Features
<h1>1. Data Encryption (cry.py)</h1>
● Implements encryption methods:
○ Quagmire1 Cipher
○ Quagmire2 Cipher
○ Complete Columnar Transposition
○ Compressocrat Cipher
● Generates labeled dataset of plaintext and ciphertext
● Saves dataset to CSV file
<h2>2. Machine Learning Models (mlcipherclassifier.py)</h2>
● Data Preprocessing:
○ TF-IDF Vectorization
○ Sequence Tokenization
● Trained Classifiers:
○ Naive Bayes
○ Neural Network
○ LSTM (Long Short-Term Memory)
● Generates detailed performance reports
● Predicts encryption algorithm based on input
<h3>3. Interactive Streamlit App (streamlitapp.py)</h3>
● Visualizes model performance metrics
● Displays confusion matrices
● Allows real-time cipher algorithm prediction
● User-friendly interface for inputting plaintext and ciphertext
Technologies Used
Programming Language
● Python 3.8+
Libraries and Frameworks
● Cryptography
● Data Processing:
○ pandas
○ numpy
● Machine Learning:
○ scikit-learn
○ TensorFlow
○ Keras
● Web Interface:
○ Streamlit
Tools
● Dataset Manipulation: csv, json
● Development Environment: PyCharm
Usage
1. Generate dataset by running cry.py
2. Train models by initializing CipherClassifier
3. Launch Streamlit app to view reports or make predictions
Project Methodology
Cipher Implementation
● Quagmire1 Cipher:
○ Uses single keyword
○ Substitution based on Vigenère cipher table
● Quagmire2 Cipher:
○ Uses two keywords
○ More complex encryption mechanism
Dataset Generation
● 100 plaintext samples encrypted using Quagmire1 and Quagmire2
● Labeled dataset with ciphertext and corresponding algorithm
Machine Learning Models
● Naive Bayes: Probabilistic baseline classifier
● Neural Networks: Learn abstract feature
