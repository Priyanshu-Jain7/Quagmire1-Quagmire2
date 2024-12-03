import random
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import messagebox, filedialog
import csv
import os


# ----------------------------
# Encryption Algorithms
# ----------------------------

def create_keyed_alphabet(keyword):
    keyword_unique = ''.join(sorted(set(keyword), key=keyword.index))
    remaining_letters = ''.join([ch for ch in string.ascii_uppercase if ch not in keyword_unique])
    return keyword_unique + remaining_letters


def shift_alphabet(alphabet, shift):
    return alphabet[shift:] + alphabet[:shift]


def encrypt_quagmire1(plaintext, key, indicator_key):
    plaintext = plaintext.upper().replace(" ", "")
    key_alphabet = create_keyed_alphabet(key.upper())
    indicator_key = indicator_key.upper()
    period = len(indicator_key)
    ciphertext = []

    for i, char in enumerate(plaintext):
        row = shift_alphabet(key_alphabet, string.ascii_uppercase.index(indicator_key[i % period]))
        plaintext_index = string.ascii_uppercase.index(char)
        ciphertext.append(row[plaintext_index])

    return ''.join(ciphertext)


def encrypt_quagmire2(plaintext, key, indicator_key):
    plaintext = plaintext.upper().replace(" ", "")
    cipher_alphabet = create_keyed_alphabet(key.upper())
    indicator_key = indicator_key.upper()
    period = len(indicator_key)
    ciphertext = []

    for i, char in enumerate(plaintext):
        row = shift_alphabet(string.ascii_uppercase, string.ascii_uppercase.index(indicator_key[i % period]))
        plaintext_index = cipher_alphabet.index(char)
        ciphertext.append(row[plaintext_index])

    return ''.join(ciphertext)


# ----------------------------
# Dataset Generation
# ----------------------------

def generate_dataset(num_samples=200):
    ciphers = {
        "Quagmire I": lambda text: encrypt_quagmire1(text, "SPRINGFEVER", "FLOWER"),
        "Quagmire II": lambda text: encrypt_quagmire2(text, "SPRINGFEVER", "FLOWER"),
    }
    dataset = []
    for _ in range(num_samples):
        plaintext = ''.join(random.choices(string.ascii_uppercase + " ", k=random.randint(20, 30)))
        for cipher_name, cipher_func in ciphers.items():
            ciphertext = cipher_func(plaintext)
            dataset.append((ciphertext, cipher_name))

    dataset_df = pd.DataFrame(dataset, columns=["Ciphertext", "Cipher Name"])
    dataset_df.to_csv("cipher_dataset.csv", index=False)
    return dataset_df


dataset_df = generate_dataset()

# ----------------------------
# Model Training and Evaluation
# ----------------------------

X = dataset_df["Ciphertext"]
y = dataset_df["Cipher Name"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Naive Bayes Classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vectorized, y_train)
y_pred_nb = nb_classifier.predict(X_test_vectorized)
nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_report = classification_report(y_test, y_pred_nb, output_dict=True)

# MLP Classifier (Neural Network)
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_classifier.fit(X_train_vectorized, y_train)
y_pred_mlp = mlp_classifier.predict(X_test_vectorized)
mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
mlp_report = classification_report(y_test, y_pred_mlp, output_dict=True)

# Save evaluation results
with open("model_results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Model", "Accuracy", "Precision (Quagmire I)", "Recall (Quagmire I)", "F1-Score (Quagmire I)",
                     "Precision (Quagmire II)", "Recall (Quagmire II)", "F1-Score (Quagmire II)"])

    # For Naive Bayes
    nb_quagmire1_metrics = nb_report["Quagmire I"]
    nb_quagmire2_metrics = nb_report["Quagmire II"]
    writer.writerow(["Naive Bayes", nb_accuracy, nb_quagmire1_metrics["precision"], nb_quagmire1_metrics["recall"],
                     nb_quagmire1_metrics["f1-score"],
                     nb_quagmire2_metrics["precision"], nb_quagmire2_metrics["recall"],
                     nb_quagmire2_metrics["f1-score"]])

    # For MLP Classifier
    mlp_quagmire1_metrics = mlp_report["Quagmire I"]
    mlp_quagmire2_metrics = mlp_report["Quagmire II"]
    writer.writerow(
        ["MLP Classifier", mlp_accuracy, mlp_quagmire1_metrics["precision"], mlp_quagmire1_metrics["recall"],
         mlp_quagmire1_metrics["f1-score"],
         mlp_quagmire2_metrics["precision"], mlp_quagmire2_metrics["recall"], mlp_quagmire2_metrics["f1-score"]])

# ----------------------------
# Print Model Results
# ----------------------------

print(f"Naive Bayes Accuracy: {nb_accuracy * 100:.2f}%")
print("Naive Bayes Classification Report:")
print(nb_report)

print(f"MLP Classifier Accuracy: {mlp_accuracy * 100:.2f}%")
print("MLP Classification Report:")
print(mlp_report)


# ----------------------------
# GUI Implementation
# ----------------------------

def encrypt_and_predict():
    plaintext = plaintext_entry.get().strip()
    if not plaintext:
        messagebox.showerror("Input Error", "Please enter plaintext.")
        return

    # Encrypt plaintext using Quagmire I
    ciphertext = encrypt_quagmire1(plaintext, "SPRINGFEVER", "FLOWER")

    # Predict using Naive Bayes and MLP
    input_vectorized = vectorizer.transform([ciphertext])
    nb_prediction = nb_classifier.predict(input_vectorized)[0]
    mlp_prediction = mlp_classifier.predict(input_vectorized)[0]

    # Update labels with ciphertext and predictions
    ciphertext_label.config(text=f"Generated Ciphertext: {ciphertext}")
    result_label.config(text=f"Naive Bayes: {nb_prediction}, MLP: {mlp_prediction}")

    # Save result to CSV
    with open("results.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([plaintext, ciphertext, nb_prediction, mlp_prediction])


def load_csv():
    filepath = filedialog.askopenfilename(title="Select CSV File", filetypes=(("CSV Files", "*.csv"),))
    if filepath:
        os.startfile(filepath)


# Set up GUI
window = tk.Tk()
window.title("Cipher Type Classifier")

# GUI Components
tk.Label(window, text="Enter Plaintext:").pack(pady=5)
plaintext_entry = tk.Entry(window, width=50)
plaintext_entry.pack(pady=5)

predict_button = tk.Button(window, text="Encrypt & Predict", command=encrypt_and_predict)
predict_button.pack(pady=10)

ciphertext_label = tk.Label(window, text="Generated Ciphertext: ", font=("Arial", 10))
ciphertext_label.pack(pady=5)

result_label = tk.Label(window, text="Prediction Results: ", font=("Arial", 12))
result_label.pack(pady=10)

view_csv_button = tk.Button(window, text="View Results CSV", command=load_csv)
view_csv_button.pack(pady=10)

# Run GUI
window.mainloop()
