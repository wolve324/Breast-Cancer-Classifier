# Breast-Cancer-Classifier
A k-NN model for breast cancer classification.

## Overview
This project is a machine learning model designed to classify breast cancer tumors as **malignant** or **benign** using the k-Nearest Neighbors (k-NN) algorithm. The dataset used is the built-in breast cancer dataset from Scikit-learn.

## Objective
The goal of this project is to:
1. Train a k-NN classifier to predict the class of tumors based on their features.
2. Evaluate the model's performance for different values of `k`.
3. Visualize the relationship between `k` and validation accuracy.

## Dataset
The dataset is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which includes:
- **Features**: 30 numerical features such as radius, texture, smoothness, etc.
- **Target**: Binary classification (0 = malignant, 1 = benign).
- **Size**: 569 samples.

## Project Steps
1. **Load Dataset**:
   - Used Scikit-learn's `load_breast_cancer` function to load the dataset.
   
2. **Split Data**:
   - Split the dataset into training (80%) and validation (20%) sets.

3. **Train Model**:
   - Trained a k-NN classifier for different values of `k` (from 1 to 99).

4. **Evaluate Model**:
   - Computed validation accuracy for each `k` and stored the results.

5. **Visualize Results**:
   - Plotted `k` vs. validation accuracy using Matplotlib.

## Results
- The model's performance varied significantly with different values of `k`.
- The plot of `k` vs. validation accuracy provides insights into the optimal choice of `k` for this dataset.

## Key Python Libraries
- `scikit-learn`: For dataset handling and k-NN implementation.
- `matplotlib`: For visualizing validation accuracy.

## Code Highlights
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Training the k-NN model
for k in range(1, 100):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    accuracies.append(classifier.score(validation_data, validation_labels))

# Plotting results
plt.plot(range(1, 100), accuracies)
plt.title("Breast Cancer Classifier Accuracy")
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.show()
```

## Future Improvements
1. Add a confusion matrix for detailed performance evaluation.
2. Explore hyperparameter tuning using `GridSearchCV`.
3. Compare performance with other algorithms (e.g., Logistic Regression, SVM).
4. Deploy the model using **Streamlit** or **Flask** for user interaction.

## Repository Structure
```
Breast-Cancer-Classifier/
├── main.py ( AI_ML_project1 for me)           # Code for training and evaluating the model
├── README.md             # Documentation of the project
├── accuracy_plot.png     # Visualization of accuracy vs. k
└── requirements.txt      # Dependencies
```

## How to Run
1. Clone this repository.
2. Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python main.py
   ```

## Contact
For questions or suggestions, feel free to contact [Your Name] at [Your Email].
