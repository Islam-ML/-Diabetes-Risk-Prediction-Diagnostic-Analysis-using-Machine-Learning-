# 🏥 Diabetes Risk Prediction & Diagnostic Analysis
**
## 📄 Overview
Diabetes is a chronic disease that affects millions worldwide. Early detection is crucial for effective management. This project applies **Machine Learning** techniques to predict the likelihood of diabetes in patients based on diagnostic measurements (such as Glucose, BMI, Age, etc.).

The workflow includes detailed **Exploratory Data Analysis (EDA)**, data preprocessing, handling class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**, and benchmarking multiple classification models to determine the best performance.

## 🚀 Key Features
*   **Automated Data Retrieval**: Downloads the dataset directly via `kagglehub`.
*   **Comprehensive EDA**: Visualizes data distributions, correlations, and outliers using Histograms, Boxplots, and Heatmaps.
*   **Advanced Preprocessing**: Includes feature scaling (StandardScaler) and stratified train-test splitting.
*   **Imbalance Handling**: Utilizes **SMOTE** to generate synthetic samples for the minority class, ensuring the model is not biased.
*   **Multi-Model Evaluation**: Compares:
    *   Logistic Regression
    *   Support Vector Machine (SVM)
    *   Gradient Boosting Classifier
*   **Performance Metrics**: Evaluates models based on Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.

## 🛠️ Technologies Used
*   **Python**: Core programming language.
*   **Pandas & NumPy**: Data manipulation and numerical operations.
*   **Matplotlib & Seaborn**: Data visualization.
*   **Scikit-Learn**: Machine learning models and evaluation metrics.
*   **Imbalanced-Learn**: For applying SMOTE.
*   **KaggleHub**: For dataset integration.

## 📂 Dataset
The project uses the **Pima Indians Diabetes Dataset** (sourced via Akshay Dattatray Khare on Kaggle).
*   **Inputs**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.
*   **Target**: Outcome (0 = Non-Diabetic, 1 = Diabetic).

## ⚙️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/diabetes-prediction.git
    cd diabetes-prediction
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn kagglehub
    ```

3.  **Run the script:**
    ```bash
    python main.py
    ```

## 📊 Methodology
1.  **Data Ingestion**: Load data using KaggleHub API.
2.  **Visualization**: Analyze feature distributions to understand skewness and detect outliers.
3.  **Preprocessing**: 
    *   Split data into training/testing sets.
    *   Standardize features to mean=0 and variance=1.
4.  **Resampling**: Apply SMOTE to the training set to balance the 'Diabetic' vs 'Non-Diabetic' classes.
5.  **Modeling**: Train models on the balanced dataset.
6.  **Evaluation**: Test on unseen data and save the best-performing model.

## 📈 Results Overview
The system generates a classification report for each model. Typically, ensemble methods like **Gradient Boosting** demonstrate superior performance in distinguishing between classes due to their ability to handle complex non-linear relationships.

*Specific accuracy scores and confusion matrices are generated at runtime.*

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements.

## 📜 License
This project is open-source and available under the MIT License.
