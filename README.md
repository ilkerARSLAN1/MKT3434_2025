
# Machine Learning Course GUI

## 📚 Project Overview

This project presents a graphical user interface (GUI) developed for the Machine Learning course (MKT 3434).  
It allows users to apply various **machine learning** and **dimensionality reduction** techniques interactively on popular datasets like Iris and Breast Cancer datasets.

Key focus areas:
- Dimensionality Reduction (PCA, LDA, t-SNE, UMAP)
- Clustering (KMeans, Elbow Method)
- Supervised Learning (Regression, Classification)
- Model Evaluation with K-Fold Cross-Validation
- Neural Networks (Basic MLP training)

---

## ✨ Features

- **Dataset Management:** Load built-in or custom CSV datasets, flexible Train/Validation/Test splitting, and scaling options.
- **Dimensionality Reduction:** Apply PCA, LDA, t-SNE, and UMAP with visualizations.
- **Clustering:** Perform KMeans clustering, determine optimal `k` using Elbow Method, and visualize clusters.
- **Supervised Learning:** Train models like Logistic Regression, Decision Trees, Random Forest, Naive Bayes, and Support Vector Machines.
- **Cross-Validation:** 5-Fold CV with model and metric selection (Accuracy, MSE, RMSE).
- **Neural Network Training:** Configure and train Multi-Layer Perceptron (MLP) models.
- **Dynamic Visualization:** Automatic plotting of evaluation results and training history.
- **Interactive Controls:** Real-time parameter tuning for projections, clustering, and model training.

---

## 📊 Datasets Used

- Iris Dataset
- Breast Cancer Dataset
- Digits Dataset
- California Housing Dataset
- Custom CSV Datasets (user-uploaded)

---

## 🛠️ Main Functionalities

| Feature | Description |
|:--------|:------------|
| **PCA** | Visualizes explained variance, shows first principal component direction. |
| **Eigenvectors** | Calculates eigenvalues and eigenvectors of a custom covariance matrix. |
| **LDA** | Projects dataset onto Linear Discriminants for class separation. |
| **t-SNE** | Projects dataset into 2D or 3D preserving local similarities. |
| **UMAP** | Faster alternative to t-SNE, preserving local and global structures. |
| **KMeans** | Cluster analysis with Elbow method and alternative cluster quality scores. |
| **Cross Validation** | Perform model evaluation using 5-Fold CV, dynamic model/metric selection. |
| **Neural Networks** | Build and train basic MLP architectures manually. |

---

## 🚀 How to Run

1. **Clone the repository**  
```bash
git clone <repository-url>
```

2. **Install Dependencies**  
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow umap-learn plotly kneed pyqt6
```

3. **Launch the Application**  
```bash
python main.py
```

> `main.py` contains the `MLCourseGUI` class that initializes and runs the full application.

---

## ⚙️ GUI Layout Summary

- **Data Management:** Dataset loading, scaling, train/val/test split adjustments.
- **Tabs:**
  - **Classical ML:** Regression, classification, k-fold CV.
  - **Dimensionality Reduction:** PCA, LDA, t-SNE, UMAP, KMeans.
  - **Deep Learning:** Configure and train basic neural networks.
  - **Reinforcement Learning:** (Structure prepared for future development.)

---

## 🧩 Important Notes

- **t-SNE vs UMAP:** UMAP is preferred for large datasets due to faster computation.
- **PCA:** Useful for global variance analysis, but weaker for local structure separation compared to t-SNE.
- **Cross-Validation:** Due to splitting/scaling, exact fold sizes may slightly vary (e.g., [21,21,21,21,20] instead of exact [20,20,20,20,20]).
- **Plotly Integration:** Enhanced KMeans visualization is provided for better cluster separation insights.
- **Error Handling:** Most critical operations are surrounded by try-except blocks to provide user-friendly error dialogs.

---

## 📑 References

- Scikit-Learn Documentation
- TensorFlow Keras API
- UMAP-Learn Official Documentation
- Plotly Express for Interactive Visualizations
- Kneedle Algorithm for Elbow Point Detection

---

### ✏️ Author

> İlker Arslan  
> Gazi University | Industrial Engineering | 21067024
