
# Machine Learning Course GUI

## 📚 Project Overview

This project is a **professional Machine Learning GUI** developed as part of the MKT3434 course at Yildiz Technical University. It enables students, researchers, and engineers to perform supervised learning, clustering, dimensionality reduction, and model evaluation **without coding**, using a fully interactive interface.

The application uses powerful libraries:
- **Python 3.10**
- **scikit-learn** (Machine Learning)
- **TensorFlow Keras** (Deep Learning)
- **UMAP-learn** (Dimensionality Reduction)
- **Plotly** (Interactive Visualizations)
- **PyQt6** (GUI Development)

---

## ✨ Features

- **Dataset Management:** Load built-in or custom datasets, apply scaling, split into Train/Validation/Test.
- **Dimensionality Reduction:** PCA, LDA, t-SNE, and UMAP with full visualization.
- **Clustering:** KMeans with Elbow Method for optimal `k`, and clustering quality analysis.
- **Supervised Learning:** Train Logistic Regression, Decision Tree, Random Forest, SVM, Naive Bayes, Linear Regression.
- **Cross-Validation:** Perform 5-Fold CV with selectable metrics (Accuracy, MSE, RMSE).
- **Neural Network Training:** Build and train MLP models manually (dynamic layer creation).
- **Dynamic Visualization:** Interactive charts of performance metrics and projections.
- **Error Handling:** All critical actions wrapped with user-friendly error dialogs.

---

## 📊 Datasets Supported

- Iris Dataset
- Breast Cancer Dataset
- Digits Dataset
- California Housing Dataset
- Custom CSV Uploads

---

## 🛠️ Main Functionalities

| Feature | Description | Image |
|:--------|:------------|:------|
| **PCA** | Visualizes explained variance, shows first principal component direction. | [![PCA Explained Variance](./Ekran%20görüntüleri/PCA%20Explained%20Variance%20Ratio.png)](https://github.com/ilkerARSLAN1/MKT3434_2025/blob/main/Ekran%20görüntüleri/PCA%20Explained%20Variance%20Ratio.png) |
| **Eigenvectors** | Calculates eigenvalues and eigenvectors of a covariance matrix. | [![Eigenvalues & Eigenvectors](./Ekran%20görüntüleri/Eigenvalues%20and%20Eigenvectors%20from%20Covariance%20Matrix%20%CE%A3.png)](https://github.com/ilkerARSLAN1/MKT3434_2025/blob/main/Ekran%20görüntüleri/Eigenvalues%20and%20Eigenvectors%20from%20Covariance%20Matrix%20%CE%A3.png) |
| **First Principal Component** | Projects data along the 1st PC direction. | [![PCA Direction Vector](./Ekran%20görüntüleri/First%20Principal%20Component%20Visualization.png)](https://github.com/ilkerARSLAN1/MKT3434_2025/blob/main/Ekran%20görüntüleri/First%20Principal%20Component%20Visualization.png) |
| **LDA** | Class separation using Linear Discriminants. | [![LDA Projection](./Ekran%20görüntüleri/Linear%20Discriminant%20Analysis%20(LDA)%20Projection.png)](https://github.com/ilkerARSLAN1/MKT3434_2025/blob/main/Ekran%20görüntüleri/Linear%20Discriminant%20Analysis%20(LDA)%20Projection.png) |
| **t-SNE 2D** | t-SNE projection preserving local structure. | [![t-SNE 2D](./Ekran%20görüntüleri/t-SNE%20Projection%20of%20the%20Breast%20Cancer%20Dataset.png)](https://github.com/ilkerARSLAN1/MKT3434_2025/blob/main/Ekran%20görüntüleri/t-SNE%20Projection%20of%20the%20Breast%20Cancer%20Dataset.png) |
| **t-SNE 3D** | 3D t-SNE visualization of data. | [![t-SNE 3D](./Ekran%20görüntüleri/7B%20t-SNE%20Projection%20(3D).png)](https://github.com/ilkerARSLAN1/MKT3434_2025/blob/main/Ekran%20görüntüleri/7B%20t-SNE%20Projection%20(3D).png) |
| **UMAP** | Faster t-SNE alternative for local-global structure. | [![UMAP Projection](./Ekran%20görüntüleri/UMAP%20Projection%20(3D).png)](https://github.com/ilkerARSLAN1/MKT3434_2025/blob/main/Ekran%20görüntüleri/UMAP%20Projection%20(3D).png) |
| **KMeans** | Cluster analysis + Elbow Method. | [![KMeans & Elbow](./Ekran%20görüntüleri/K-Means%20Clustering%20Results%20and%20Elbow%20Method.png)](https://github.com/ilkerARSLAN1/MKT3434_2025/blob/main/Ekran%20görüntüleri/K-Means%20Clustering%20Results%20and%20Elbow%20Method.png) |
| **Plotly KMeans** | Interactive cluster visualization. | [![KMeans Plotly](./Ekran%20görüntüleri/Means%20Clustering%20Visualized%20with%20Plotly%20(k=3).png)](https://github.com/ilkerARSLAN1/MKT3434_2025/blob/main/Ekran%20görüntüleri/Means%20Clustering%20Visualized%20with%20Plotly%20(k=3).png) |

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
> Yıldız Technic University | Mechatronic Eng| 21067024
