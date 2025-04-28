# Kullanıcının istediği gibi, görsel yolları "Ekran görüntüleri" klasörünü referans gösterecek şekilde düzenlenmiş README.md oluşturuyoruz.

readme_updated_content = """
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

![Initial View](./Ekran%20g%C3%B6r%C3%BCnt%C3%BCleri/nitial%20view%20of%20the%20Machine%20Learning%20Course%20GUI.png)

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

| Feature | Description | Image |
|:--------|:------------|:------|
| **PCA** | Visualizes explained variance, shows first principal component direction. | ![PCA Explained Variance](./Ekran%20g%C3%B6r%C3%BCnt%C3%BCleri/PCA%20Explained%20Variance%20Ratio.png) |
| **Eigenvectors** | Calculates eigenvalues and eigenvectors of a custom covariance matrix. | ![Eigenvalues & Eigenvectors](./Ekran%20g%C3%B6r%C3%BCnt%C3%BCleri/Eigenvalues%20and%20Eigenvectors%20from%20Covariance%20Matrix%20%CE%A3.png) |
| **First Principal Component** | Visualizes the direction vector of the first principal component. | ![PCA Direction Vector](./Ekran%20g%C3%B6r%C3%BCnt%C3%BCleri/First%20Principal%20Component%20Visualization.png) |
| **LDA** | Projects dataset onto Linear Discriminants for class separation. | ![LDA Projection](./Ekran%20g%C3%B6r%C3%BCnt%C3%BCleri/Linear%20Discriminant%20Analysis%20(LDA)%20Projection.png) |
| **t-SNE** | Projects dataset into 2D preserving local similarities. | ![t-SNE 2D](./Ekran%20g%C3%B6r%C3%BCnt%C3%BCleri/t-SNE%20Projection%20of%20the%20Breast%20Cancer%20Dataset.png) |
| **t-SNE (3D)** | Visualizes dataset structure in 3D space. | ![t-SNE 3D](./Ekran%20g%C3%B6r%C3%BCnt%C3%BCleri/7B%20t-SNE%20Projection%20(3D).png) |
| **UMAP** | Faster alternative to t-SNE, preserving local and global structures. | ![UMAP Projection](./Ekran%20g%C3%B6r%C3%BCnt%C3%BCleri/UMAP%20Projection%20(3D).png) |
| **KMeans** | Cluster analysis with Elbow method and visualization. | ![KMeans & Elbow](./Ekran%20g%C3%B6r%C3%BCnt%C3%BCleri/K-Means%20Clustering%20Results%20and%20Elbow%20Method.png) |
| **Plotly KMeans** | Interactive clustering visualization. | ![KMeans Plotly](./Ekran%20g%C3%B6r%C3%BCnt%C3%BCleri/Means%20Clustering%20Visualized%20with%20Plotly%20(k=3).png) |

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
