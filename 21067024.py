import sys
from kneed import KneeLocator
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                           QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                           QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                           QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                           QDialog, QLineEdit)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn import datasets, preprocessing, model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, hinge_loss, log_loss
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from sklearn.svm import SVC, SVR
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)
        
        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        
        # Initialize data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        
        # Neural network configuration
        self.layer_config = []
        
        # Create components
        self.create_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()
    def apply_pca(self):
        """Apply PCA and show explained variance"""
        try:
            n_components = self.pca_components_spin.value()
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(self.X_train)

            # Explained variance ratio plot
            self.pca_variance_canvas.figure.clear()
            ax = self.pca_variance_canvas.figure.add_subplot(111)
            ax.bar(range(1, n_components+1), pca.explained_variance_ratio_, color='skyblue')
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_title('PCA - Explained Variance')
            self.pca_variance_canvas.draw()

            self.status_bar.showMessage("PCA Applied Successfully.")

        except Exception as e:
            self.show_error(f"PCA failed: {str(e)}")
    def show_pca_direction(self):
        try:
            from sklearn.decomposition import PCA

            n_components = self.pca_components_spin.value()
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(self.X_train)

            first_pc = pca.components_[0]  # 1. principal direction

            # Yeni: Grafik çizimi
            self.pca_direction_canvas.figure.clear()
            ax = self.pca_direction_canvas.figure.add_subplot(111)
            ax.scatter(X_pca[:, 0], X_pca[:, 1], c=self.y_train, cmap='viridis', edgecolor='k')
            ax.arrow(0, 0, first_pc[0]*3, first_pc[1]*3, color='red', width=0.05, head_width=0.2, label='1st PC')
            ax.set_title("PCA Direction Vector")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.legend()
            self.pca_direction_canvas.draw()

            # Bilgi mesajı da kalsın
            message = f"First Principal Component (Direction Vector):\n{np.round(first_pc, 4)}"
            QMessageBox.information(self, "PCA Direction", message)

        except Exception as e:
            self.show_error(f"Failed to compute PCA direction: {str(e)}")



    def apply_umap(self):
        """Apply UMAP and show projection"""
        try:
            import umap

            n_neighbors = self.umap_neighbors_spin.value()
            min_dist = self.umap_min_dist_spin.value()
            n_components = 2 if self.umap_dim_combo.currentText() == "2D" else 3

            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
            X_umap = reducer.fit_transform(self.X_train)

            self.umap_canvas.figure.clear()
            ax = self.umap_canvas.figure.add_subplot(111, projection='3d' if n_components == 3 else None)

            scatter = ax.scatter(
                X_umap[:, 0],
                X_umap[:, 1] if n_components >= 2 else None,
                X_umap[:, 2] if n_components == 3 else None,
                c=self.y_train,
                cmap='Spectral',
                edgecolor='k'
            )
            ax.set_title(f"UMAP Projection ({n_components}D)")
            self.umap_canvas.draw()

            self.status_bar.showMessage("UMAP projection complete.")

        except Exception as e:
            self.show_error(f"UMAP failed: {str(e)}")
    def compute_cov_eigen(self):
        import numpy as np
        sigma = np.array([[5, 2], [2, 3]])
        vals, vecs = np.linalg.eig(sigma)

        msg = f"""Covariance Matrix Σ:
    [[5 2]
    [2 3]]

    Eigenvalues:
    {np.round(vals, 4)}

    Eigenvectors (each column is a direction):
    {np.round(vecs, 4)}

    Principal 1D projection direction:
    {np.round(vecs[:, 0], 4)}
    """
        QMessageBox.information(self, "Eigenvector Result", msg)



    def apply_tsne(self):
        """Apply t-SNE for 2D/3D visualization"""
        try:
            from sklearn.manifold import TSNE

            perplexity = self.tsne_perplexity_spin.value()
            n_components = 2 if self.tsne_dimension_combo.currentText() == "2D" else 3

            # t-SNE uygulama
            tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
            X_tsne = tsne.fit_transform(self.X_train)

            # Plot
            self.tsne_canvas.figure.clear()
            ax = self.tsne_canvas.figure.add_subplot(111, projection='3d' if n_components == 3 else None)

            scatter = ax.scatter(
                X_tsne[:, 0],
                X_tsne[:, 1] if n_components >= 2 else None,
                X_tsne[:, 2] if n_components == 3 else None,
                c=self.y_train,
                cmap='plasma',
                edgecolor='k'
            )

            ax.set_title(f"t-SNE Projection ({n_components}D)")
            self.tsne_canvas.draw()

            self.status_bar.showMessage("t-SNE projection completed.")

        except Exception as e:
            self.show_error(f"t-SNE failed: {str(e)}")
    def run_cross_validation(self):
        """Perform k-fold cross-validation and report metrics + fold sizes"""
        try:
            k = self.cv_k_spin.value()
            model_name = self.cv_model_combo.currentText()

            # Model seçimi
            if model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=200)
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier()
            elif model_name == "Random Forest":
                model = RandomForestClassifier()
            else:
                self.show_error("Model not supported for CV")
                return

            # K-Fold ayarları
            kf = model_selection.KFold(n_splits=k, shuffle=True, random_state=42)
            acc_list, mse_list, rmse_list = [], [], []
            fold_sizes = []

            for train_idx, test_idx in kf.split(self.X_train):
                X_tr, X_te = self.X_train[train_idx], self.X_train[test_idx]
                y_tr, y_te = self.y_train[train_idx], self.y_train[test_idx]

                fold_sizes.append(len(test_idx))

                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)

                acc_list.append(accuracy_score(y_te, y_pred))
                mse_list.append(mean_squared_error(y_te, y_pred))
                rmse_list.append(np.sqrt(mean_squared_error(y_te, y_pred)))

            # METİN RAPORU
            metrics_text = "Cross-Validation Results:\n"
            metrics_text += f"Fold Sizes: {fold_sizes}\n"

            selected_metric = self.cv_metric_combo.currentText()

            if selected_metric == "Accuracy":
                metrics_text += f"Accuracy: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}\n"
            elif selected_metric == "MSE":
                metrics_text += f"MSE: {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}\n"
            elif selected_metric == "RMSE":
                metrics_text += f"RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}\n"

            self.metrics_text.setText(metrics_text)
            self.status_bar.showMessage("Cross-Validation Complete.")

            # GRAFİK
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            if selected_metric == "Accuracy":
                ax.plot(range(1, k + 1), acc_list, label="Accuracy", marker='o')
            elif selected_metric == "MSE":
                ax.plot(range(1, k + 1), mse_list, label="MSE", marker='o')
            elif selected_metric == "RMSE":
                ax.plot(range(1, k + 1), rmse_list, label="RMSE", marker='o')

            ax.set_xlabel("Fold")
            ax.set_ylabel("Metric Value")
            ax.set_title("Cross-Validation Metrics per Fold")
            ax.legend()
            self.canvas.draw()

        except Exception as e:
            self.show_error(f"CV failed: {str(e)}")



    def apply_lda(self):
        """Apply LDA and plot class separation"""
        try:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            n_components = self.lda_components_spin.value()

            # ❗ Kullanıcının girdiği bileşen sayısı geçerli mi kontrol et
            n_classes = len(np.unique(self.y_train))
            n_features = self.X_train.shape[1]
            max_components = min(n_features, n_classes - 1)

            if n_components > max_components:
                self.show_error(f"LDA failed: You selected {n_components} components, "
                                f"but maximum allowed is {max_components}.")
                return

            lda = LinearDiscriminantAnalysis(n_components=n_components)
            X_lda = lda.fit_transform(self.X_train, self.y_train)

            # Plot the result
            self.lda_scatter_canvas.figure.clear()
            ax = self.lda_scatter_canvas.figure.add_subplot(111)
            scatter = ax.scatter(X_lda[:, 0], np.zeros_like(X_lda[:, 0]),
                                c=self.y_train, cmap='viridis', edgecolor='k')
            ax.set_title("LDA Projection")
            ax.set_xlabel("LD1")
            if X_lda.shape[1] > 1:
                ax.scatter(X_lda[:, 1], np.zeros_like(X_lda[:, 1]),
                        c=self.y_train, cmap='viridis', edgecolor='k')
                ax.set_ylabel("LD2")
            self.lda_scatter_canvas.draw()

            # ✅ Separation score GUI'de göster
            separation_score = lda.explained_variance_ratio_.sum()
            self.status_bar.showMessage(f"LDA Applied. Separation Score: {separation_score:.4f}")
            self.metrics_text.setText(f"""
    LDA Projection Summary:
    - Requested Components: {n_components}
    - Separation Score (Explained Variance Sum): {separation_score:.4f}
    """)

        except Exception as e:
            self.show_error(f"LDA failed: {str(e)}")

    def apply_kmeans(self):
        """Apply KMeans clustering and plot Elbow + Clusters"""
        try:
            k = self.kmeans_cluster_spin.value()

            # Elbow method için 1-10 arası inertia değeri hesapla
            inertias = []
            for i in range(1, 11):
                km = KMeans(n_clusters=i, random_state=42)
                km.fit(self.X_train)
                inertias.append(km.inertia_)
            k_range = range(1, 11)
            kneedle = KneeLocator(k_range, inertias, curve="convex", direction="decreasing")
            optimal_k = kneedle.knee

            print(f"Optimum k (elbow point): {optimal_k}")
            self.status_bar.showMessage(f"Optimum k (elbow): {optimal_k}")



            # Seçilen k ile KMeans yap
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(self.X_train)

            # Plot
            # Plot Elbow grafiği
            self.kmeans_canvas.figure.clear()
            ax1 = self.kmeans_canvas.figure.add_subplot(121)
            ax1.plot(k_range, inertias, marker='o', label="Inertia")

            # 🔴 Optimum k noktasını göster
            if optimal_k:
                ax1.axvline(optimal_k, color='red', linestyle='--', label=f'Elbow at k={optimal_k}')

            ax1.set_title("Elbow Method")
            ax1.set_xlabel("Number of Clusters")
            ax1.set_ylabel("Inertia")
            ax1.legend()


            # PCA ile 2D görselleştirme
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(self.X_train)
            ax2 = self.kmeans_canvas.figure.add_subplot(122)
            scatter = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='tab10', edgecolor='k')
            ax2.set_title(f"KMeans Clustering (k={k})")

            self.kmeans_canvas.draw()
            self.status_bar.showMessage("KMeans clustering & Elbow Method completed.")
            # Kümeleme kalite metrikleri
            from sklearn.metrics import silhouette_score, davies_bouldin_score
            sil_score = silhouette_score(self.X_train, clusters)
            db_score = davies_bouldin_score(self.X_train, clusters)

            # Metrikleri GUI'de göster
            self.metrics_text.setText(f"""
    KMeans Clustering Results (k={k}):
    Silhouette Score: {sil_score:.4f}
    Davies-Bouldin Index: {db_score:.4f}
            """)
            # 🔢 Alternatif k değerleri için metrik karşılaştırması
            comparison_text = "k | Silhouette | Davies-Bouldin\n"
            comparison_text += "-" * 35 + "\n"

            for alt_k in range(2, 11):  # k=2 to k=10
                alt_kmeans = KMeans(n_clusters=alt_k, random_state=42)
                alt_clusters = alt_kmeans.fit_predict(self.X_train)

                try:
                    sil = silhouette_score(self.X_train, alt_clusters)
                    db = davies_bouldin_score(self.X_train, alt_clusters)
                    comparison_text += f"{alt_k:<2} | {sil:.4f}     | {db:.4f}\n"
                except Exception:
                    comparison_text += f"{alt_k:<2} | Error\n"

            # 📌 Sonuçları metrics_text’e ekle
            self.metrics_text.append("\nAlternative Cluster Quality Comparison:\n")
            self.metrics_text.append(comparison_text)



        except Exception as e:
            self.show_error(f"KMeans failed: {str(e)}")



    
    def huber_loss(self, y_true, y_pred, delta=1.0):
        """
        Calculate Huber Loss.
        
        Parameters:
         y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        delta (float): Threshold for switching between MSE and MAE. Default is 1.0.
        
        Returns:
            float: Huber Loss value.
        """
        error = y_true - y_pred
        is_small_error = np.abs(error) <= delta
        squared_loss = 0.5 * np.square(error)
        linear_loss = delta * (np.abs(error) - 0.5 * delta)
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))
    
        
    def train_model(self, model_name, param_widgets):
        """Train the selected model using scikit-learn's built-in classes."""
        try:
            if model_name == "Linear Regression":
                # Get parameters
                fit_intercept = param_widgets["fit_intercept"].isChecked()
        
                # Create and train the model
                model = LinearRegression(fit_intercept=fit_intercept)
                model.fit(self.X_train, self.y_train)
                self.current_model = model
        
                # Make predictions
                y_pred = model.predict(self.X_test)
        
                # Calculate selected loss
                loss_function = self.loss_regression_combo.currentText()
                if loss_function == "MSE":
                    loss = mean_squared_error(self.y_test, y_pred)
                elif loss_function == "MAE":
                    loss = mean_absolute_error(self.y_test, y_pred)
                elif loss_function == "Huber Loss":
                    loss = self.huber_loss(self.y_test, y_pred)
        
                # Update visualization and metrics
                self.update_visualization(y_pred)
                self.update_metrics(y_pred, loss)
        
                self.status_bar.showMessage(f"Linear Regression Training Complete. Loss: {loss:.4f}")
    
            elif model_name == "Logistic Regression":
                # Get parameters
                C = param_widgets["C"].value()
                max_iter = param_widgets["max_iter"].value()
                multi_class = param_widgets["multi_class"].currentText()
        
                # Create and train the model
                model = LogisticRegression(C=C, max_iter=max_iter, multi_class=multi_class)
                model.fit(self.X_train, self.y_train)
        
                # Make predictions
                y_pred = model.predict(self.X_test)
                self.current_model = model
        
                # Calculate selected loss
                loss_function = self.loss_classification_combo.currentText()
                if loss_function == "Cross-Entropy":
                    loss = log_loss(self.y_test, model.predict_proba(self.X_test))
                elif loss_function == "Hinge Loss":
                    loss = hinge_loss(self.y_test, y_pred)
        
                # Update visualization and metrics
                self.update_visualization(y_pred)
                self.update_metrics(y_pred, loss)
        
                self.status_bar.showMessage(f"Logistic Regression Training Complete. Loss: {loss:.4f}")
            
            elif model_name == "Support Vector Machine (Regression)":
                # Get parameters
                C = param_widgets["C"].value()
                kernel = param_widgets["kernel"].currentText()
                degree = param_widgets["degree"].value() if kernel == "poly" else 3

                # Create and train the model
                model = SVR(kernel=kernel, C=C, degree=degree)
                model.fit(self.X_train, self.y_train)
                self.current_model = model  # Set the current model

                # Make predictions
                y_pred = model.predict(self.X_test)

                # Calculate regression loss
                loss_function = self.loss_regression_combo.currentText()
                if loss_function == "MSE":
                    loss = mean_squared_error(self.y_test, y_pred)
                elif loss_function == "MAE":
                    loss = mean_absolute_error(self.y_test, y_pred)
                elif loss_function == "Huber Loss":
                    loss = self.huber_loss(self.y_test, y_pred)

                # Update visualization and metrics
                self.update_visualization(y_pred)
                self.update_metrics(y_pred, loss)

                self.status_bar.showMessage(f"SVM (Regression) Training Complete. Loss: {loss:.4f}")

            elif model_name == "Support Vector Machine":
                # Get parameters
                C = param_widgets["C"].value()
                kernel = param_widgets["kernel"].currentText()
                degree = param_widgets["degree"].value() if kernel == "poly" else 3

                # Determine if it's classification or regression
                if len(np.unique(self.y_train)) > 10:  # Regression
                    model = SVR(kernel=kernel, C=C, degree=degree)
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                    self.current_model = model
    
                    # Calculate regression loss
                    loss_function = self.loss_regression_combo.currentText()
                    if loss_function == "MSE":
                        loss = mean_squared_error(self.y_test, y_pred)
                    elif loss_function == "MAE":
                        loss = mean_absolute_error(self.y_test, y_pred)
                    elif loss_function == "Huber Loss":
                        loss = self.huber_loss(self.y_test, y_pred)

                else:  # Classification
                    model = SVC(kernel=kernel, C=C, degree=degree)
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                    self.current_model = model
    
                    # Calculate classification loss
                    loss_function = self.loss_classification_combo.currentText()
                    if loss_function == "Cross-Entropy":
                        loss = log_loss(self.y_test, model.decision_function(self.X_test))
                    elif loss_function == "Hinge Loss":
                        loss = hinge_loss(self.y_test, y_pred)

                # Update visualization and metrics
                self.update_visualization(y_pred)
                self.update_metrics(y_pred, loss)

                self.status_bar.showMessage(f"SVM Training Complete. Loss: {loss:.4f}")

            elif model_name == "Naive Bayes":
                # Get parameters
                var_smoothing = param_widgets["var_smoothing"].value()

                # Create and train the model
                model = GaussianNB(var_smoothing=var_smoothing)
                model.fit(self.X_train, self.y_train)
                self.current_model = model  # Set the current model

                # Make predictions
                y_pred = model.predict(self.X_test)

                # Calculate selected loss
                loss_function = self.loss_classification_combo.currentText()
                if loss_function == "Cross-Entropy":
                    loss = log_loss(self.y_test, model.predict_proba(self.X_test))
                elif loss_function == "Hinge Loss":
                    loss = hinge_loss(self.y_test, y_pred)

                # Update visualization and metrics
                self.update_visualization(y_pred)
                self.update_metrics(y_pred, loss)

                self.status_bar.showMessage(f"Naive Bayes Training Complete. Loss: {loss:.4f}")

            else:
                self.show_error(f"Model {model_name} not implemented yet.")

        except Exception as e:
            self.show_error(f"Error training model: {str(e)}")
            print(f"Error details: {e}")  # Print the error to the console for debugging
    
    def load_dataset(self):
        """Load selected dataset"""
        if self.dataset_combo.currentText() == "Load Custom Dataset":
            return  # çünkü bu durumda data = ... satırı çalışmaz, load_custom_data ayrı

        try:
            dataset_name = self.dataset_combo.currentText()
            
            # Load selected dataset
            if dataset_name == "Iris Dataset":
                data = datasets.load_iris()
            elif dataset_name == "Breast Cancer Dataset":
                data = datasets.load_breast_cancer()
            elif dataset_name == "Digits Dataset":
                data = datasets.load_digits()
            elif dataset_name == "Boston Housing Dataset":
                data = datasets.fetch_california_housing()

            # Bu noktadan sonra tüm datasetler için ortak bölge:
            train_ratio = self.train_spin.value() / 100.0
            val_ratio = self.val_spin.value() / 100.0
            test_ratio = self.test_spin.value() / 100.0

            if round(train_ratio + val_ratio + test_ratio, 2) != 1.0:

                self.show_error("Train + Val + Test ratios must add up to 100%.")
                return

            X_temp, self.X_test, y_temp, self.y_test = model_selection.train_test_split(
                data.data, data.target,
                test_size=test_ratio,
                random_state=42
            )

            val_adjusted = val_ratio / (train_ratio + val_ratio)
            self.X_train, self.X_val, self.y_train, self.y_val = model_selection.train_test_split(
                X_temp, y_temp,
                test_size=val_adjusted,
                random_state=42
            )

            # Scaling işlemi
            self.apply_scaling()
            self.status_bar.showMessage(f"Loaded {dataset_name}")

            
        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")
    
    def load_custom_data(self):
        """Load custom dataset from CSV file"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Dataset",
                "",
                "CSV files (*.csv)"
            )
            
            if file_name:
                # Load data
                data = pd.read_csv(file_name)
                
                # Ask user to select target column
                target_col = self.select_target_column(data.columns)
                
                if target_col:
                    X = data.drop(target_col, axis=1)
                    y = data[target_col]
                    
                    # Split data
                   # Yeni: Train/Val/Test oranlarını al
                    train_ratio = self.train_spin.value() / 100.0
                    val_ratio = self.val_spin.value() / 100.0
                    test_ratio = self.test_spin.value() / 100.0

                    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 0.01:
                        self.show_error("Train + Val + Test ratios must add up to 100%.")
                        return

                    # İlk olarak test ayrılır
                    X_temp, self.X_test, y_temp, self.y_test = model_selection.train_test_split(
                        data.data, data.target,
                        test_size=test_ratio,
                        random_state=42
                    )

                    # Ardından train ve validation ayrılır
                    val_adjusted = val_ratio / (train_ratio + val_ratio)
                    self.X_train, self.X_val, self.y_train, self.y_val = model_selection.train_test_split(
                        X_temp, y_temp,
                        test_size=val_adjusted,
                        random_state=42
)

                    
                    # Apply scaling if selected
                    self.apply_scaling()
                    
                    self.status_bar.showMessage(f"Loaded custom dataset: {file_name}")
                    
        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")
    
    def select_target_column(self, columns):
        """Dialog to select target column from dataset"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dialog)
        
        combo = QComboBox()
        combo.addItems(columns)
        layout.addWidget(combo)
        
        btn = QPushButton("Select")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None
    
    def apply_scaling(self):
        """Apply selected scaling method to the data"""
        scaling_method = self.scaling_combo.currentText()
        
        if scaling_method != "No Scaling":
            try:
                if scaling_method == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scaling_method == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scaling_method == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()
                
                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)
                
            except Exception as e:
                self.show_error(f"Error applying scaling: {str(e)}")
    def create_data_section(self):
        """Create the data loading and preprocessing section"""
        data_group = QGroupBox("Data Management")
        data_layout = QHBoxLayout()
                # Train/Validation/Test split ratios
        self.train_spin = QSpinBox()
        self.train_spin.setRange(10, 90)
        self.train_spin.setValue(70)

        self.val_spin = QSpinBox()
        self.val_spin.setRange(0, 90)
        self.val_spin.setValue(15)

        self.test_spin = QSpinBox()
        self.test_spin.setRange(10, 90)
        self.test_spin.setValue(15)

        data_layout.addWidget(QLabel("Train %:"))
        data_layout.addWidget(self.train_spin)
        data_layout.addWidget(QLabel("Val %:"))
        data_layout.addWidget(self.val_spin)
        data_layout.addWidget(QLabel("Test %:"))
        data_layout.addWidget(self.test_spin)

        
        # Dataset selection
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset",
            "Breast Cancer Dataset",
            "Digits Dataset",
            "Boston Housing Dataset",
            "MNIST Dataset"
        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)
        
        # Data loading button
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_custom_data)
        
        # Preprocessing options
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems([
            "No Scaling",
            "Standard Scaling",
            "Min-Max Scaling",
            "Robust Scaling"
        ])
        
        # Train-test split options
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)
        self.split_spin.setSingleStep(0.1)
        
        # Add widgets to layout
        data_layout.addWidget(QLabel("Dataset:"))
        data_layout.addWidget(self.dataset_combo)
        data_layout.addWidget(self.load_btn)
        data_layout.addWidget(QLabel("Scaling:"))
        data_layout.addWidget(self.scaling_combo)
        data_layout.addWidget(QLabel("Test Split:"))
        data_layout.addWidget(self.split_spin)
        
        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)
    
    def create_tabs(self):
        """Create tabs for different ML topics"""
        self.tab_widget = QTabWidget()
        
        # Create individual tabs
        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Reinforcement Learning", self.create_rl_tab)
        ]
        
        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)
        
        self.layout.addWidget(self.tab_widget)
    
    def create_classical_ml_tab(self):
        """Create the classical machine learning algorithms tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # Regression section
        regression_group = QGroupBox("Regression")
        regression_layout = QVBoxLayout()

        # Loss function selection for regression
        loss_regression_layout = QHBoxLayout()
        loss_regression_layout.addWidget(QLabel("Loss Function:"))
        self.loss_regression_combo = QComboBox()
        self.loss_regression_combo.addItems(["MSE", "MAE", "Huber Loss"])
        loss_regression_layout.addWidget(self.loss_regression_combo)
        regression_layout.addLayout(loss_regression_layout)

        # Linear Regression
        lr_group = self.create_algorithm_group(
            "Linear Regression",
            {"fit_intercept": "checkbox"}
        )
        regression_layout.addWidget(lr_group)

        # SVM for Regression
        svm_regression_group = self.create_algorithm_group(
            "Support Vector Machine (Regression)",
            {"C": "double",
            "kernel": ["linear", "rbf", "poly"],
            "degree": "int"}
        )
        regression_layout.addWidget(svm_regression_group)

        regression_group.setLayout(regression_layout)
        layout.addWidget(regression_group, 0, 0)

        # Classification section
        classification_group = QGroupBox("Classification")
        classification_layout = QVBoxLayout()

        # Loss function selection for classification
        loss_classification_layout = QHBoxLayout()
        loss_classification_layout.addWidget(QLabel("Loss Function:"))
        self.loss_classification_combo = QComboBox()
        self.loss_classification_combo.addItems(["Cross-Entropy", "Hinge Loss"])
        loss_classification_layout.addWidget(self.loss_classification_combo)
        classification_layout.addLayout(loss_classification_layout)

        # Logistic Regression
        logistic_group = self.create_algorithm_group(
            "Logistic Regression",
            {"C": "double",
            "max_iter": "int",
            "multi_class": ["ovr", "multinomial"]}
        )
        classification_layout.addWidget(logistic_group)

        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group, 0, 1)
        
        # Naive Bayes
        nb_group = self.create_algorithm_group(
            "Naive Bayes",
            {"var_smoothing": "double"}  # GaussianNB parameter
        )
        classification_layout.addWidget(nb_group)
                # Cross-validation bölümü
        cv_group = QGroupBox("K-Fold Cross-Validation")
        cv_layout = QVBoxLayout()

        self.cv_k_spin = QSpinBox()
        self.cv_k_spin.setRange(2, 10)
        self.cv_k_spin.setValue(5)
        cv_layout.addWidget(QLabel("Number of Folds (k):"))
        cv_layout.addWidget(self.cv_k_spin)

        self.cv_model_combo = QComboBox()
        # Metrik seçimi
        self.cv_metric_combo = QComboBox()
        self.cv_metric_combo.addItems(["Accuracy", "MSE", "RMSE"])
        cv_layout.addWidget(QLabel("Metric:"))
        cv_layout.addWidget(self.cv_metric_combo)

        self.cv_model_combo.addItems(["Logistic Regression", "Decision Tree", "Random Forest"])
        cv_layout.addWidget(QLabel("Model:"))
        cv_layout.addWidget(self.cv_model_combo)

        cv_btn = QPushButton("Run Cross-Validation")
        cv_btn.clicked.connect(self.run_cross_validation)
        cv_layout.addWidget(cv_btn)

        cv_group.setLayout(cv_layout)
        layout.addWidget(cv_group, 1, 1)  # Sağ tarafa ekle


        return widget
    
    def create_dim_reduction_tab(self):
        """Create the dimensionality reduction tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # ========== PCA SECTION ==========
        pca_group = QGroupBox("Principal Component Analysis (PCA)")
        pca_layout = QVBoxLayout()

        # PCA n_components seçimi
        self.pca_components_spin = QSpinBox()
        self.pca_components_spin.setRange(1, 10)
        self.pca_components_spin.setValue(2)
        pca_layout.addWidget(QLabel("Number of Components:"))
        pca_layout.addWidget(self.pca_components_spin)

        # PCA butonu
        pca_btn = QPushButton("Apply PCA")
        # Yeni: PCA direction gösterme butonu
        show_pca_dir_btn = QPushButton("Show PCA Direction")
        show_pca_dir_btn.clicked.connect(self.show_pca_direction)
        pca_layout.addWidget(show_pca_dir_btn)
        pca_btn.clicked.connect(self.apply_pca)
        pca_layout.addWidget(pca_btn)
        # Eigenvector gösterme butonu
        eigen_btn = QPushButton("Compute Eigenvectors for Σ=[[5,2],[2,3]]")
        eigen_btn.clicked.connect(self.compute_cov_eigen)
        pca_layout.addWidget(eigen_btn)


        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group, 0, 0)

        # Placeholder for PCA variance plot
        self.pca_variance_canvas = FigureCanvas(Figure(figsize=(4, 3)))
        layout.addWidget(self.pca_variance_canvas, 0, 1)
        # PCA yön vektörü çizimi için ikinci canvas
        self.pca_direction_canvas = FigureCanvas(Figure(figsize=(4, 3)))
        layout.addWidget(self.pca_direction_canvas, 0, 2)


        widget.setLayout(layout)


            # ========== LDA SECTION ==========
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        lda_group = QGroupBox("Linear Discriminant Analysis (LDA)")
        lda_layout = QVBoxLayout()

        # LDA n_components seçimi
        self.lda_components_spin = QSpinBox()
        self.lda_components_spin.setRange(1, 10)
        self.lda_components_spin.setValue(2)
        lda_layout.addWidget(QLabel("Number of Components:"))
        lda_layout.addWidget(self.lda_components_spin)

        # LDA butonu
        lda_btn = QPushButton("Apply LDA")
        lda_btn.clicked.connect(self.apply_lda)
        lda_layout.addWidget(lda_btn)

        lda_group.setLayout(lda_layout)
        layout.addWidget(lda_group, 1, 0)

        # Placeholder for LDA scatter plot
        self.lda_scatter_canvas = FigureCanvas(Figure(figsize=(4, 3)))
        layout.addWidget(self.lda_scatter_canvas, 1, 1)
            # ========== t-SNE SECTION ==========
        from sklearn.manifold import TSNE

        tsne_group = QGroupBox("t-SNE Projection")
        tsne_layout = QVBoxLayout()

        # Perplexity seçimi
        self.tsne_perplexity_spin = QSpinBox()
        self.tsne_perplexity_spin.setRange(5, 100)
        self.tsne_perplexity_spin.setValue(30)
        tsne_layout.addWidget(QLabel("Perplexity:"))
        tsne_layout.addWidget(self.tsne_perplexity_spin)

        # Boyut seçimi (2D/3D)
        self.tsne_dimension_combo = QComboBox()
        self.tsne_dimension_combo.addItems(["2D", "3D"])
        tsne_layout.addWidget(QLabel("Projection Dimension:"))
        tsne_layout.addWidget(self.tsne_dimension_combo)

        # Buton
        tsne_btn = QPushButton("Apply t-SNE")
        tsne_btn.clicked.connect(self.apply_tsne)
        tsne_layout.addWidget(tsne_btn)

        tsne_group.setLayout(tsne_layout)
        layout.addWidget(tsne_group, 2, 0)

        # Placeholder for t-SNE plot
        self.tsne_canvas = FigureCanvas(Figure(figsize=(4, 3)))
        layout.addWidget(self.tsne_canvas, 2, 1)
                # ========== KMeans SECTION ==========
        kmeans_group = QGroupBox("K-Means Clustering")
        kmeans_layout = QVBoxLayout()

        self.kmeans_cluster_spin = QSpinBox()
        self.kmeans_cluster_spin.setRange(1, 10)
        self.kmeans_cluster_spin.setValue(3)
        kmeans_layout.addWidget(QLabel("Number of Clusters (k):"))
        kmeans_layout.addWidget(self.kmeans_cluster_spin)

        # KMeans butonu
        kmeans_btn = QPushButton("Apply KMeans + Elbow Method")
        kmeans_btn.clicked.connect(self.apply_kmeans)
        kmeans_layout.addWidget(kmeans_btn)
        plotly_btn = QPushButton("Show KMeans (Plotly)")
        plotly_btn.clicked.connect(self.show_plotly_kmeans)
        kmeans_layout.addWidget(plotly_btn)


        kmeans_group.setLayout(kmeans_layout)
        layout.addWidget(kmeans_group, 3, 0)

        # Elbow ve clustering sonuçları için canvas
        self.kmeans_canvas = FigureCanvas(Figure(figsize=(4, 3)))
        layout.addWidget(self.kmeans_canvas, 3, 1)

        # ========== UMAP SECTION ==========
        import umap

        umap_group = QGroupBox("UMAP Projection")
        umap_layout = QVBoxLayout()

        self.umap_neighbors_spin = QSpinBox()
        self.umap_neighbors_spin.setRange(2, 100)
        self.umap_neighbors_spin.setValue(15)
        umap_layout.addWidget(QLabel("Number of Neighbors:"))
        umap_layout.addWidget(self.umap_neighbors_spin)

        self.umap_min_dist_spin = QDoubleSpinBox()
        self.umap_min_dist_spin.setRange(0.0, 1.0)
        self.umap_min_dist_spin.setSingleStep(0.05)
        self.umap_min_dist_spin.setValue(0.1)
        umap_layout.addWidget(QLabel("Min Distance:"))
        umap_layout.addWidget(self.umap_min_dist_spin)

        self.umap_dim_combo = QComboBox()
        self.umap_dim_combo.addItems(["2D", "3D"])
        umap_layout.addWidget(QLabel("Projection Dimension:"))
        umap_layout.addWidget(self.umap_dim_combo)

        umap_btn = QPushButton("Apply UMAP")
        umap_btn.clicked.connect(self.apply_umap)
        umap_layout.addWidget(umap_btn)

        umap_group.setLayout(umap_layout)
        layout.addWidget(umap_group, 4, 0)

        self.umap_canvas = FigureCanvas(Figure(figsize=(4, 3)))
        layout.addWidget(self.umap_canvas, 4, 1)




        return widget

    def show_plotly_kmeans(self):
        from sklearn.decomposition import PCA
        import plotly.express as px
        import numpy as np
        import pandas as pd
        from sklearn.cluster import KMeans

        # 2D'ye indir (PCA ile)
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(self.X_train)

        # KMeans uygula
        k = self.kmeans_cluster_spin.value()
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(self.X_train)

        # DataFrame'e aktar
        df = pd.DataFrame({
            'PC1': X_2d[:, 0],
            'PC2': X_2d[:, 1],
            'Cluster': clusters.astype(str)
        })

        fig = px.scatter(df, x='PC1', y='PC2', color='Cluster',
                        title=f"KMeans with Plotly (k={k})",
                        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'})
        fig.show()


    
    def create_rl_tab(self):
        """Create the reinforcement learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Environment selection
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()
        
        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1"
        ])
        env_layout.addWidget(self.env_combo)
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group, 0, 0)
        
        # RL Algorithm selection
        algo_group = QGroupBox("RL Algorithm")
        algo_layout = QVBoxLayout()
        
        self.rl_algo_combo = QComboBox()
        self.rl_algo_combo.addItems([
            "Q-Learning",
            "SARSA",
            "DQN"
        ])
        algo_layout.addWidget(self.rl_algo_combo)
        
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group, 0, 1)
        
        return widget
    
    def create_visualization(self):
        """Create the visualization section"""
        viz_group = QGroupBox("Visualization")
        viz_layout = QHBoxLayout()
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)
        
        # Metrics display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        viz_layout.addWidget(self.metrics_text)
        
        viz_group.setLayout(viz_layout)
        self.layout.addWidget(viz_group)
    
    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def create_algorithm_group(self, name, params):
        """Helper method to create algorithm parameter groups"""
        group = QGroupBox(name)
        layout = QVBoxLayout()
        
        # Create parameter inputs
        param_widgets = {}
        for param_name, param_type in params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param_name}:"))
            
            if param_type == "int":
                widget = QSpinBox()
                widget.setRange(1, 1000)
            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setRange(0.0001, 1000.0)
                widget.setSingleStep(0.1)
            elif param_type == "checkbox":
                widget = QCheckBox()
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems(param_type)
            
            param_layout.addWidget(widget)
            param_widgets[param_name] = widget
            layout.addLayout(param_layout)
        
        # Add train button
        train_btn = QPushButton(f"Train {name}")
        train_btn.clicked.connect(lambda: self.train_model(name, param_widgets))
        layout.addWidget(train_btn)
        
        group.setLayout(layout)
        return group

    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)
       
    def create_deep_learning_tab(self):
        """Create the deep learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # MLP section
        mlp_group = QGroupBox("Multi-Layer Perceptron")
        mlp_layout = QVBoxLayout()
        
        # Layer configuration
        self.layer_config = []
        layer_btn = QPushButton("Add Layer")
        layer_btn.clicked.connect(self.add_layer_dialog)
        mlp_layout.addWidget(layer_btn)
        
        # Training parameters
        training_params_group = self.create_training_params_group()
        mlp_layout.addWidget(training_params_group)
        
        # Train button
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        mlp_layout.addWidget(train_btn)
        
        mlp_group.setLayout(mlp_layout)
        layout.addWidget(mlp_group, 0, 0)
        
        # CNN section
        cnn_group = QGroupBox("Convolutional Neural Network")
        cnn_layout = QVBoxLayout()
        
        # CNN architecture controls
        cnn_controls = self.create_cnn_controls()
        cnn_layout.addWidget(cnn_controls)
        
        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group, 0, 1)
        
        # RNN section
        rnn_group = QGroupBox("Recurrent Neural Network")
        rnn_layout = QVBoxLayout()
        
        # RNN architecture controls
        rnn_controls = self.create_rnn_controls()
        rnn_layout.addWidget(rnn_controls)
        
        rnn_group.setLayout(rnn_layout)
        layout.addWidget(rnn_group, 1, 0)
        
        return widget
    
    def add_layer_dialog(self):
        """Open a dialog to add a neural network layer"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        layout = QVBoxLayout(dialog)
        
        # Layer type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems(["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)
        
        # Parameters input
        params_group = QGroupBox("Layer Parameters")
        params_layout = QVBoxLayout()
        
        # Dynamic parameter inputs based on layer type
        self.layer_param_inputs = {}
        
        def update_params():
            # Clear existing parameter inputs
            for widget in list(self.layer_param_inputs.values()):
                params_layout.removeWidget(widget)
                widget.deleteLater()
            self.layer_param_inputs.clear()
            
            layer_type = type_combo.currentText()
            if layer_type == "Dense":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1000)
                units_input.setValue(32)
                self.layer_param_inputs["units"] = units_input
                
                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax"])
                self.layer_param_inputs["activation"] = activation_combo
                
                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)
            
            elif layer_type == "Conv2D":
                filters_label = QLabel("Filters:")
                filters_input = QSpinBox()
                filters_input.setRange(1, 1000)
                filters_input.setValue(32)
                self.layer_param_inputs["filters"] = filters_input
                
                kernel_label = QLabel("Kernel Size:")
                kernel_input = QLineEdit()
                kernel_input.setText("3, 3")
                self.layer_param_inputs["kernel_size"] = kernel_input
                
                params_layout.addWidget(filters_label)
                params_layout.addWidget(filters_input)
                params_layout.addWidget(kernel_label)
                params_layout.addWidget(kernel_input)
            
            elif layer_type == "Dropout":
                rate_label = QLabel("Dropout Rate:")
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 1.0)
                rate_input.setValue(0.5)
                rate_input.setSingleStep(0.1)
                self.layer_param_inputs["rate"] = rate_input
                
                params_layout.addWidget(rate_label)
                params_layout.addWidget(rate_input)
        
        type_combo.currentIndexChanged.connect(update_params)
        update_params()  # Initial update
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        def add_layer():
            layer_type = type_combo.currentText()
            
            # Collect parameters
            layer_params = {}
            for param_name, widget in self.layer_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    layer_params[param_name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    # Handle kernel size or other tuple-like inputs
                    if param_name == "kernel_size":
                        layer_params[param_name] = tuple(map(int, widget.text().split(',')))
            
            self.layer_config.append({
                "type": layer_type,
                "params": layer_params
            })
            
            dialog.accept()
        
        add_btn.clicked.connect(add_layer)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec()
    
    def create_training_params_group(self):
        """Create group for neural network training parameters"""
        group = QGroupBox("Training Parameters")
        layout = QVBoxLayout()
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1000)
        self.batch_size_spin.setValue(32)
        batch_layout.addWidget(self.batch_size_spin)
        layout.addLayout(batch_layout)
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)
        
        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.001)
        lr_layout.addWidget(self.lr_spin)
        layout.addLayout(lr_layout)
        
        group.setLayout(layout)
        return group
    
    def create_cnn_controls(self):
        """Create controls for Convolutional Neural Network"""
        group = QGroupBox("CNN Architecture")
        layout = QVBoxLayout()
        
        # Placeholder for CNN-specific controls
        label = QLabel("CNN Controls (To be implemented)")
        layout.addWidget(label)
        
        group.setLayout(layout)
        return group
    
    def create_rnn_controls(self):
        """Create controls for Recurrent Neural Network"""
        group = QGroupBox("RNN Architecture")
        layout = QVBoxLayout()
        
        # Placeholder for RNN-specific controls
        label = QLabel("RNN Controls (To be implemented)")
        layout.addWidget(label)
        
        group.setLayout(layout)
        return group
    
    def train_neural_network(self):
        """Train the neural network with current configuration"""
        if not self.layer_config:
            self.show_error("Please add at least one layer to the network")
            return
        
        try:
            # Create and compile model
            model = self.create_neural_network()
            
            # Get training parameters
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()
            
            # Prepare data for neural network
            if len(self.X_train.shape) == 1:
                X_train = self.X_train.reshape(-1, 1)
                X_test = self.X_test.reshape(-1, 1)
            else:
                X_train = self.X_train
                X_test = self.X_test
            
            # One-hot encode target for classification
            y_train = tf.keras.utils.to_categorical(self.y_train)
            y_test = tf.keras.utils.to_categorical(self.y_test)
            
            # Compile model
            optimizer = optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            
            # Train model
            history = model.fit(X_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=(X_test, y_test),
                                callbacks=[self.create_progress_callback()])
            
            # Update visualization with training history
            self.plot_training_history(history)
            
            self.status_bar.showMessage("Neural Network Training Complete")
            
        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")
    
    def create_neural_network(self):
        """Create neural network based on current configuration"""
        model = models.Sequential()
        
        # Add layers based on configuration
        for layer_config in self.layer_config:
            layer_type = layer_config["type"]
            params = layer_config["params"]
            
            if layer_type == "Dense":
                model.add(layers.Dense(**params))
            elif layer_type == "Conv2D":
                # Add input shape for the first layer
                if len(model.layers) == 0:
                    params['input_shape'] = self.X_train.shape[1:]
                model.add(layers.Conv2D(**params))
            elif layer_type == "MaxPooling2D":
                model.add(layers.MaxPooling2D())
            elif layer_type == "Flatten":
                model.add(layers.Flatten())
            elif layer_type == "Dropout":
                model.add(layers.Dropout(**params))
        
        # Add output layer based on number of classes
        num_classes = len(np.unique(self.y_train))
        model.add(layers.Dense(num_classes, activation='softmax'))
                
        return model

   
        
    def train_neural_network(self):
        """Train the neural network"""
        try:
            # Create and compile model
            model = self.create_neural_network()
            
            # Get training parameters
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()
            
            # Compile model
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
            
            # Train model
            history = model.fit(self.X_train, self.y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(self.X_test, self.y_test),
                              callbacks=[self.create_progress_callback()])
            
            # Update visualization with training history
            self.plot_training_history(history)
            
        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")
            
    def create_progress_callback(self):
        """Create callback for updating progress bar during training"""
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, progress_bar):
                super().__init__()
                self.progress_bar = progress_bar
                
            def on_epoch_end(self, epoch, logs=None):
                progress = int(((epoch + 1) / self.params['epochs']) * 100)
                self.progress_bar.setValue(progress)
                
        return ProgressCallback(self.progress_bar)
        
    def update_visualization(self, y_pred):
        """Update the visualization with current results"""
        self.figure.clear()
        
        # Create appropriate visualization based on data
        if len(np.unique(self.y_test)) > 10:  # Regression
            ax = self.figure.add_subplot(111)
            ax.scatter(self.y_test, y_pred)
            ax.plot([self.y_test.min(), self.y_test.max()],
                   [self.y_test.min(), self.y_test.max()],
                   'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            
        else:  # Classification
            if self.X_train.shape[1] > 2:  # Use PCA for visualization
                pca = PCA(n_components=2)
                X_test_2d = pca.fit_transform(self.X_test)
                
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1],
                                   c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)
                
            else:  # Direct 2D visualization
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(self.X_test[:, 0], self.X_test[:, 1],
                                   c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)
        
        self.canvas.draw()
        
    def update_metrics(self, y_pred, loss):
        """Update metrics display with loss value."""
        metrics_text = "Model Performance Metrics:\n\n"
        metrics_text += f"Loss: {loss:.4f}\n"
    
        if len(np.unique(self.y_test)) > 10:  # Regression
            r2 = self.current_model.score(self.X_test, self.y_test)
            metrics_text += f"R² Score: {r2:.4f}"
        else:  # Classification
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            metrics_text += f"Accuracy: {accuracy:.4f}\n\n"
            metrics_text += "Confusion Matrix:\n"
            metrics_text += str(conf_matrix)
    
        self.metrics_text.setText(metrics_text)
        
    def plot_training_history(self, history):
        """Plot neural network training history"""
        self.figure.clear()
        
        # Plot training & validation accuracy
        ax1 = self.figure.add_subplot(211)
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Test'])
        
        # Plot training & validation loss
        ax2 = self.figure.add_subplot(212)
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Test'])
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)

def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()

