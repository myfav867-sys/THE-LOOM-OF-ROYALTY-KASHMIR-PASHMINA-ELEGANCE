"""
The Loom of Royalty: Kashmir Pashmina Elegance
Machine Learning Training Pipeline

This script provides a comprehensive machine learning framework for analyzing,
classifying, and predicting characteristics of Kashmir pashmina shawls.

Features:
- Data loading and preprocessing
- Exploratory data analysis
- Feature engineering
- Model training (classification & clustering)
- Model evaluation and validation
- Visualization and reporting
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import warnings

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, silhouette_score
)
from sklearn.decomposition import PCA
import pickle
import json

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pashmina_ml_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PashminaDataProcessor:
    """Handle data loading and preprocessing for pashmina shawl dataset"""
    
    def __init__(self, data_path):
        """
        Initialize data processor
        
        Args:
            data_path (str): Path to the dataset CSV file
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.encoders = {}
        
    def load_data(self):
        """Load dataset from CSV file"""
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Dataset loaded successfully. Shape: {self.df.shape}")
            logger.info(f"Columns: {list(self.df.columns)}")
            return self.df
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {self.data_path}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            sys.exit(1)
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        logger.info("=== Exploratory Data Analysis ===")
        
        # Basic statistics
        logger.info(f"Dataset shape: {self.df.shape}")
        logger.info(f"Data types:\n{self.df.dtypes}")
        logger.info(f"Missing values:\n{self.df.isnull().sum()}")
        logger.info(f"Statistical summary:\n{self.df.describe()}")
        
        # Visualize distributions
        self._plot_distributions()
        
    def _plot_distributions(self):
        """Plot feature distributions"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(
                (len(numeric_cols) + 1) // 2, 2, figsize=(12, 10)
            )
            axes = axes.flatten()
            
            for idx, col in enumerate(numeric_cols):
                axes[idx].hist(self.df[col], bins=30, edgecolor='black')
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig('data_distributions.png', dpi=300, bbox_inches='tight')
            logger.info("Distribution plots saved as 'data_distributions.png'")
            plt.close()
    
    def handle_missing_values(self, strategy='mean'):
        """
        Handle missing values in dataset
        
        Args:
            strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
        """
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        if strategy == 'mean':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'drop':
            self.df = self.df.dropna()
        
        logger.info(f"Missing values after handling: {self.df.isnull().sum().sum()}")
    
    def encode_categorical_features(self):
        """Encode categorical features"""
        logger.info("Encoding categorical features")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.encoders:
                encoder = LabelEncoder()
                self.df[col] = encoder.fit_transform(self.df[col].astype(str))
                self.encoders[col] = encoder
                logger.info(f"Encoded column: {col}")
    
    def feature_engineering(self):
        """Create new features from existing data"""
        logger.info("Performing feature engineering")
        
        # Example: Create interaction features if applicable
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Add polynomial features for first two numeric columns
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            self.df[f'{col1}_x_{col2}'] = self.df[col1] * self.df[col2]
            logger.info(f"Created interaction feature: {col1}_x_{col2}")
    
    def split_data(self, target_col=None, test_size=0.2, random_state=42):
        """
        Split data into train and test sets
        
        Args:
            target_col (str): Target column name for classification
            test_size (float): Proportion of test set
            random_state (int): Random seed
        """
        logger.info(f"Splitting data with test_size={test_size}")
        
        if target_col is None:
            # Use last column as target by default
            target_col = self.df.columns[-1]
        
        if target_col not in self.df.columns:
            logger.error(f"Target column '{target_col}' not found in dataset")
            return False
        
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Training set size: {self.X_train.shape}")
        logger.info(f"Test set size: {self.X_test.shape}")
        
        return True
    
    def scale_features(self):
        """Scale numerical features"""
        logger.info("Scaling features")
        
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        logger.info("Features scaled successfully")


class PashminaModelTrainer:
    """Train and evaluate machine learning models"""
    
    def __init__(self):
        """Initialize model trainer"""
        self.models = {}
        self.results = {}
        
    def train_random_forest(self, X_train, y_train, n_estimators=100):
        """
        Train Random Forest classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators (int): Number of trees
        """
        logger.info("Training Random Forest Classifier")
        
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        self.models['RandomForest'] = rf_model
        
        logger.info("Random Forest training completed")
        return rf_model
    
    def train_gradient_boosting(self, X_train, y_train, n_estimators=100):
        """
        Train Gradient Boosting classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators (int): Number of boosting stages
        """
        logger.info("Training Gradient Boosting Classifier")
        
        gb_model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        gb_model.fit(X_train, y_train)
        self.models['GradientBoosting'] = gb_model
        
        logger.info("Gradient Boosting training completed")
        return gb_model
    
    def train_logistic_regression(self, X_train, y_train):
        """
        Train Logistic Regression classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info("Training Logistic Regression")
        
        lr_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        
        lr_model.fit(X_train, y_train)
        self.models['LogisticRegression'] = lr_model
        
        logger.info("Logistic Regression training completed")
        return lr_model
    
    def train_kmeans_clustering(self, X_train, n_clusters=3):
        """
        Train K-Means clustering model
        
        Args:
            X_train: Training features
            n_clusters (int): Number of clusters
        """
        logger.info(f"Training K-Means Clustering with {n_clusters} clusters")
        
        kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        
        kmeans_model.fit(X_train)
        self.models['KMeans'] = kmeans_model
        
        silhouette_avg = silhouette_score(X_train, kmeans_model.labels_)
        logger.info(f"K-Means Silhouette Score: {silhouette_avg:.4f}")
        
        return kmeans_model
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        logger.info("=== Model Evaluation ===")
        
        for model_name, model in self.models.items():
            if model_name == 'KMeans':
                continue  # Skip evaluation for clustering
            
            logger.info(f"\nEvaluating {model_name}")
            
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            self.results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1-Score: {f1:.4f}")
            
            # Classification report
            logger.info(f"\nClassification Report for {model_name}:")
            logger.info(classification_report(y_test, y_pred))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            self._plot_confusion_matrix(cm, model_name)
    
    def _plot_confusion_matrix(self, cm, model_name):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved as 'confusion_matrix_{model_name}.png'")
        plt.close()
    
    def feature_importance(self, X_train):
        """Display feature importance from tree-based models"""
        logger.info("=== Feature Importance ===")
        
        for model_name in ['RandomForest', 'GradientBoosting']:
            if model_name not in self.models:
                continue
            
            model = self.models[model_name]
            importances = model.feature_importances_
            
            feature_importance_df = pd.DataFrame({
                'feature': range(len(importances)),
                'importance': importances
            }).sort_values('importance', ascending=False).head(10)
            
            logger.info(f"\nTop 10 Features for {model_name}:")
            logger.info(feature_importance_df)
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance_df['feature'].astype(str), feature_importance_df['importance'])
            plt.xlabel('Importance')
            plt.title(f'Top 10 Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig(f'feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved as 'feature_importance_{model_name}.png'")
            plt.close()
    
    def save_models(self, output_dir='models'):
        """Save trained models to disk"""
        logger.info(f"Saving models to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filepath = os.path.join(output_dir, f'{model_name}.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {model_name} to {filepath}")
        
        # Save results summary
        results_filepath = os.path.join(output_dir, 'results_summary.json')
        with open(results_filepath, 'w') as f:
            json.dump(self.results, f, indent=4)
        logger.info(f"Saved results summary to {results_filepath}")


def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("Kashmir Pashmina Shawl ML Training Pipeline")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # Configuration
    DATA_PATH = 'data/pashmina_shawls.csv'  # Update with your dataset path
    TARGET_COLUMN = 'quality_grade'  # Update with your target column
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Step 1: Data Processing
    logger.info("\n### STEP 1: DATA PROCESSING ###")
    processor = PashminaDataProcessor(DATA_PATH)
    processor.load_data()
    processor.explore_data()
    processor.handle_missing_values(strategy='mean')
    processor.encode_categorical_features()
    processor.feature_engineering()
    processor.split_data(target_col=TARGET_COLUMN, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    processor.scale_features()
    
    # Step 2: Model Training
    logger.info("\n### STEP 2: MODEL TRAINING ###")
    trainer = PashminaModelTrainer()
    trainer.train_random_forest(processor.X_train, processor.y_train)
    trainer.train_gradient_boosting(processor.X_train, processor.y_train)
    trainer.train_logistic_regression(processor.X_train, processor.y_train)
    trainer.train_kmeans_clustering(processor.X_train, n_clusters=3)
    
    # Step 3: Model Evaluation
    logger.info("\n### STEP 3: MODEL EVALUATION ###")
    trainer.evaluate_models(processor.X_test, processor.y_test)
    
    # Step 4: Feature Importance
    logger.info("\n### STEP 4: FEATURE IMPORTANCE ###")
    trainer.feature_importance(processor.X_train)
    
    # Step 5: Save Models
    logger.info("\n### STEP 5: SAVING MODELS ###")
    trainer.save_models(output_dir='trained_models')
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Pipeline Completed Successfully!")
    logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # Print results summary
    logger.info("\n### RESULTS SUMMARY ###")
    for model_name, metrics in trainer.results.items():
        logger.info(f"\n{model_name}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
