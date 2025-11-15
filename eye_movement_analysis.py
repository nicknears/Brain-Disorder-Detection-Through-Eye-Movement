"""
Eye Movement Analysis for Brain Disorder Detection
===================================================

This module provides comprehensive tools for analyzing eye movement data
collected from wearable sensors to detect brain disorders.

Author: Research Team
Date: 2024
License: MIT
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class EyeMovementAnalyzer:
    """
    A comprehensive class for analyzing eye movement data from wearable sensors.
    
    This class provides methods for data preprocessing, feature extraction,
    statistical analysis, and machine learning-based classification.
    """
    
    def __init__(self, sampling_rate: int = 1000):
        """
        Initialize the EyeMovementAnalyzer.
        
        Parameters:
        -----------
        sampling_rate : int
            Sampling rate of the eye movement data in Hz (default: 1000)
        """
        self.sampling_rate = sampling_rate
        self.scaler = StandardScaler()
        self.pca = None
        self.model = None
        self.feature_names = []
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load eye movement data from a CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file containing eye movement data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the eye movement data
        """
        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded successfully: {data.shape[0]} samples, {data.shape[1]} features")
            return data
        except FileNotFoundError:
            print(f"Error: File {filepath} not found")
            raise
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame, 
                       lowcut: float = 0.1, 
                       highcut: float = 50.0) -> pd.DataFrame:
        """
        Preprocess eye movement data by filtering and removing artifacts.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw eye movement data
        lowcut : float
            Low cutoff frequency for bandpass filter (Hz)
        highcut : float
            High cutoff frequency for bandpass filter (Hz)
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed data
        """
        processed_data = data.copy()
        
        # Apply bandpass filter to remove noise
        if 'x_position' in processed_data.columns and 'y_position' in processed_data.columns:
            nyquist = self.sampling_rate / 2
            low = lowcut / nyquist
            high = highcut / nyquist
            
            b, a = butter(4, [low, high], btype='band')
            
            processed_data['x_position'] = filtfilt(b, a, processed_data['x_position'])
            processed_data['y_position'] = filtfilt(b, a, processed_data['y_position'])
        
        # Remove outliers (values beyond 3 standard deviations)
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean = processed_data[col].mean()
            std = processed_data[col].std()
            processed_data = processed_data[
                (processed_data[col] >= mean - 3*std) & 
                (processed_data[col] <= mean + 3*std)
            ]
        
        print(f"Data preprocessed: {processed_data.shape[0]} samples remaining")
        return processed_data
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from eye movement data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Preprocessed eye movement data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing extracted features
        """
        features = []
        
        if 'x_position' in data.columns and 'y_position' in data.columns:
            # Calculate velocity
            x_velocity = np.diff(data['x_position']) * self.sampling_rate
            y_velocity = np.diff(data['y_position']) * self.sampling_rate
            velocity = np.sqrt(x_velocity**2 + y_velocity**2)
            
            # Calculate acceleration
            acceleration = np.diff(velocity) * self.sampling_rate
            
            # Saccade detection (rapid eye movements)
            saccade_threshold = np.percentile(velocity, 95)
            saccades = velocity > saccade_threshold
            
            # Extract features
            features_dict = {
                'mean_velocity': np.mean(velocity),
                'std_velocity': np.std(velocity),
                'max_velocity': np.max(velocity),
                'mean_acceleration': np.mean(acceleration),
                'std_acceleration': np.std(acceleration),
                'saccade_count': np.sum(saccades),
                'saccade_rate': np.sum(saccades) / len(velocity),
                'mean_fixation_duration': self._calculate_fixation_duration(velocity, saccade_threshold),
                'fixation_stability': self._calculate_fixation_stability(data),
            }
            
            features.append(features_dict)
        
        # Add statistical features for other columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['x_position', 'y_position']:
                features[0][f'{col}_mean'] = data[col].mean()
                features[0][f'{col}_std'] = data[col].std()
                features[0][f'{col}_median'] = data[col].median()
        
        features_df = pd.DataFrame(features)
        self.feature_names = features_df.columns.tolist()
        
        return features_df
    
    def _calculate_fixation_duration(self, velocity: np.ndarray, 
                                     threshold: float) -> float:
        """Calculate mean fixation duration."""
        fixations = velocity <= threshold
        fixation_durations = []
        current_duration = 0
        
        for is_fixation in fixations:
            if is_fixation:
                current_duration += 1
            else:
                if current_duration > 0:
                    fixation_durations.append(current_duration / self.sampling_rate)
                current_duration = 0
        
        if current_duration > 0:
            fixation_durations.append(current_duration / self.sampling_rate)
        
        return np.mean(fixation_durations) if fixation_durations else 0.0
    
    def _calculate_fixation_stability(self, data: pd.DataFrame) -> float:
        """Calculate fixation stability as inverse of position variance."""
        if 'x_position' in data.columns and 'y_position' in data.columns:
            position_variance = np.var(data[['x_position', 'y_position']].values)
            return 1.0 / (1.0 + position_variance)  # Inverse variance, bounded
        return 0.0
    
    def perform_statistical_analysis(self, features: pd.DataFrame, 
                                    labels: pd.Series) -> Dict:
        """
        Perform statistical analysis comparing groups.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Extracted features
        labels : pd.Series
            Group labels (e.g., 'healthy' vs 'disorder')
            
        Returns:
        --------
        Dict
            Dictionary containing statistical test results
        """
        results = {}
        unique_labels = labels.unique()
        
        if len(unique_labels) != 2:
            print("Warning: Statistical analysis requires exactly 2 groups")
            return results
        
        group1, group2 = unique_labels
        group1_data = features[labels == group1]
        group2_data = features[labels == group2]
        
        for feature in features.columns:
            # Perform t-test
            stat, p_value = stats.ttest_ind(
                group1_data[feature].dropna(),
                group2_data[feature].dropna()
            )
            
            # Calculate descriptive statistics
            results[feature] = {
                't_statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                f'{group1}_mean': group1_data[feature].mean(),
                f'{group1}_std': group1_data[feature].std(),
                f'{group2}_mean': group2_data[feature].mean(),
                f'{group2}_std': group2_data[feature].std(),
            }
        
        return results
    
    def apply_pca(self, features: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
        """
        Apply Principal Component Analysis for dimensionality reduction.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix
        n_components : int
            Number of principal components to retain
            
        Returns:
        --------
        pd.DataFrame
            Transformed features
        """
        self.pca = PCA(n_components=n_components)
        features_scaled = self.scaler.fit_transform(features)
        features_pca = self.pca.fit_transform(features_scaled)
        
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        print(f"PCA applied: {n_components} components explain {explained_variance:.2%} of variance")
        
        return pd.DataFrame(
            features_pca,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
    
    def train_classifier(self, features: pd.DataFrame, 
                        labels: pd.Series,
                        model_type: str = 'random_forest',
                        test_size: float = 0.2) -> Dict:
        """
        Train a machine learning classifier.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix
        labels : pd.Series
            Class labels
        model_type : str
            Type of classifier ('random_forest' or 'svm')
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        Dict
            Dictionary containing model performance metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'svm':
            self.model = SVC(kernel='rbf', probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'model_type': model_type
        }
        
        print(f"\nModel Performance ({model_type}):")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(results['classification_report'])
        
        return results
    
    def visualize_results(self, features: pd.DataFrame, 
                         labels: pd.Series,
                         save_path: Optional[str] = None):
        """
        Create visualizations of the analysis results.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix
        labels : pd.Series
            Class labels
        save_path : str, optional
            Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Feature distribution comparison
        if len(features.columns) > 0:
            feature = features.columns[0]
            for label in labels.unique():
                data = features[labels == label][feature]
                axes[0, 0].hist(data, alpha=0.7, label=label, bins=30)
            axes[0, 0].set_xlabel(feature)
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Feature Distribution by Group')
            axes[0, 0].legend()
        
        # Correlation matrix
        if len(features.columns) > 1:
            corr_matrix = features.corr()
            sns.heatmap(corr_matrix, ax=axes[0, 1], cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            axes[0, 1].set_title('Feature Correlation Matrix')
        
        # PCA visualization (if applied)
        if self.pca is not None and len(features.columns) >= 2:
            features_scaled = self.scaler.transform(features)
            features_pca = self.pca.transform(features_scaled)
            scatter = axes[1, 0].scatter(features_pca[:, 0], features_pca[:, 1],
                                        c=labels.astype('category').cat.codes,
                                        cmap='viridis', alpha=0.6)
            axes[1, 0].set_xlabel('First Principal Component')
            axes[1, 0].set_ylabel('Second Principal Component')
            axes[1, 0].set_title('PCA Visualization')
            plt.colorbar(scatter, ax=axes[1, 0])
        
        # Box plot for key features
        if len(features.columns) > 0:
            key_features = features.columns[:min(4, len(features.columns))]
            feature_data = []
            feature_names = []
            group_labels = []
            
            for feature in key_features:
                for label in labels.unique():
                    feature_data.extend(features[labels == label][feature].values)
                    feature_names.extend([feature] * len(features[labels == label]))
                    group_labels.extend([label] * len(features[labels == label]))
            
            df_plot = pd.DataFrame({
                'Feature': feature_names,
                'Value': feature_data,
                'Group': group_labels
            })
            
            sns.boxplot(data=df_plot, x='Feature', y='Value', hue='Group', ax=axes[1, 1])
            axes[1, 1].set_title('Feature Comparison by Group')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()


def main():
    """
    Example usage of the EyeMovementAnalyzer class.
    """
    # Initialize analyzer
    analyzer = EyeMovementAnalyzer(sampling_rate=1000)
    
    # Example: Generate synthetic data for demonstration
    print("Generating synthetic eye movement data for demonstration...")
    np.random.seed(42)
    
    # Generate synthetic data for healthy and disorder groups
    n_samples = 100
    time_points = 1000
    
    healthy_data = []
    disorder_data = []
    
    for i in range(n_samples):
        # Healthy: smooth eye movements
        t = np.linspace(0, 1, time_points)
        x_healthy = np.cumsum(np.random.randn(time_points) * 0.01)
        y_healthy = np.cumsum(np.random.randn(time_points) * 0.01)
        
        healthy_data.append({
            'x_position': x_healthy,
            'y_position': y_healthy,
            'subject_id': f'healthy_{i}',
            'group': 'healthy'
        })
        
        # Disorder: more erratic movements
        x_disorder = np.cumsum(np.random.randn(time_points) * 0.05)
        y_disorder = np.cumsum(np.random.randn(time_points) * 0.05)
        
        disorder_data.append({
            'x_position': x_disorder,
            'y_position': y_disorder,
            'subject_id': f'disorder_{i}',
            'group': 'disorder'
        })
    
    # Process data
    all_features = []
    all_labels = []
    
    for data_dict in healthy_data + disorder_data:
        df = pd.DataFrame({
            'x_position': data_dict['x_position'],
            'y_position': data_dict['y_position']
        })
        
        processed = analyzer.preprocess_data(df)
        features = analyzer.extract_features(processed)
        all_features.append(features)
        all_labels.append(data_dict['group'])
    
    features_df = pd.concat(all_features, ignore_index=True)
    labels_series = pd.Series(all_labels)
    
    # Statistical analysis
    print("\nPerforming statistical analysis...")
    stats_results = analyzer.perform_statistical_analysis(features_df, labels_series)
    
    # Print significant features
    print("\nSignificant features (p < 0.05):")
    for feature, result in stats_results.items():
        if result['significant']:
            print(f"{feature}: p = {result['p_value']:.4f}")
    
    # Apply PCA
    print("\nApplying PCA...")
    features_pca = analyzer.apply_pca(features_df, n_components=5)
    
    # Train classifier
    print("\nTraining classifier...")
    model_results = analyzer.train_classifier(features_pca, labels_series)
    
    # Visualize results
    print("\nGenerating visualizations...")
    analyzer.visualize_results(features_pca, labels_series)


if __name__ == "__main__":
    main()

