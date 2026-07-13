# ============================================================
# EVALUATION MODULE: Model Performance Metrics & Visualization
# Generates Accuracy, Precision, Recall, F1, Confusion Matrix,
# ROC-AUC curves, and model comparison charts.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive evaluation of ML diagnostic models.
    Generates metrics, confusion matrices, ROC-AUC curves,
    and comparison charts across all models.
    """

    def __init__(self, ml_classifier):
        """
        Args:
            ml_classifier: An instance of MLDiagnosticClassifier
                           (already trained or will be trained).
        """
        self.ml = ml_classifier
        if not self.ml.is_trained:
            self.ml.train(verbose=False)

    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute Accuracy, Precision, Recall, F1 for each model"""
        # Generate test data
        df = self.ml._generate_synthetic_data(2000)
        X = df[self.ml.SYMPTOM_FEATURES].values
        y = self.ml.label_encoder.transform(df['disease'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        results = {}

        for name, model in self.ml.models.items():
            # Re-fit on training data to ensure consistency
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results[name] = {
                'accuracy':  round(accuracy_score(y_test, y_pred), 4),
                'precision': round(precision_score(
                    y_test, y_pred, average='weighted'), 4),
                'recall':    round(recall_score(
                    y_test, y_pred, average='weighted'), 4),
                'f1_score':  round(f1_score(
                    y_test, y_pred, average='weighted'), 4),
            }

        self._X_test = X_test
        self._y_test = y_test
        self._X_train = X_train
        self._y_train = y_train
        return results

    def print_metrics_table(self, results: Dict[str, Dict[str, float]]):
        """Print a formatted metrics table"""
        print(f"\n  {'Model':<22} {'Accuracy':<10} {'Precision':<10} "
              f"{'Recall':<10} {'F1-Score':<10}")
        print(f"  {'─'*62}")
        for name, metrics in results.items():
            print(f"  {name:<22} {metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} "
                  f"{metrics['f1_score']:<10.4f}")

    def plot_confusion_matrices(self):
        """Generate confusion matrix heatmaps for all models"""
        labels = self.ml.label_encoder.classes_
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))

        for ax, (name, model) in zip(axes, self.ml.models.items()):
            model.fit(self._X_train, self._y_train)
            y_pred = model.predict(self._X_test)
            cm = confusion_matrix(self._y_test, y_pred)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_title(f"{name}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            plt.setp(ax.xaxis.get_majorticklabels(),
                     rotation=45, ha='right')

        plt.suptitle("Confusion Matrices — All ML Models",
                     fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig("confusion_matrices.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✅ Saved: confusion_matrices.png")

    def plot_roc_curves(self):
        """Generate ROC-AUC curves (one-vs-rest for multiclass)"""
        labels = self.ml.label_encoder.classes_
        n_classes = len(labels)
        y_test_bin = label_binarize(self._y_test, classes=range(n_classes))

        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

        for ax, (name, model) in zip(axes, self.ml.models.items()):
            model.fit(self._X_train, self._y_train)

            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(self._X_test)
            else:
                y_score = model.decision_function(self._X_test)

            # Plot ROC for each class
            for i, (label, color) in enumerate(zip(labels, colors)):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, linewidth=1.5,
                        label=f"{label} (AUC={roc_auc:.2f})")

            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
            ax.set_title(f"{name}", fontsize=12, fontweight='bold')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc='lower right', fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.suptitle("ROC-AUC Curves — One-vs-Rest (All Models)",
                     fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig("roc_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✅ Saved: roc_curves.png")

    def plot_model_comparison(self, results: Dict[str, Dict[str, float]]):
        """Generate a bar chart comparing all models across metrics"""
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(results.keys())
        n_models = len(model_names)
        n_metrics = len(metrics_names)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(n_metrics)
        width = 0.25
        colors = ['#3498db', '#2ecc71', '#e74c3c']

        for i, model in enumerate(model_names):
            values = [results[model][m] for m in metrics_names]
            bars = ax.bar(x + i * width, values, width,
                          label=model, color=colors[i], alpha=0.85)
            # Add value labels on top of bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.3f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

        ax.set_xlabel("Metric", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Comparison — All Metrics",
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(
            ['Accuracy', 'Precision', 'Recall', 'F1-Score'], fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig("model_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✅ Saved: model_comparison.png")

    def plot_per_class_metrics(self):
        """Generate per-class precision/recall/f1 for the best model"""
        best_model = self.ml.best_model
        best_name = self.ml.best_model_name
        labels = self.ml.label_encoder.classes_

        best_model.fit(self._X_train, self._y_train)
        y_pred = best_model.predict(self._X_test)

        report = classification_report(
            self._y_test, y_pred,
            target_names=labels, output_dict=True)

        # Extract per-class metrics
        diseases = [d for d in labels if d in report]
        precision_vals = [report[d]['precision'] for d in diseases]
        recall_vals = [report[d]['recall'] for d in diseases]
        f1_vals = [report[d]['f1-score'] for d in diseases]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(diseases))
        width = 0.25

        ax.bar(x - width, precision_vals, width,
               label='Precision', color='#3498db', alpha=0.85)
        ax.bar(x, recall_vals, width,
               label='Recall', color='#2ecc71', alpha=0.85)
        ax.bar(x + width, f1_vals, width,
               label='F1-Score', color='#e74c3c', alpha=0.85)

        ax.set_xlabel("Disease", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(f"Per-Class Metrics — {best_name}",
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(diseases, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig("per_class_metrics.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✅ Saved: per_class_metrics.png")

    def run_full_evaluation(self):
        """Run the complete evaluation pipeline"""
        print("\n  ── Computing Metrics ──")
        results = self.compute_metrics()
        self.print_metrics_table(results)

        print("\n  ── Generating Confusion Matrices ──")
        self.plot_confusion_matrices()

        print("\n  ── Generating ROC-AUC Curves ──")
        self.plot_roc_curves()

        print("\n  ── Generating Model Comparison Chart ──")
        self.plot_model_comparison(results)

        print("\n  ── Generating Per-Class Metrics ──")
        self.plot_per_class_metrics()

        print(f"\n  ✅ Full evaluation complete! Charts saved:")
        print(f"     • confusion_matrices.png")
        print(f"     • roc_curves.png")
        print(f"     • model_comparison.png")
        print(f"     • per_class_metrics.png")
