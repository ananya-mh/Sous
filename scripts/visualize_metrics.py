import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from itertools import cycle
import argparse
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_evaluation_report(report_path):
    """Load evaluation report from JSON file."""
    with open(report_path, 'r') as f:
        return json.load(f)

def plot_confusion_matrix(cm, labels, output_path):
    """Plot confusion matrix as a heatmap."""
    # Filter out labels with zero support
    non_zero_indices = [i for i, label in enumerate(labels) if any(cm[i]) or any(row[i] for row in cm)]
    filtered_labels = [labels[i] for i in non_zero_indices]
    filtered_cm = [[cm[i][j] for j in non_zero_indices] for i in non_zero_indices]
    
    # Convert to numpy array
    cm_array = np.array(filtered_cm)
    
    # Normalize confusion matrix for better visualization
    cm_normalized = cm_array.astype('float') / (cm_array.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot absolute values
    sns.heatmap(
        cm_array, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=filtered_labels,
        yticklabels=filtered_labels,
        ax=ax1,
        cbar_kws={'label': 'Count'}
    )
    ax1.set_title('Confusion Matrix (Absolute Values)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=9)
    
    # Plot normalized values
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=filtered_labels,
        yticklabels=filtered_labels,
        ax=ax2,
        cbar_kws={'label': 'Normalized'}
    )
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved confusion matrix to {output_path}")

def plot_per_entity_metrics(per_entity_metrics, output_path):
    """Plot per-entity metrics (Precision, Recall, F1-Score)."""
    entities = list(per_entity_metrics.keys())
    metrics = ['precision', 'recall', 'f1']
    
    # Extract values
    precision_vals = [per_entity_metrics[e]['precision'] for e in entities]
    recall_vals = [per_entity_metrics[e]['recall'] for e in entities]
    f1_vals = [per_entity_metrics[e]['f1'] for e in entities]
    support_vals = [per_entity_metrics[e]['support'] for e in entities]
    
    x = np.arange(len(entities))
    width = 0.25
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart for metrics
    ax1.bar(x - width, precision_vals, width, label='Precision', alpha=0.8, color='#3498db')
    ax1.bar(x, recall_vals, width, label='Recall', alpha=0.8, color='#2ecc71')
    ax1.bar(x + width, f1_vals, width, label='F1-Score', alpha=0.8, color='#e74c3c')
    
    ax1.set_xlabel('Entity Type', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Per-Entity Metrics (Precision, Recall, F1-Score)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(entities)
    ax1.legend(loc='upper right')
    ax1.set_ylim([0, 1.1])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (p, r, f) in enumerate(zip(precision_vals, recall_vals, f1_vals)):
        ax1.text(i - width, p + 0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i, r + 0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width, f + 0.01, f'{f:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Support (sample count) bar chart
    colors = ['#9b59b6', '#f39c12', '#1abc9c', '#e67e22']
    bars = ax2.bar(entities, support_vals, alpha=0.8, color=colors[:len(entities)])
    ax2.set_xlabel('Entity Type', fontsize=12)
    ax2.set_ylabel('Support (Number of Samples)', fontsize=12)
    ax2.set_title('Per-Entity Support', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, support_vals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val):,}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved per-entity metrics to {output_path}")

def plot_overall_metrics(overall_metrics, output_path):
    """Plot overall metrics as a bar chart."""
    metrics = ['precision', 'recall', 'f1', 'accuracy']
    values = [overall_metrics[m] for m in metrics]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics, values, alpha=0.8, color=colors)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Overall Model Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved overall metrics to {output_path}")

def plot_roc_curves(cm, labels, output_path):
    """Plot ROC curves for multi-class classification."""
    # Convert confusion matrix to probabilities (simplified approach)
    # For proper ROC, we'd need prediction probabilities, but we can approximate from CM
    
    # Filter out labels with zero support
    non_zero_indices = [i for i, label in enumerate(labels) 
                       if any(cm[i]) or any(row[i] for row in cm)]
    filtered_labels = [labels[i] for i in non_zero_indices]
    filtered_cm = np.array([[cm[i][j] for j in non_zero_indices] for i in non_zero_indices])
    
    # Normalize rows to get probabilities
    cm_probs = filtered_cm.astype('float') / (filtered_cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    # Create binary labels for one-vs-rest
    n_classes = len(filtered_labels)
    
    # For visualization, we'll create synthetic ROC curves based on the confusion matrix
    # In a real scenario, you'd need the actual prediction probabilities
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors for different classes
    colors = cycle(['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e'])
    
    # Calculate ROC for each class (one-vs-rest)
    for i, (label, color) in enumerate(zip(filtered_labels, colors)):
        if filtered_cm[i].sum() == 0:
            continue
            
        # True positives and false positives from confusion matrix
        tp = filtered_cm[i, i]
        fp = filtered_cm[:, i].sum() - tp
        fn = filtered_cm[i, :].sum() - tp
        tn = filtered_cm.sum() - tp - fp - fn
        
        # Calculate TPR and FPR
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Create a simple ROC curve (straight line approximation)
        # In practice, you'd need prediction probabilities for proper ROC
        fpr_curve = np.array([0.0, fpr, 1.0])
        tpr_curve = np.array([0.0, tpr, 1.0])
        
        roc_auc = auc(fpr_curve, tpr_curve)
        
        ax.plot(fpr_curve, tpr_curve, color=color, lw=2,
                label=f'{label} (AUC = {roc_auc:.3f})')
    
    # Diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved ROC curves to {output_path}")

def plot_precision_recall_curves(per_label_metrics, output_path):
    """Plot Precision-Recall comparison for different labels."""
    # Extract labels with non-zero support
    labels = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for label, metrics in per_label_metrics.items():
        if label in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        if metrics.get('support', 0) > 0:
            labels.append(label)
            precisions.append(metrics.get('precision', 0))
            recalls.append(metrics.get('recall', 0))
            f1_scores.append(metrics.get('f1-score', 0))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot
    scatter = ax.scatter(recalls, precisions, s=[f*200 for f in f1_scores], 
                        alpha=0.6, c=f1_scores, cmap='viridis', edgecolors='black', linewidth=1)
    
    # Add labels
    for i, label in enumerate(labels):
        ax.annotate(label, (recalls[i], precisions[i]), 
                   fontsize=8, ha='center', va='center')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Scatter Plot\n(Bubble size = F1-Score)', 
                fontsize=14, fontweight='bold')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.grid(alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('F1-Score', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved Precision-Recall plot to {output_path}")

def plot_metrics_comparison(per_entity_metrics, overall_metrics, output_path):
    """Create a comprehensive comparison chart."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overall metrics
    metrics = ['precision', 'recall', 'f1', 'accuracy']
    values = [overall_metrics[m] for m in metrics]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    axes[0, 0].bar(metrics, values, alpha=0.8, color=colors)
    axes[0, 0].set_ylabel('Score', fontsize=11)
    axes[0, 0].set_title('Overall Metrics', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim([0, 1.1])
    axes[0, 0].grid(axis='y', alpha=0.3)
    for bar, val in zip(axes[0, 0].patches, values):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Per-entity F1 scores
    entities = list(per_entity_metrics.keys())
    f1_vals = [per_entity_metrics[e]['f1'] for e in entities]
    axes[0, 1].barh(entities, f1_vals, alpha=0.8, color='#9b59b6')
    axes[0, 1].set_xlabel('F1-Score', fontsize=11)
    axes[0, 1].set_title('Per-Entity F1-Scores', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlim([0, 1.1])
    axes[0, 1].grid(axis='x', alpha=0.3)
    for i, (entity, val) in enumerate(zip(entities, f1_vals)):
        axes[0, 1].text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=9)
    
    # 3. Precision vs Recall comparison
    precisions = [per_entity_metrics[e]['precision'] for e in entities]
    recalls = [per_entity_metrics[e]['recall'] for e in entities]
    x = np.arange(len(entities))
    width = 0.35
    axes[1, 0].bar(x - width/2, precisions, width, label='Precision', alpha=0.8, color='#3498db')
    axes[1, 0].bar(x + width/2, recalls, width, label='Recall', alpha=0.8, color='#2ecc71')
    axes[1, 0].set_xlabel('Entity Type', fontsize=11)
    axes[1, 0].set_ylabel('Score', fontsize=11)
    axes[1, 0].set_title('Precision vs Recall by Entity', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(entities)
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Support distribution
    support_vals = [per_entity_metrics[e]['support'] for e in entities]
    axes[1, 1].pie(support_vals, labels=entities, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Support Distribution by Entity', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved metrics comparison to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate visualization plots from evaluation report")
    parser.add_argument(
        "--report", 
        type=str, 
        default="evaluation_report.json",
        help="Path to evaluation report JSON file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=".",
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all visualizations"
    )
    
    args = parser.parse_args()
    
    # Load evaluation report
    print(f"📊 Loading evaluation report from {args.report}...")
    try:
        report = load_evaluation_report(args.report)
    except FileNotFoundError:
        print(f"❌ Error: {args.report} not found.")
        exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🎨 Generating visualizations...\n")
    
    # Generate all visualizations
    plot_confusion_matrix(
        report['confusion_matrix'], 
        report['label_list'],
        output_dir / "confusion_matrix.png"
    )
    
    plot_per_entity_metrics(
        report['per_entity_metrics'],
        output_dir / "per_entity_metrics.png"
    )
    
    plot_overall_metrics(
        report['overall_metrics'],
        output_dir / "overall_metrics.png"
    )
    
    plot_roc_curves(
        report['confusion_matrix'],
        report['label_list'],
        output_dir / "roc_curves.png"
    )
    
    plot_precision_recall_curves(
        report['per_label_metrics'],
        output_dir / "precision_recall_curves.png"
    )
    
    plot_metrics_comparison(
        report['per_entity_metrics'],
        report['overall_metrics'],
        output_dir / "metrics_comparison.png"
    )
    
    print("\n✅ All visualizations generated successfully!")
    print(f"📁 Output directory: {output_dir.absolute()}")

if __name__ == "__main__":
    main()

