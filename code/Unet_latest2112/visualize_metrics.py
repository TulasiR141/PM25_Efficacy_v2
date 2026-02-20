import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("📊 Creating comprehensive metrics visualizations...")


def plot_training_vs_test_metrics(val_results, test_results, model_name):
    """Compare validation and test metrics including boundary metrics"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    metrics = ['Dice', 'IoU', 'Accuracy', 'Boundary IoU', 'Boundary F1', 'Area Accuracy']
    val_values = [
        val_results.get('dice', 0.0),
        val_results.get('jaccard_coeff', 0.0),
        val_results.get('accuracy_spheroid', 0.0),
        val_results.get('boundary_iou', 0.0),
        val_results.get('boundary_f1', 0.0),
        val_results.get('area_accuracy', 0.0)
    ]
    test_values = [
        test_results.get('dice_mean', 0.0),
        test_results.get('iou_mean', 0.0),
        test_results.get('accuracy_mean', 0.0),
        test_results.get('boundary_iou_mean', 0.0),
        test_results.get('boundary_f1_mean', 0.0) if 'boundary_f1_mean' in test_results else 0.0,
        test_results.get('area_accuracy_mean', 0.0)
    ]
    test_stds = [
        test_results.get('dice_std', 0.0),
        test_results.get('iou_std', 0.0),
        test_results.get('accuracy_std', 0.0),
        test_results.get('boundary_iou_std', 0.0),
        test_results.get('boundary_f1_std', 0.0) if 'boundary_f1_std' in test_results else 0.0,
        test_results.get('area_accuracy_std', 0.0)
    ]
    
    colors = ['steelblue', 'coral', 'mediumseagreen', 'purple', 'orange', 'teal']
    
    for idx, (ax, metric, val_val, test_val, test_std, color) in enumerate(zip(
        axes, metrics, val_values, test_values, test_stds, colors)):
        
        x = np.arange(1)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, val_val, width, label='Validation', 
                      color=color, alpha=0.6)
        bars2 = ax.bar(x + width/2, test_val, width, label='Test', 
                      color=color, alpha=0.9, yerr=test_std, capsize=5)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([''])
        ax.set_ylim([0.75, 1.0])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        if val_val > 0:
            ax.text(x - width/2, val_val + 0.01, f'{val_val:.3f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        if test_val > 0:
            ax.text(x + width/2, test_val + test_std + 0.01, f'{test_val:.3f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle(f'Validation vs Test Performance - {model_name}', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('metrics_val_vs_test.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: metrics_val_vs_test.png")
    plt.close()


def plot_test_distribution(dice_scores, iou_scores, accuracy_scores, model_name):
    """Plot distribution of metrics across test samples"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_data = [
        (dice_scores, 'Dice Score', 'steelblue'),
        (iou_scores, 'IoU (Jaccard)', 'coral'),
        (accuracy_scores, 'Pixel Accuracy', 'mediumseagreen')
    ]
    
    for ax, (scores, title, color) in zip(axes, metrics_data):
        # Histogram
        ax.hist(scores, bins=20, color=color, alpha=0.7, edgecolor='black')
        
        # Add mean line
        mean_val = np.mean(scores)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.3f}')
        
        # Add median line
        median_val = np.median(scores)
        ax.axvline(median_val, color='orange', linestyle=':', linewidth=2,
                  label=f'Median: {median_val:.3f}')
        
        ax.set_xlabel('Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{title} Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Test Set Metrics Distribution - {model_name}', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('metrics_test_distribution.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: metrics_test_distribution.png")
    plt.close()


def plot_box_plots(dice_scores, iou_scores, accuracy_scores, model_name):
    """Create box plots for test metrics"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = [dice_scores, iou_scores, accuracy_scores]
    labels = ['Dice', 'IoU', 'Accuracy']
    colors = ['steelblue', 'coral', 'mediumseagreen']
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True, 
                    notch=True, showmeans=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], linewidth=1.5)
    
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title(f'Test Set Metrics Summary - {model_name}', 
                fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.8, 1.0])
    
    # Add mean values as text
    for i, (scores, label) in enumerate(zip(data, labels), 1):
        mean_val = np.mean(scores)
        ax.text(i, mean_val + 0.01, f'{mean_val:.3f}', 
               ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('metrics_boxplots.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: metrics_boxplots.png")
    plt.close()


def plot_per_sample_metrics(dice_scores, iou_scores, accuracy_scores, model_name):
    """Plot metrics for each test sample"""
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(dice_scores))
    
    ax.plot(x, dice_scores, 'o-', label='Dice', linewidth=2, markersize=6, color='steelblue')
    ax.plot(x, iou_scores, 's-', label='IoU', linewidth=2, markersize=6, color='coral')
    ax.plot(x, accuracy_scores, '^-', label='Accuracy', linewidth=2, markersize=6, color='mediumseagreen')
    
    # Add mean lines
    ax.axhline(np.mean(dice_scores), color='steelblue', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(np.mean(iou_scores), color='coral', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(np.mean(accuracy_scores), color='mediumseagreen', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Test Sample Index', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Per-Sample Metrics - {model_name}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_ylim([0.85, 1.0])
    
    plt.tight_layout()
    plt.savefig('metrics_per_sample.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: metrics_per_sample.png")
    plt.close()


def plot_correlation_heatmap(dice_scores, iou_scores, accuracy_scores, model_name):
    """Plot correlation between metrics"""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create correlation matrix
    data_matrix = np.array([dice_scores, iou_scores, accuracy_scores])
    corr_matrix = np.corrcoef(data_matrix)
    
    # Plot heatmap
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks and labels
    labels = ['Dice', 'IoU', 'Accuracy']
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    
    # Add correlation values
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                         ha="center", va="center", color="black", fontsize=14, fontweight='bold')
    
    ax.set_title(f'Metrics Correlation - {model_name}', fontsize=16, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('metrics_correlation.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: metrics_correlation.png")
    plt.close()


def create_summary_table(val_results, test_results, model_name):
    """Create a summary comparison table"""
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    table_data = [
        ['Metric', 'Validation', 'Test (Mean ± Std)', 'Test Range'],
        ['Dice Score', 
         f"{val_results.get('dice', 0.9579):.4f}", 
         f"{test_results['dice_mean']:.4f} ± {test_results['dice_std']:.4f}",
         f"[{test_results['dice_min']:.4f}, {test_results['dice_max']:.4f}]"],
        ['IoU (Jaccard)', 
         f"{val_results.get('jaccard_coeff', 0.9191):.4f}", 
         f"{test_results['iou_mean']:.4f} ± {test_results['iou_std']:.4f}",
         f"[{test_results['iou_min']:.4f}, {test_results['iou_max']:.4f}]"],
        ['Pixel Accuracy', 
         f"{val_results.get('accuracy_spheroid', 0.9895):.4f}", 
         f"{test_results['accuracy_mean']:.4f} ± {test_results['accuracy_std']:.4f}",
         f"[{test_results['accuracy_min']:.4f}, {test_results['accuracy_max']:.4f}]"],
        ['# Samples', 
         f"{val_results.get('samples', 210)}", 
         f"{test_results['num_samples']}", 
         '-'],
        ['Inference Time', 
         f"{val_results.get('inference_time', 328.18):.2f}s", 
         f"{test_results['inference_time']:.2f}s", 
         f"{test_results['avg_time_per_image']:.3f}s/img"]
    ]
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.2, 0.3, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    plt.title(f'Performance Summary - {model_name}', 
             fontsize=16, fontweight='bold', pad=20)
    plt.savefig('metrics_summary_table.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: metrics_summary_table.png")
    plt.close()


def generate_all_visualizations(val_results, test_results, dice_scores, 
                                iou_scores, accuracy_scores, model_name):
    """Generate all visualization plots"""
    
    print("\n📊 Generating all visualization plots...")
    print("="*70)
    
    plot_training_vs_test_metrics(val_results, test_results, model_name)
    plot_test_distribution(dice_scores, iou_scores, accuracy_scores, model_name)
    plot_box_plots(dice_scores, iou_scores, accuracy_scores, model_name)
    plot_per_sample_metrics(dice_scores, iou_scores, accuracy_scores, model_name)
    plot_correlation_heatmap(dice_scores, iou_scores, accuracy_scores, model_name)
    create_summary_table(val_results, test_results, model_name)
    
    print("="*70)
    print("✨ All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  - metrics_val_vs_test.png: Validation vs Test comparison")
    print("  - metrics_test_distribution.png: Distribution histograms")
    print("  - metrics_boxplots.png: Box plots summary")
    print("  - metrics_per_sample.png: Per-sample performance")
    print("  - metrics_correlation.png: Metrics correlation heatmap")
    print("  - metrics_summary_table.png: Summary table")


if __name__ == "__main__":
    # Example usage with your results
    val_results = {
        'dice': 0.9579,
        'jaccard_coeff': 0.9191,
        'accuracy_spheroid': 0.9895,
        'inference_time': 328.18,
        'samples': 210
    }
    
    # You would get these from running test_evaluation.py
    # For now, using example values
    test_results = {
        'num_samples': 100,
        'inference_time': 150.0,
        'avg_time_per_image': 1.5,
        'dice_mean': 0.945,
        'dice_std': 0.025,
        'dice_min': 0.880,
        'dice_max': 0.985,
        'iou_mean': 0.898,
        'iou_std': 0.032,
        'iou_min': 0.810,
        'iou_max': 0.970,
        'accuracy_mean': 0.982,
        'accuracy_std': 0.012,
        'accuracy_min': 0.950,
        'accuracy_max': 0.998,
    }
    
    # Generate dummy test scores
    np.random.seed(42)
    dice_scores = np.random.normal(0.945, 0.025, 100)
    iou_scores = np.random.normal(0.898, 0.032, 100)
    accuracy_scores = np.random.normal(0.982, 0.012, 100)
    
    generate_all_visualizations(
        val_results, test_results, 
        dice_scores, iou_scores, accuracy_scores,
        'unet_resnet34_spheroid'
    )