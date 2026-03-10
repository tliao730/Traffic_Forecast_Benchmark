"""
Generate hourly mean absolute error bar chart in PDF format.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate hourly MAE bar chart in PDF'
    )
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to hourly_mean_abs_error.csv')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for PDF')
    parser.add_argument('--title', type=str, default='Mean Absolute Error by Hour',
                        help='Plot title')
    return parser.parse_args()


def plot_hourly_errors_pdf(csv_path, output_path, title):
    """
    Create hourly error bar chart and save as PDF.
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    ax.bar(df['hour'], df['mean_abs_error'], width=0.8, 
           color='#2E86C1', alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Customize plot
    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([f'{h:02d}' for h in range(24)])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Save as PDF
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"Hourly MAE plot saved to: {output_path}")


def main():
    args = parse_args()
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate plot
    plot_hourly_errors_pdf(args.csv, args.output, args.title)


if __name__ == '__main__':
    main()
