"""
Time-of-Day Difficulty Profiling
Generate comprehensive analysis of prediction difficulty across time-of-day slots.
Includes 4 plots:
1. MAE by time-of-day (prediction difficulty)
2. Mean flow by time-of-day (scale effect)
3. Mean |Δflow| by time-of-day (non-stationarity/turning point effect)
4. Missing/outlier rate by time-of-day (data quality effect)
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate time-of-day difficulty profiling with 4 plots'
    )
    parser.add_argument('--pred-csv', type=str, required=True,
                        help='Path to prediction CSV (with truth and pred values)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for the PDF plot')
    parser.add_argument('--save-csv', action='store_true',
                        help='Save processed data to CSV')
    parser.add_argument('--outlier-threshold', type=float, default=3.0,
                        help='Z-score threshold for outlier detection (default: 3.0)')
    
    return parser.parse_args()


def load_and_process_data(csv_path, outlier_threshold=3.0):
    """
    Load prediction CSV and compute time-of-day statistics.
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    # Separate truth and predictions
    truth_df = df[df['stat'] == 'truth'].copy()
    pred_df = df[df['stat'] == 'mean'].copy()
    
    # Merge to compute errors
    merged = truth_df.merge(
        pred_df,
        on=['ds_config', 'sample_idx', 'item_id', 'timestamp', 'dim'],
        suffixes=('_truth', '_pred')
    )
    merged['abs_error'] = (merged['value_pred'] - merged['value_truth']).abs()
    
    # Extract time features
    merged['hour'] = merged['timestamp'].dt.hour
    merged['minute'] = merged['timestamp'].dt.minute
    
    # Detect missing values
    merged['is_missing'] = merged['value_truth'].isna()
    
    # Detect outliers using z-score on truth values
    # Group by item_id to compute item-specific z-scores
    def detect_outliers(group):
        mean_val = group['value_truth'].mean()
        std_val = group['value_truth'].std()
        if std_val > 0:
            group['z_score'] = (group['value_truth'] - mean_val) / std_val
            group['is_outlier'] = group['z_score'].abs() > outlier_threshold
        else:
            group['is_outlier'] = False
        return group
    
    merged = merged.groupby('item_id', group_keys=False).apply(detect_outliers)
    
    # Sort by timestamp for flow change computation
    merged = merged.sort_values(['item_id', 'timestamp'])
    
    # Compute flow change (|Δflow|) within each item
    merged['flow_change'] = merged.groupby('item_id')['value_truth'].diff().abs()
    
    return merged


def compute_tod_statistics(merged_df):
    """
    Compute time-of-day statistics for all 4 metrics.
    Returns DataFrame with slot (0-47 for 30-min intervals) and metrics.
    """
    # Group by hour and minute first
    tod_stats = merged_df.groupby(['hour', 'minute']).agg({
        'abs_error': 'mean',           # MAE
        'value_truth': 'mean',         # Mean flow
        'flow_change': 'mean',         # Mean |Δflow|
        'is_missing': 'mean',          # Missing rate
        'is_outlier': 'mean'           # Outlier rate
    }).reset_index()
    
    # Create slot index: 0-47 for 48 30-minute intervals
    tod_stats['slot'] = tod_stats['hour'] * 2 + tod_stats['minute'] // 30
    
    # Aggregate by slot to get one value per slot
    tod_final = tod_stats.groupby('slot').agg({
        'hour': 'first',
        'abs_error': 'mean',
        'value_truth': 'mean',
        'flow_change': 'mean',
        'is_missing': 'mean',
        'is_outlier': 'mean'
    }).reset_index()
    
    # Add minute column (0 or 30)
    tod_final['minute'] = (tod_final['slot'] % 2) * 30
    
    # Rename columns for clarity
    tod_final.rename(columns={
        'abs_error': 'mean_mae',
        'value_truth': 'mean_flow',
        'flow_change': 'mean_flow_change',
        'is_missing': 'missing_rate',
        'is_outlier': 'outlier_rate'
    }, inplace=True)
    
    # Compute combined data quality issue rate
    tod_final['data_quality_issue_rate'] = tod_final['missing_rate'] + tod_final['outlier_rate']
    
    # Sort by slot
    tod_final = tod_final.sort_values('slot').reset_index(drop=True)
    
    return tod_final


def plot_tod_profiling(tod_df, output_path):
    """
    Create 4-panel plot for time-of-day difficulty profiling.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Common x-axis settings
    hour_ticks = np.arange(0, 48, 2)
    hour_labels = [f'{h:02d}:00' for h in range(24)]
    
    # Plot 1: MAE by time-of-day
    ax1 = axes[0, 0]
    ax1.bar(tod_df['slot'], tod_df['mean_mae'], width=0.8, 
            color='#2E86C1', alpha=0.8, edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Time of Day (30-min intervals)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Prediction Difficulty by Time-of-Day', fontsize=12, fontweight='bold')
    ax1.set_xticks(hour_ticks)
    ax1.set_xticklabels(hour_labels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Mean flow by time-of-day
    ax2 = axes[0, 1]
    ax2.bar(tod_df['slot'], tod_df['mean_flow'], width=0.8,
            color='#28B463', alpha=0.8, edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Time of Day (30-min intervals)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Mean Flow', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Mean Flow by Time-of-Day\n(Scale Effect)', fontsize=12, fontweight='bold')
    ax2.set_xticks(hour_ticks)
    ax2.set_xticklabels(hour_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Mean |Δflow| by time-of-day
    ax3 = axes[1, 0]
    ax3.bar(tod_df['slot'], tod_df['mean_flow_change'], width=0.8,
            color='#E67E22', alpha=0.8, edgecolor='white', linewidth=0.5)
    ax3.set_xlabel('Time of Day (30-min intervals)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Mean |Δflow|', fontsize=11, fontweight='bold')
    ax3.set_title('(c) Flow Variability by Time-of-Day\n(Non-stationarity Effect)', fontsize=12, fontweight='bold')
    ax3.set_xticks(hour_ticks)
    ax3.set_xticklabels(hour_labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Data quality issues by time-of-day
    ax4 = axes[1, 1]
    # Stack missing and outlier rates
    ax4.bar(tod_df['slot'], tod_df['missing_rate'], width=0.8,
            label='Missing Rate', color='#E74C3C', alpha=0.8, edgecolor='white', linewidth=0.5)
    ax4.bar(tod_df['slot'], tod_df['outlier_rate'], width=0.8,
            bottom=tod_df['missing_rate'], label='Outlier Rate',
            color='#C0392B', alpha=0.8, edgecolor='white', linewidth=0.5)
    ax4.set_xlabel('Time of Day (30-min intervals)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Rate', fontsize=11, fontweight='bold')
    ax4.set_title('(d) Data Quality Issues by Time-of-Day', fontsize=12, fontweight='bold')
    ax4.set_xticks(hour_ticks)
    ax4.set_xticklabels(hour_labels, rotation=45, ha='right')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Plot saved to: {output_path}")
    plt.close()


def print_statistics(tod_df):
    """
    Print summary statistics for all metrics.
    """
    print("\n" + "="*60)
    print("Time-of-Day Difficulty Profiling Statistics")
    print("="*60)
    
    print(f"\nProcessed {len(tod_df)} time-of-day slots (expected 48)")
    
    print("\n1. Prediction Difficulty (MAE):")
    print(f"   Mean: {tod_df['mean_mae'].mean():.2f}")
    print(f"   Std:  {tod_df['mean_mae'].std():.2f}")
    print(f"   Min:  {tod_df['mean_mae'].min():.2f}")
    print(f"   Max:  {tod_df['mean_mae'].max():.2f}")
    
    print("\n2. Mean Flow (Scale):")
    print(f"   Mean: {tod_df['mean_flow'].mean():.2f}")
    print(f"   Std:  {tod_df['mean_flow'].std():.2f}")
    print(f"   Min:  {tod_df['mean_flow'].min():.2f}")
    print(f"   Max:  {tod_df['mean_flow'].max():.2f}")
    
    print("\n3. Flow Variability (|Δflow|):")
    print(f"   Mean: {tod_df['mean_flow_change'].mean():.2f}")
    print(f"   Std:  {tod_df['mean_flow_change'].std():.2f}")
    print(f"   Min:  {tod_df['mean_flow_change'].min():.2f}")
    print(f"   Max:  {tod_df['mean_flow_change'].max():.2f}")
    
    print("\n4. Data Quality Issues:")
    print(f"   Mean missing rate: {tod_df['missing_rate'].mean()*100:.3f}%")
    print(f"   Mean outlier rate: {tod_df['outlier_rate'].mean()*100:.3f}%")
    print(f"   Total issue rate:  {tod_df['data_quality_issue_rate'].mean()*100:.3f}%")
    
    # Identify hardest and easiest time slots
    print("\nTop 5 hardest time slots (highest MAE):")
    hardest = tod_df.nlargest(5, 'mean_mae')[['slot', 'hour', 'minute', 'mean_mae']]
    for _, row in hardest.iterrows():
        print(f"  {int(row['hour']):02d}:{int(row['minute']):02d} (slot {int(row['slot'])}): MAE={row['mean_mae']:.2f}")
    
    print("\nTop 5 easiest time slots (lowest MAE):")
    easiest = tod_df.nsmallest(5, 'mean_mae')[['slot', 'hour', 'minute', 'mean_mae']]
    for _, row in easiest.iterrows():
        print(f"  {int(row['hour']):02d}:{int(row['minute']):02d} (slot {int(row['slot'])}): MAE={row['mean_mae']:.2f}")
    
    print("\n" + "="*60)


def main():
    args = parse_args()
    
    # Load and process data
    merged_df = load_and_process_data(args.pred_csv, args.outlier_threshold)
    
    # Compute time-of-day statistics
    tod_df = compute_tod_statistics(merged_df)
    
    # Print statistics
    print_statistics(tod_df)
    
    # Create plot
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_tod_profiling(tod_df, args.output)
    
    # Save CSV if requested
    if args.save_csv:
        csv_output = output_path.with_suffix('.csv')
        tod_df.to_csv(csv_output, index=False)
        print(f"Processed data saved to: {csv_output}")


if __name__ == '__main__':
    main()
