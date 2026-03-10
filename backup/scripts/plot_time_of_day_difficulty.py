"""
Plot time-of-day prediction difficulty from time_mean_abs_error.csv
Groups timestamps into 48 30-minute slots (00:00-23:30) and visualizes mean error patterns.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_and_process_data(csv_path):
    """
    Load time_mean_abs_error.csv and compute time-of-day (48 slots) aggregated errors.
    
    Args:
        csv_path: Path to CSV with columns [timestamp, mean_abs_error]
        
    Returns:
        DataFrame with columns [slot, hour, minute, mean_abs_error]
        where slot is 0-47 (48 30-min intervals per day)
    """
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    
    # Group by time-of-day (hour, minute) and compute mean error
    tod_errors = df.groupby(['hour', 'minute'])['mean_abs_error'].mean().reset_index()
    
    # Create slot index: 0-47 for 48 30-minute intervals
    # Slot = hour * 2 + minute // 30
    tod_errors['slot'] = tod_errors['hour'] * 2 + tod_errors['minute'] // 30
    
    # Aggregate by slot to get one value per slot (average across all minutes in that slot)
    tod_errors = tod_errors.groupby('slot', as_index=False).agg({
        'hour': 'first',
        'minute': lambda x: (tod_errors.loc[x.index, 'slot'].iloc[0] % 2) * 30,  # 0 or 30
        'mean_abs_error': 'mean'
    })
    
    # Sort by slot for proper plotting
    tod_errors = tod_errors.sort_values('slot').reset_index(drop=True)
    
    return tod_errors


def plot_time_of_day_difficulty(tod_errors, output_path, title=None):
    """
    Create line plot showing prediction difficulty across 48 30-minute time slots.
    
    Args:
        tod_errors: DataFrame with [slot, hour, minute, mean_abs_error]
        output_path: Path to save the plot
        title: Optional custom title
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    
    ax.bar(tod_errors['slot'], tod_errors['mean_abs_error'], 
           width=0.8, color='#2E86C1', alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Customize x-axis to show hour labels
    # Show labels every 2 slots (every hour)
    hour_ticks = np.arange(0, 48, 2)
    hour_labels = [f'{h:02d}:00' for h in range(24)]
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels(hour_labels, rotation=45, ha='right')
    
    # Add minor ticks for 30-min intervals
    ax.set_xticks(np.arange(0, 48), minor=True)
    ax.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=1)
    ax.grid(True, which='minor', alpha=0.1, linestyle=':', linewidth=0.5)
    
    ax.set_xlabel('Time of Day (30-minute intervals)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
    
    if title is None:
        title = 'Prediction Difficulty by Time of Day (48 x 30-min slots)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot time-of-day prediction difficulty from time_mean_abs_error.csv'
    )
    parser.add_argument('--csv', type=str, 
                        default='outputs/astgcn/time_mean_abs_error.csv',
                        help='Path to time_mean_abs_error.csv')
    parser.add_argument('--output', type=str, 
                        default='outputs/astgcn/time_of_day_difficulty.pdf',
                        help='Output path for the plot (PDF)')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom plot title')
    parser.add_argument('--save-csv', action='store_true',
                        help='Also save processed time-of-day data to CSV')
    
    args = parser.parse_args()
    
    # Load and process data
    print(f"Loading data from: {args.csv}")
    tod_errors = load_and_process_data(args.csv)
    print(f"Processed {len(tod_errors)} time-of-day slots (expected 48)")
    print(f"\nError statistics:")
    print(f"  Mean: {tod_errors['mean_abs_error'].mean():.2f}")
    print(f"  Std:  {tod_errors['mean_abs_error'].std():.2f}")
    print(f"  Min:  {tod_errors['mean_abs_error'].min():.2f}")
    print(f"  Max:  {tod_errors['mean_abs_error'].max():.2f}")
    
    # Find hardest and easiest time slots
    hardest = tod_errors.nlargest(5, 'mean_abs_error')[['slot', 'hour', 'minute', 'mean_abs_error']]
    easiest = tod_errors.nsmallest(5, 'mean_abs_error')[['slot', 'hour', 'minute', 'mean_abs_error']]
    print(f"\nTop 5 hardest time slots:")
    for _, row in hardest.iterrows():
        print(f"  {int(row['hour']):02d}:{int(row['minute']):02d} (slot {int(row['slot'])}): {row['mean_abs_error']:.2f}")
    print(f"\nTop 5 easiest time slots:")
    for _, row in easiest.iterrows():
        print(f"  {int(row['hour']):02d}:{int(row['minute']):02d} (slot {int(row['slot'])}): {row['mean_abs_error']:.2f}")
    
    # Create plot
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_time_of_day_difficulty(tod_errors, args.output, args.title)
    
    # Optionally save processed CSV
    if args.save_csv:
        csv_output = output_path.with_suffix('.csv')
        tod_errors.to_csv(csv_output, index=False)
        print(f"Processed data saved to: {csv_output}")


if __name__ == '__main__':
    main()
