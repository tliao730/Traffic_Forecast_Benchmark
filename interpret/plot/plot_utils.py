import os
import numpy as np
import matplotlib.pyplot as plt
import shap


def _infer_context_length(shap_values, context_length):
    if context_length is not None:
        return context_length
    return int(np.abs(shap_values).mean(axis=0).shape[0])


def plot_shap_summary(shap_values, X_sample, explainer, output_dir='./shap_plots', model_name='ridge'):
    os.makedirs(output_dir, exist_ok=True)

    context_length = X_sample.shape[1]
    feature_names = [f't-{context_length - i}' for i in range(context_length)]

    print(f"\nGenerating SHAP plots...")

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      plot_type='bar', show=False)
    plt.title(f'{model_name.upper()} - Feature Importance (Mean |SHAP|)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_shap_summary_bar.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_name}_shap_summary_bar.pdf")

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title(f'{model_name.upper()} - SHAP Summary Plot')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_shap_summary_beeswarm.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_name}_shap_summary_beeswarm.pdf")

    plt.figure(figsize=(14, 8))
    shap.plots.heatmap(shap.Explanation(values=shap_values,
                                        data=X_sample,
                                        feature_names=feature_names),
                       show=False)
    plt.title(f'{model_name.upper()} - SHAP Heatmap')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_shap_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_name}_shap_heatmap.pdf")

    plt.figure(figsize=(20, 3))
    shap.force_plot(explainer.expected_value, shap_values[0], X_sample[0],
                    feature_names=feature_names, matplotlib=True, show=False)
    plt.title(f'{model_name.upper()} - SHAP Force Plot (First Sample)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_shap_force_first.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_name}_shap_force_first.pdf")


def plot_temporal_importance(shap_values, context_length=None,
                             output_dir='./shap_plots', model_name='model'):
    os.makedirs(output_dir, exist_ok=True)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    context_length = _infer_context_length(shap_values, context_length)

    plt.figure(figsize=(16, 6))
    time_points = np.arange(context_length)
    plt.plot(time_points, mean_abs_shap[:context_length], linewidth=1.5, color='steelblue', alpha=0.7)
    plt.fill_between(time_points, 0, mean_abs_shap[:context_length], alpha=0.3, color='steelblue')

    plt.xlabel('Observed index (0 oldest → t-1 most recent)', fontsize=12)
    plt.ylabel('Mean |SHAP value|', fontsize=12)
    plt.title(f'{model_name.upper()} - Temporal Feature Importance', fontsize=14)
    plt.grid(axis='y', alpha=0.3)

    top_k = min(10, context_length)
    top_indices = np.argsort(mean_abs_shap[:context_length])[-top_k:]
    plt.scatter(top_indices, mean_abs_shap[top_indices],
                color='red', s=100, zorder=5, label=f'Top {top_k} points')

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_temporal_importance_full.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_name}_temporal_importance_full.pdf")

    zoom_length = min(200, context_length)
    plt.figure(figsize=(14, 6))
    plt.bar(range(zoom_length), mean_abs_shap[:zoom_length],
            color='steelblue', alpha=0.7)
    plt.xlabel(f'Most Recent {zoom_length} steps (right edge = t-1)', fontsize=12)
    plt.ylabel('Mean |SHAP value|', fontsize=12)
    plt.title(f'{model_name.upper()} - Recent History Importance (Zoomed)', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_temporal_importance_recent.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_name}_temporal_importance_recent.pdf")

    print(f"\nTop {top_k} most important time points (0 oldest → t-1 most recent):")
    top_indices = np.argsort(mean_abs_shap[:context_length])[-top_k:][::-1]
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. Position {idx} (t-{context_length-idx}): {mean_abs_shap[idx]:.6f}")


def plot_time_windows(shap_values, context_length=None,
                      output_dir='./shap_plots', model_name='model'):
    os.makedirs(output_dir, exist_ok=True)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    context_length = _infer_context_length(shap_values, context_length)

    n_lags = context_length
    distant = mean_abs_shap[:n_lags//3].sum()
    medium = mean_abs_shap[n_lags//3:2*n_lags//3].sum()
    recent = mean_abs_shap[2*n_lags//3:n_lags].sum()
    #import pdb; pdb.set_trace()
    plt.figure(figsize=(12, 7))
    third = n_lags // 3
    windows = [
        f'Recent\n(lag 1-{third})',
        f'Medium\n(lag {third+1}-{third*2})',
        f'Distant\n(lag {third*2+1}-{n_lags})'
    ]
    values = [recent, medium, distant]
    colors = ['#ff6b6b', '#4ecdc4', '#95e1d3']

    bars = plt.bar(windows, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    plt.title(f'{model_name.upper()} - Importance by Time Window', fontsize=15, fontweight='bold')
    plt.gca().set_yticks([])
    plt.gca().spines['left'].set_visible(False)

    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'{100*value/sum(values):.1f}%',
                 ha='center', va='center', fontsize=14, fontweight='bold', color='black')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_time_windows.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_name}_time_windows.pdf")

    total = sum(values)
    print(f"\nImportance by time window:")
    print(f"  Recent (lag 1-{third}): {recent:.4f} ({100*recent/total:.1f}%)")
    print(f"  Medium (lag {third+1}-{third*2}): {medium:.4f} ({100*medium/total:.1f}%)")
    print(f"  Distant (lag {third*2+1}-{n_lags}): {distant:.4f} ({100*distant/total:.1f}%)")


def plot_shap_heatmap(shap_values, X_sample,
                      output_dir='./shap_plots', model_name='model'):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(16, 8))

    if X_sample.shape[1] > 200:
        step = X_sample.shape[1] // 200
        shap_plot = shap_values[:, ::step]
    else:
        shap_plot = shap_values

    plt.imshow(shap_plot, aspect='auto', cmap='RdBu_r',
               interpolation='nearest')
    plt.colorbar(label='SHAP value')
    plt.xlabel('Observed index (0 oldest → t-1 most recent)', fontsize=12)
    plt.ylabel('Sample Index', fontsize=12)
    plt.title(f'{model_name.upper()} - SHAP Values Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_shap_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_name}_shap_heatmap.pdf")


def plot_sample_with_shap_and_prediction(X_sample, y_sample, shap_values, predict_future_fn=None,
                                         num_examples=3,
                                         output_dir='./shap_plots',
                                         model_name='model',
                                         predicted_future_all=None):
    os.makedirs(output_dir, exist_ok=True)

    num_examples = min(num_examples, len(X_sample))

    for idx in range(num_examples):
        fig, axes = plt.subplots(3, 1, figsize=(16, 10))

        context_length = X_sample.shape[1]
        is_multi_step = y_sample.ndim == 2
        if is_multi_step:
            prediction_length = y_sample.shape[1]
            future_time = np.arange(context_length, context_length + prediction_length)
        else:
            future_time = [context_length]

        historical_time = np.arange(context_length)

        historical_data = X_sample[idx]
        actual_future = y_sample[idx]
        shap_vals = shap_values[idx]

        if predicted_future_all is not None:
            predicted_future = predicted_future_all[idx]
        else:
            predicted_future = predict_future_fn(historical_data)
        # import pdb; pdb.set_trace()
        if is_multi_step:
            predicted_future = np.asarray(predicted_future).reshape(-1)[:prediction_length]

        ax1 = axes[0]
        scatter = ax1.scatter(historical_time, historical_data,
                              c=shap_vals, cmap='RdYlGn',
                              s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.plot(historical_time, historical_data, 'b-', alpha=0.3, linewidth=1)
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label('SHAP Value', fontsize=10)
        ax1.axvline(x=context_length, color='red', linestyle='--', linewidth=2, label='Forecast start')
        ax1.set_xlabel('Observed index (0 oldest → t-1 most recent)', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.set_title(f'Sample {idx+1}: Historical Data (colored by SHAP importance)',
                      fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2 = axes[1]
        colors = ['red' if x > 0 else 'blue' for x in shap_vals]
        ax2.bar(historical_time, shap_vals, color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.axvline(x=context_length, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Observed index (0 oldest → t-1 most recent)', fontsize=12)
        ax2.set_ylabel('SHAP Value', fontsize=12)
        ax2.set_title('SHAP Values: Contribution of Each Historical Point\n'
                      'Red = Increases prediction, Blue = Decreases prediction',
                      fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3)

        ax3 = axes[2]
        ax3.plot(historical_time, historical_data, 'b-', linewidth=2,
                 label='Historical Data', alpha=0.7)
        if is_multi_step:
            ax3.plot(future_time, actual_future, 'go-', markersize=6,
                     label='Actual Future', alpha=0.8)
            ax3.plot(future_time, predicted_future, 'rs--', markersize=6,
                     label='Predicted Future', alpha=0.8)
        else:
            ax3.plot(future_time, [actual_future], 'go', markersize=10,
                     label='Actual Next Value', alpha=0.8)
            ax3.plot(future_time, [predicted_future], 'rs', markersize=10,
                     label='Predicted Next Value', alpha=0.8)

        if not is_multi_step:
            ax3.plot(
                [historical_time[-1], context_length],
                [historical_data[-1], actual_future],
                'g--',
                alpha=0.4,
                linewidth=1,
            )
            ax3.plot(
                [historical_time[-1], context_length],
                [historical_data[-1], predicted_future],
                'r--',
                alpha=0.4,
                linewidth=1,
            )

        if is_multi_step:
            ax3.axvspan(
                context_length,
                context_length + prediction_length - 1,
                alpha=0.1,
                color='yellow',
                label='Forecast Horizon',
            )
        else:
            ax3.axvspan(
                context_length - 0.5,
                context_length + 0.5,
                alpha=0.1,
                color='yellow',
                label='Forecast Point',
            )
        ax3.axvline(x=context_length, color='red', linestyle='--', linewidth=2)

        ax3.set_xlabel('Time Steps (prediction starts at right edge)', fontsize=12)
        ax3.set_ylabel('Value', fontsize=12)
        ax3.set_title('Full Time Series: History + Actual vs Predicted Next Value',
                      fontsize=13, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(alpha=0.3)

        if is_multi_step:
            mae = np.mean(np.abs(actual_future - predicted_future))
            mse = np.mean((actual_future - predicted_future) ** 2)
            ax3.text(0.02, 0.98,
                     f'MAE: {mae:.2f}\n'
                     f'MSE: {mse:.2f}',
                     transform=ax3.transAxes, fontsize=11,
                     verticalalignment='top', bbox=dict(boxstyle='round',
                     facecolor='wheat', alpha=0.5))
        else:
            error = abs(actual_future - predicted_future)
            ax3.text(0.02, 0.98,
                     f'Actual: {actual_future:.2f}\n'
                     f'Predicted: {predicted_future:.2f}\n'
                     f'Error: {error:.2f}',
                     transform=ax3.transAxes, fontsize=11,
                     verticalalignment='top', bbox=dict(boxstyle='round',
                     facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{model_name}_sample_{idx+1}_detailed.pdf',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {model_name}_sample_{idx+1}_detailed.pdf")

    print(f"\nGenerated {num_examples} detailed sample visualizations!")
