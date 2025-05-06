"""
Visualization utilities for wafer scratch detection.
"""

import matplotlib.pyplot as plt
import numpy as np

# Remove seaborn dependency
# import seaborn as sns

def plot_wafer(wafer_name, wafer_data, ax=None, title=None):
    """
    Plot a wafer with good dies, bad dies, and scratch dies

    Args:
        wafer_name: Name of the wafer
        wafer_data: DataFrame containing wafer data
        ax: Matplotlib axis to plot on (creates a new one if None)
        title: Title for the plot (uses wafer_name if None)

    Returns:
        ax: The matplotlib axis that was used for plotting
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Filter to the specific wafer
    if 'WaferName' in wafer_data.columns:
        wafer_df = wafer_data[wafer_data['WaferName'] == wafer_name]
    else:
        wafer_df = wafer_data

    # Plot all dies
    good_dies = wafer_df[wafer_df['IsGoodDie'] == 1]
    bad_dies = wafer_df[wafer_df['IsGoodDie'] == 0]

    # Plot scratch dies if available
    scratch_label = 'Scratch Die'
    if 'IsScratchDie' in wafer_df.columns:
        scratch_dies = wafer_df[wafer_df['IsScratchDie'] == 1]
        scratch_label = 'Actual Scratch Die'
    elif 'IsScratchDie_Pred' in wafer_df.columns:
        scratch_dies = wafer_df[wafer_df['IsScratchDie_Pred'] == 1]
        scratch_label = 'Predicted Scratch Die'
    else:
        scratch_dies = wafer_df.head(0)  # Empty DataFrame with same structure

    # Plot
    ax.scatter(good_dies['DieX'], good_dies['DieY'], c='green', alpha=0.5, label='Good Die')
    ax.scatter(bad_dies['DieX'], bad_dies['DieY'], c='red', alpha=0.5, label='Bad Die')
    ax.scatter(scratch_dies['DieX'], scratch_dies['DieY'], edgecolors='black',
               facecolors='none', s=100, label=scratch_label)

    # Set title and labels
    if title is None:
        title = f'Wafer: {wafer_name}'
    ax.set_title(title)
    ax.set_xlabel('DieX')
    ax.set_ylabel('DieY')
    ax.legend()

    return ax

def visualize_scratch_detection(wafer_df, prediction_df, wafer_name=None):
    """
    Visualize the original wafer and the scratch detection results

    Args:
        wafer_df: Original wafer DataFrame
        prediction_df: DataFrame with scratch predictions
        wafer_name: Name of the wafer for the title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot original wafer
    plot_wafer(wafer_name, wafer_df, ax=ax1, title='Original Wafer')

    # Plot predictions
    if 'IsScratchDie_Pred' not in prediction_df.columns:
        raise ValueError("Prediction DataFrame must contain an 'IsScratchDie_Pred' column")

    title = 'Predicted Scratches'
    if wafer_name:
        title += f' for Wafer {wafer_name}'

    plot_wafer(wafer_name, prediction_df, ax=ax2, title=title)

    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """
    Plot training history

    Args:
        history: History object from model.fit()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='lower right')

    # Plot loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.show()

    # Plot additional metrics if available
    metrics = ['precision', 'recall', 'f1_score']
    available_metrics = [m for m in metrics if m in history.history]

    if available_metrics:
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(16, 5))
        if len(available_metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(available_metrics):
            axes[i].plot(history.history[metric])
            axes[i].plot(history.history[f'val_{metric}'])
            axes[i].set_title(f'Model {metric.capitalize()}')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_xlabel('Epoch')
            axes[i].legend(['Train', 'Validation'], loc='lower right')

        plt.tight_layout()
        plt.show()

def plot_wafer_yield_distribution(train_data, low_yield_threshold=None):
    """
    Plot the distribution of wafer yields

    Args:
        train_data: DataFrame containing training data
        low_yield_threshold: Threshold for low yield wafers (draws a vertical line)
    """
    # Calculate yield for each wafer
    wafer_yield = train_data.groupby('WaferName')['IsGoodDie'].mean()

    plt.figure(figsize=(12, 6))
    plt.hist(wafer_yield, bins=30, alpha=0.7)

    if low_yield_threshold is not None:
        plt.axvline(low_yield_threshold, color='r', linestyle='--',
                   label=f'Low yield threshold: {low_yield_threshold:.2f}')

    plt.title('Distribution of Wafer Yield')
    plt.xlabel('Yield (% of Good Dies)')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

    return wafer_yield