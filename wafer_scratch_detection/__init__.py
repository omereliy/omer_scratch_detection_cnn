"""
Wafer Scratch Detection package for semiconductor wafer analysis.
"""

from .models.vit_adapter import load_and_adapt_vit_model
from .models.batch_norm import train_with_batch_norm, create_fine_tuning_pipeline
from .scratch_detection.detection import detect_scratch_dies_advanced
from .data.preprocessing import generate_wafer_map, prepare_dataset
from .data.visualization import visualize_scratch_detection, plot_wafer