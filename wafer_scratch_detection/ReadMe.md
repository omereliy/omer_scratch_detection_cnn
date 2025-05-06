"""
# Wafer Scratch Detection

A Python package for detecting scratches on semiconductor wafers using deep learning.

```bash
# Install the package when in parent directory
pip install wafer_scratch_detection

# For hyperparameter tuning functionality
pip install -e .[hyperparameter_tuning]
```

## Usage

### Basic usage

```python
from wafer_scratch_detection.data.preprocessing import prepare_dataset
from wafer_scratch_detection.models.vit_adapter import load_and_adapt_vit_model
from wafer_scratch_detection.models.batch_norm import train_with_batch_norm
from wafer_scratch_detection.scratch_detection.detection import detect_scratch_dies_advanced
import pandas as pd
# Load and preprocess data
train_data = pd.read_csv('scratch_train.csv')
X_train, y_train, X_val, y_val = prepare_dataset(train_data)

# Load and adapt model
model = load_and_adapt_vit_model()

# Train the model
history = train_with_batch_norm(model, X_train, y_train, X_val, y_val)

# Detect scratches on a wafer
test_data = pd.read_csv('scratch_test.csv')
wafer_df = test_data[test_data['WaferName'] == 'Wafer1']
prediction_df = detect_scratch_dies_advanced(wafer_df, model)
```

### Running the full pipeline

```bash
python main.py
```