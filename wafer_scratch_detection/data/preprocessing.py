import numpy as np
import pandas as pd


def generate_wafer_map(wafer_df, size=52):
    """
    Generate a wafer map image from the dataframe of a single wafer

    Args:
        wafer_df: DataFrame containing dies from a single wafer
        size: Size of the output image (size x size pixels)

    Returns:
        image: A 2D array representing the wafer map with 0s for empty spots,
               1s for good dies, 2s for bad dies, 3s for scratch dies
    """
    # Initialize empty wafer map
    wafer_map = np.zeros((size, size))

    # Fill in the wafer map with die information
    for _, die in wafer_df.iterrows():
        x, y = int(die['DieX']), int(die['DieY'])
        if x < size and y < size:  # Ensure we don't exceed the map size
            if die['IsGoodDie'] == 1:
                wafer_map[y, x] = 1  # Good die
            else:
                wafer_map[y, x] = 2  # Bad die
                if 'IsScratchDie' in die and die['IsScratchDie'] == 1:
                    wafer_map[y, x] = 3  # Scratch die

    return wafer_map


def prepare_dataset(train_data, test_size=0.2, random_state=42, low_yield_threshold=0.8):
    """
    Prepare dataset for training and validation

    Args:
        train_data: DataFrame containing training data
        test_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility
        low_yield_threshold: Threshold for filtering out low yield wafers

    Returns:
        X_train, y_train, X_val, y_val: Training and validation data
    """
    from sklearn.model_selection import train_test_split

    # Calculate yield for each wafer
    wafer_yield = train_data.groupby('WaferName')['IsGoodDie'].mean()

    # Filter out wafers with low yield
    low_yield_wafers = wafer_yield[wafer_yield <= low_yield_threshold].index
    filtered_train_data = train_data[~train_data['WaferName'].isin(low_yield_wafers)]

    # Generate wafer maps and labels
    wafer_maps = {}
    labels = {}

    for wafer_name, wafer_df in filtered_train_data.groupby('WaferName'):
        wafer_maps[wafer_name] = generate_wafer_map(wafer_df)
        labels[wafer_name] = 1 if wafer_df['IsScratchDie'].max() == 1 else 0

    # Split into training and validation sets
    unique_wafers = list(wafer_maps.keys())
    train_wafers, val_wafers = train_test_split(
        unique_wafers,
        test_size=test_size,
        random_state=random_state,
        stratify=[labels[w] for w in unique_wafers]
    )

    # Create datasets
    X_train = np.array([wafer_maps[w] for w in train_wafers])
    y_train = np.array([labels[w] for w in train_wafers])
    X_val = np.array([wafer_maps[w] for w in val_wafers])
    y_val = np.array([labels[w] for w in val_wafers])

    # Reshape for CNN input (add channel dimension)
    X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2], 1)
    X_val = X_val.reshape(-1, X_val.shape[1], X_val.shape[2], 1)

    # Convert to RGB-like input if needed (for pre-trained models)
    X_train_rgb = np.repeat(X_train, 3, axis=3)
    X_val_rgb = np.repeat(X_val, 3, axis=3)

    return X_train_rgb, y_train, X_val_rgb, y_val


def load_and_preprocess_test_data(test_data_path):
    """
    Load and preprocess test data

    Args:
        test_data_path: Path to test data CSV

    Returns:
        test_data: Preprocessed test data
    """
    test_data = pd.read_csv(test_data_path)

    # Additional preprocessing steps can be added here

    return test_data