"""
Vision Transformer (ViT) model adaptation for wafer scratch detection.
"""

import tensorflow as tf
try:
    # For TensorFlow 2.x where keras is part of tensorflow
    keras = tf.keras
except:
    # Fall back to standalone keras if needed
    import keras


def load_and_adapt_vit_model(input_shape=(52, 52, 1), pretrained_model_path='model_vit_v1b'):
    """
    Load and adapt the pre-trained ViT model for wafer scratch detection

    Args:
        input_shape: Shape of the input wafer maps
        pretrained_model_path: Path to the pre-trained ViT model

    Returns:
        model: Adapted model ready for fine-tuning
    """
    try:
        # Load the pre-trained model
        base_model = keras.models.load_model(pretrained_model_path)
        print("Pre-trained ViT model loaded successfully.")

        # Get the expected input shape from the pre-trained model
        expected_shape = base_model.input_shape[1:]
        print(f"Expected input shape: {expected_shape}")
        print(f"Actual input shape: {input_shape}")

        # Create an adaptation pipeline
        inputs = keras.layers.Input(shape=input_shape)

        # Convert single channel to three channels if needed
        if input_shape[-1] == 1 and expected_shape[-1] == 3:
            x = keras.layers.Conv2D(3, kernel_size=1, padding='same')(inputs)
        else:
            x = inputs

        # Resize if dimensions don't match
        if input_shape[0] != expected_shape[0] or input_shape[1] != expected_shape[1]:
            x = keras.layers.Resizing(expected_shape[0], expected_shape[1])(x)

        # Add normalization to match the expected input distribution
        x = keras.layers.Normalization(
            mean=[0.485, 0.456, 0.406],
            variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2]
        )(x)

        # Get the base model without its input layer
        for layer in base_model.layers[1:]:
            x = layer(x)

        # Add new classification head
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)

        # Create the new model
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Freeze the base model layers
        for layer in model.layers[1:-6]:  # Skip input layer and last 6 layers
            layer.trainable = False

        return model

    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        print("Creating a simple CNN model instead.")

        # Create a simple CNN model as fallback
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(32, 3, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, 3, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with appropriate optimizer and metrics

    Args:
        model: The model to compile
        learning_rate: Initial learning rate

    Returns:
        model: Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    return model


def plot_model_structure(model, filename='model_structure.png'):
    """
    Visualize the model structure

    Args:
        model: The model to visualize
        filename: Output filename for the model diagram
    """
    keras.utils.plot_model(
        model,
        to_file=filename,
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB'
    )
    print(f"Model structure saved to {filename}")
