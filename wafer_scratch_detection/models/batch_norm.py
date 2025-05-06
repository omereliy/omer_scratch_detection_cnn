"""
Enhanced training pipeline with batch normalization for wafer scratch detection.
"""
from typing import Any

import keras
from keras.api.layers import Dense, Dropout, BatchNormalization
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard


def create_fine_tuning_pipeline(base_model, num_classes=1, dropout_rate=0.5):
    """
    Create an improved fine-tuning pipeline with batch normalization

    Args:
        base_model: Pre-trained model to use as feature extractor
        num_classes: Number of output classes (1 for binary)
        dropout_rate: Dropout rate for regularization

    Returns:
        model: Fine-tuning model with batch normalization
    """
    # Create the base model with frozen layers
    base_model.trainable = False

    # Get the input and output layers of the base model
    inputs = base_model.input
    x = base_model.output

    # Add new classification layers with batch normalization
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.8)(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.5)(x)

    # Final layer
    if num_classes == 1:
        outputs = Dense(1, activation='sigmoid')(x)
    else:
        outputs = Dense(num_classes, activation='softmax')(x)

    # Create the fine-tuning model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


class GradualUnfreezing(keras.callbacks.Callback):
    """
    Custom callback for gradually unfreezing layers during training
    """
    model: Any
    def __init__(self, model, base_model_name='base_model', total_epochs=30, unfreeze_strategy='linear'):
        """
        Initialize the gradual unfreezing callback

        Args:
            model: The model being trained
            base_model_name: Name of the base model layer
            total_epochs: Total number of training epochs
            unfreeze_strategy: Strategy for unfreezing ('linear', 'exponential', or 'step')
        """
        super(GradualUnfreezing, self).__init__()
        self.model = model
        self.base_model_name = base_model_name
        self.total_epochs = total_epochs
        self.unfreeze_strategy = unfreeze_strategy
        self.base_trainable_set = False

    def on_epoch_begin(self, epoch, logs=None):
        # Find the base model
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'name') and self.base_model_name in layer.name:
                base_model = layer
                base_model_index = i
                break
        else:
            # If we can't find the base model by name, assume it's the first layer that has layers
            for i, layer in enumerate(self.model.layers):
                if hasattr(layer, 'layers'):
                    base_model = layer
                    base_model_index = i
                    break
            else:
                print("Could not find base model. Skipping gradual unfreezing.")
                return

        # Make the base model trainable if not already
        if not self.base_trainable_set:
            base_model.trainable = True
            self.base_trainable_set = True

        # Get all the layers in the base model
        base_layers = base_model.layers
        num_layers = len(base_layers)

        # Skip the first few layers (like input, normalization) which should remain frozen
        start_layer = 3  # Skip early layers

        # Calculate how many layers to unfreeze based on the strategy
        if self.unfreeze_strategy == 'linear':
            # Linear strategy: unfreeze more layers as training progresses
            layers_to_unfreeze = start_layer + int((epoch / self.total_epochs) * (num_layers - start_layer))
        elif self.unfreeze_strategy == 'exponential':
            # Exponential strategy: unfreeze more layers later in training
            layers_to_unfreeze = start_layer + int(((epoch / self.total_epochs) ** 2) * (num_layers - start_layer))
        elif self.unfreeze_strategy == 'step':
            # Step strategy: unfreeze in steps
            if epoch < self.total_epochs // 4:
                layers_to_unfreeze = start_layer
            elif epoch < self.total_epochs // 2:
                layers_to_unfreeze = start_layer + (num_layers - start_layer) // 3
            elif epoch < 3 * self.total_epochs // 4:
                layers_to_unfreeze = start_layer + 2 * (num_layers - start_layer) // 3
            else:
                layers_to_unfreeze = num_layers
        else:
            # Default to linear
            layers_to_unfreeze = start_layer + int((epoch / self.total_epochs) * (num_layers - start_layer))

        # Limit to the actual number of layers
        layers_to_unfreeze = min(layers_to_unfreeze, num_layers)

        # Set layers as trainable or non-trainable
        for i, layer in enumerate(base_layers):
            if i < layers_to_unfreeze:
                layer.trainable = False  # Keep early layers frozen
            else:
                layer.trainable = True  # Unfreeze later layers

        # Print unfreezing status every few epochs
        if epoch % 5 == 0 or epoch == self.total_epochs - 1:
            trainable_count = sum(1 for layer in base_layers if layer.trainable)
            print(f"\nEpoch {epoch}: Unfrozen {trainable_count} of {num_layers} layers in base model")


def get_lr_schedule(initial_lr=0.001, decay_rate=0.9, decay_steps=2):
    """
    Create a learning rate schedule

    Args:
        initial_lr: Initial learning rate
        decay_rate: Rate of decay
        decay_steps: Number of steps between decays

    Returns:
        schedule: Learning rate schedule function
    """

    def lr_schedule(epoch, lr):
        if epoch % decay_steps == 0 and epoch > 0:
            return lr * decay_rate
        return lr

    return lr_schedule


def create_callbacks(checkpoint_path, patience=10, min_delta=0.001, early_stop=True):
    """
    Create a set of training callbacks

    Args:
        checkpoint_path: Path to save model checkpoints
        patience: Number of epochs with no improvement before early stopping
        min_delta: Minimum change to qualify as improvement
        early_stop: Whether to use early stopping

    Returns:
        callbacks: List of callbacks
    """
    callbacks = []

    # Model checkpoint
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    callbacks.append(model_checkpoint)

    # Learning rate scheduler
    lr_scheduler = keras.callbacks.LearningRateScheduler(
        get_lr_schedule(), verbose=1
    )
    callbacks.append(lr_scheduler)

    # Reduce LR on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=patience // 2,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(reduce_lr)

    # Early stopping
    if early_stop:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

    # TensorBoard logging
    try:
        tensorboard = TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)
    except:
        print("TensorBoard callback could not be created. Continuing without it.")

    return callbacks


def train_with_batch_norm(model, X_train, y_train, X_val, y_val,
                          batch_size=16, epochs=30,
                          checkpoint_path='./checkpoints/model.h5',
                          unfreeze_strategy='linear'):
    """
    Train the model with batch normalization and advanced techniques

    Args:
        model: Model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        batch_size: Batch size
        epochs: Number of epochs
        checkpoint_path: Path to save checkpoints
        unfreeze_strategy: Strategy for unfreezing layers

    Returns:
        history: Training history
    """
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )

    # Create callbacks
    callbacks = create_callbacks(checkpoint_path)

    # Add gradual unfreezing callback
    gradual_unfreeze = GradualUnfreezing(
        model=model,
        total_epochs=epochs,
        unfreeze_strategy=unfreeze_strategy
    )
    callbacks.append(gradual_unfreeze)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    return history


def evaluate_model_performance(model, X_test, y_test):
    """
    Evaluate model performance on test data

    Args:
        model: Trained model
        X_test, y_test: Test data

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Calculate metrics
    loss, accuracy, precision, recall, auc = model.evaluate(X_test, y_test, verbose=0)

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    # Return all metrics
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }

    # Print metrics
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    return metrics
