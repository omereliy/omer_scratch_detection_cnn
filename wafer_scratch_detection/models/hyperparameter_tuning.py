"""
Hyperparameter tuning utilities for wafer scratch detection models.
"""

import numpy as np
import keras
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import KFold
from kerastuner import RandomSearch, BayesianOptimization, Hyperband



def build_model_with_hyperparameters(hp, input_shape=(52, 52, 3), base_model=None):
    """
    Build a model with hyperparameters for tuning

    Args:
        hp: HyperParameters object
        input_shape: Input shape of the model
        base_model: Pre-trained model to use as base (if provided)

    Returns:
        model: Model with hyperparameters
    """
    # If base model is provided, use it
    if base_model is not None:
        inputs = keras.layers.Input(shape=input_shape)
        base = base_model(inputs)

        # Add new classification layers
        x = base
    else:
        # Create a new model from scratch
        inputs = keras.layers.Input(shape=input_shape)

        # CNN architecture
        x = inputs
        for i in range(hp.Int('conv_blocks', 2, 5)):
            filters = hp.Int(f'filters_{i}', min_value=32, max_value=256, step=32)
            kernel_size = hp.Choice(f'kernel_size_{i}', values=[3, 5])

            x = keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                activation='relu'
            )(x)

            if hp.Boolean(f'batch_norm_{i}'):
                x = keras.layers.BatchNormalization()(x)

            if hp.Boolean(f'max_pool_{i}'):
                x = keras.layers.MaxPooling2D()(x)

        x = keras.layers.GlobalAveragePooling2D()(x)

    # Dense layers
    for i in range(hp.Int('dense_blocks', 1, 3)):
        units = hp.Int(f'units_{i}', min_value=64, max_value=512, step=64)
        x = keras.layers.Dense(units, activation='relu')(x)

        if hp.Boolean(f'batch_norm_dense_{i}'):
            x = keras.layers.BatchNormalization()(x)

        dropout_rate = hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)
        if dropout_rate > 0:
            x = keras.layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
    if optimizer_choice == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    return model


class TimeCallback(keras.callbacks.Callback):
    """Callback to track training time per epoch"""

    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.times.append(epoch_time)
        logs['time'] = epoch_time


def run_hyperparameter_search(X_train, y_train, X_val, y_val,
                              project_name='wafer_scratch_detection',
                              max_trials=10, executions_per_trial=1,
                              epochs=10, search_strategy='hyperband',
                              input_shape=(52, 52, 3),
                              base_model=None):
    """
    Run hyperparameter search

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        project_name: Name of the project
        max_trials: Maximum number of trials for the search
        executions_per_trial: Number of executions per trial
        epochs: Number of epochs per trial
        search_strategy: Search strategy ('random', 'bayesian', or 'hyperband')
        input_shape: Input shape of the model
        base_model: Pre-trained model to use as base (if provided)

    Returns:
        tuner: Hyperparameter tuner
        best_model: Best model from the search
    """

    # Define the model-building function
    def model_builder(hp):
        return build_model_with_hyperparameters(hp, input_shape, base_model)

    # Choose search strategy
    if search_strategy == 'bayesian':
        tuner = BayesianOptimization(
            model_builder,
            objective='val_accuracy',
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory='tuner_results',
            project_name=project_name
        )
    elif search_strategy == 'hyperband':
        tuner = Hyperband(
            model_builder,
            objective='val_accuracy',
            max_epochs=epochs,
            factor=3,
            directory='tuner_results',
            project_name=project_name
        )
    else:  # Default to random search
        tuner = RandomSearch(
            model_builder,
            objective='val_accuracy',
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory='tuner_results',
            project_name=project_name
        )

    # Define callbacks for each trial
    time_callback = TimeCallback()
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Search for best hyperparameters
    tuner.search(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, time_callback]
    )

    # Get the best model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)

    # Print best hyperparameters
    print("\nBest Hyperparameters:")
    print(best_hps.values)

    return tuner, best_model


def cross_validate_hyperparameters(X, y, n_splits=5, input_shape=(52, 52, 3), stratify=True, shuffle=True,
                                   random_state=42):
    """
    Cross-validate hyperparameter search with multiple folds, matching the behavior
    of the random forest training.

    Args:
        X: Data features
        y: Data labels
        n_splits: Number of folds for cross-validation
        input_shape: Input shape of the model
        stratify: Whether to stratify the folds (preserve class distribution)
        shuffle: Whether to shuffle before splitting
        random_state: Random state for reproducibility

    Returns:
        best_params_list: List of the best hyperparameters for each fold
        cv_scores: Cross-validation scores
    """

    # Use StratifiedKFold if stratify=True, otherwise use regular KFold
    if stratify:
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        split_gen = kf.split(X, y)
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        split_gen = kf.split(X)

    best_params_list = []
    cv_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    all_fold_results = []

    for fold, (train_idx, val_idx) in enumerate(split_gen):
        print(f"\nFold {fold + 1}/{n_splits}")

        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Check class distribution in train and validation sets
        print(f"Train set class distribution: {np.bincount(y_train.astype(int).flatten())}")
        print(f"Validation set class distribution: {np.bincount(y_val.astype(int).flatten())}")

        # Run hyperparameter search
        try:
            tuner, best_model = run_hyperparameter_search(
                X_train, y_train, X_val, y_val,
                project_name=f'wafer_fold_{fold}',
                max_trials=5,  # Limited trials for each fold
                epochs=5,  # Limited epochs for each fold
                input_shape=input_shape
            )

            # Evaluate best model
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            best_params_list.append(best_hps.values)

            # Train best model
            best_model.fit(
                X_train, y_train,
                epochs=10,
                validation_data=(X_val, y_val),
                callbacks=[keras.callbacks.EarlyStopping(patience=3)]
            )

            # Evaluate on validation set
            loss, accuracy, precision, recall = best_model.evaluate(X_val, y_val)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

            fold_result = {
                'fold': fold + 1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'best_params': best_hps.values
            }
            all_fold_results.append(fold_result)

            cv_scores['accuracy'].append(accuracy)
            cv_scores['precision'].append(precision)
            cv_scores['recall'].append(recall)
            cv_scores['f1'].append(f1)

            print(
                f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        except Exception as e:
            print(f"Error in fold {fold + 1}: {e}")
            continue

    # Print average scores
    if cv_scores['accuracy']:
        print("\nCross-Validation Results:")
        for metric, scores in cv_scores.items():
            print(f"Average {metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # Return detailed results for further analysis
    return best_params_list, cv_scores, all_fold_results


def visualize_hyperparameter_effects(tuner):
    """
    Visualize the effects of different hyperparameters

    Args:
        tuner: Trained hyperparameter tuner
    """

    # Extract hyperparameters and performance from all trials
    trials_data = []
    for trial in tuner.oracle.trials.values():
        if trial.score is not None:  # Only completed trials
            trial_data = {
                'score': trial.score,
                **trial.hyperparameters.values
            }
            trials_data.append(trial_data)

    if not trials_data:
        print("No completed trials to analyze.")
        return

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    # Plot common hyperparameters
    common_params = ['learning_rate', 'optimizer', 'dropout_0', 'units_0']

    for i, param in enumerate(common_params[:4]):  # Limit to 4 plots
        if param not in trials_data[0]:
            continue

        param_values = [trial[param] for trial in trials_data]
        scores = [trial['score'] for trial in trials_data]

        # For categorical parameters
        if isinstance(param_values[0], str):
            # Group by category
            categories = {}
            for val, score in zip(param_values, scores):
                if val not in categories:
                    categories[val] = []
                categories[val].append(score)

            # Calculate mean score for each category
            cat_names = list(categories.keys())
            cat_scores = [np.mean(categories[cat]) for cat in cat_names]

            # Plot
            axes[i].bar(cat_names, cat_scores)
            axes[i].set_title(f'Effect of {param}')
            axes[i].set_ylabel('Validation Score')

        else:  # For numerical parameters
            axes[i].scatter(param_values, scores)
            axes[i].set_title(f'Effect of {param}')
            axes[i].set_xlabel(param)
            axes[i].set_ylabel('Validation Score')

    plt.tight_layout()
    plt.show()


def fine_tune_best_model(best_model, X_train, y_train, X_val, y_val, epochs=30):
    """
    Fine-tune the best model from hyperparameter search

    Args:
        best_model: Best model from hyperparameter search
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of epochs for fine-tuning

    Returns:
        history: Training history
    """
    # Create callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )

    # Fine-tune the model
    history = best_model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr]
    )

    return history