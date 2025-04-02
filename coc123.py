import numpy as np

# Please write the optimal hyperparameter values you obtain in the global variable 'optimal_hyperparm' below. This
# variable should contain the values when I look at your submission. I should not have to run your code to populate this
# variable.
optimal_hyperparam = {}


class COC131:
    def q1(self, filename=None):
        """
        This function should be used to load the data. To speed-up processing in later steps, lower resolution of the
        image to 32*32. The folder names in the root directory of the dataset are the class names. After loading the
        dataset, you should save it into an instance variable self.x (for samples) and self.y (for labels). Both self.x
        and self.y should be numpy arrays of dtype float.

        :param filename: this is the name of an actual random image in the dataset. You don't need this to load the
        dataset. This is used by me for testing your implementation.
        :return res1: a one-dimensional numpy array containing the flattened low-resolution image in file 'filename'.
        Flatten the image in the row major order. The dtype for the array should be float.
        :return res2: a string containing the class name for the image in file 'filename'. This string should be same as
        one of the folder names in the originally shared dataset.
        """

        import os
        import numpy as np
        from PIL import Image

        # Dataset path pointing directly to EuroSAT_RGB
        dataset_path = "../datasets/EuroSAT_RGB/"

        images = []
        labels = []
        label_to_index = {}
        index_counter = 0

        res1 = np.zeros(1)
        res2 = ""

        # Process class folders
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)

            # Skip non-directories
            if not os.path.isdir(class_path):
                continue

            # Assign numeric index to class
            if class_name not in label_to_index:
                label_to_index[class_name] = index_counter
                index_counter += 1

            # Process images in class folder
            for img_file in os.listdir(class_path):
                if not img_file.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tif", ".bmp")
                ):
                    continue

                img_path = os.path.join(class_path, img_file)

                # Check if this matches the requested filename
                if filename and img_file == filename:
                    try:
                        img = Image.open(img_path)
                        img_resized = img.resize((32, 32))
                        img_array = np.array(img_resized, dtype=float)
                        res1 = img_array.reshape(-1)
                        res2 = class_name
                    except Exception as e:
                        print(
                            f"Error processing requested file {img_path}: {e}"
                        )

                # Process image for dataset
                try:
                    img = Image.open(img_path)
                    img_resized = img.resize((32, 32))
                    img_array = np.array(img_resized, dtype=float)
                    img_flat = img_array.reshape(-1)

                    images.append(img_flat)
                    labels.append(label_to_index[class_name])
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")

        # Store data in instance variables
        self.x = np.array(images, dtype=float)
        self.y = np.array(labels, dtype=float)

        return res1, res2

    def q2(self, inp):
        """
        This function should compute the standardized data from a given 'inp' data. The function should work for a
        dataset with any number of features.

        SD: This function standardizes the input data to have a standard deviation of 2.5. It first
        centers the data by subtracting the mean, then scales it by dividing by the standard deviation,
        and finally multiplies by 2.5 to achieve the required standard deviation. The function works
        with datasets of any dimensionality and feature count.

        :param inp: an array from which the standardized data is to be computed.
        :return res2: a numpy array containing the standardized data with standard deviation of 2.5. The array should
        have the same dimensions as the original data
        :return res1: sklearn object used for standardization.
        """
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        # Create a standard scaler object
        scaler = StandardScaler()

        # Check if input is 1D, if so reshape for sklearn
        original_shape = inp.shape
        is_1d = len(original_shape) == 1

        if is_1d:
            inp_reshaped = inp.reshape(-1, 1)
        else:
            inp_reshaped = inp

        # Fit the scaler to the data and transform it
        # This will standardize to mean=0, std=1
        standard_data = scaler.fit_transform(inp_reshaped)

        # Scale to std=2.5 by multiplying by 2.5
        standard_data_scaled = standard_data * 2.5

        # Reshape back to original if input was 1D
        if is_1d:
            standard_data_scaled = standard_data_scaled.reshape(original_shape)

        # Return the standardized data and the scaler
        return standard_data_scaled, scaler

    def q3(self, test_size=None, pre_split_data=None, hyperparam=None):
        """
        This function should build a MLP Classifier using the dataset loaded in function 'q1' and evaluate model
        performance. You can assume that the function 'q1' has been called prior to calling this function. This function
        should support hyperparameter optimizations.

        SD: This function builds and evaluates an MLP classifier on the dataset loaded in q1. It performs
        hyperparameter optimization to find the best model configuration. The function tracks loss, training accuracy,
        and testing accuracy during training. It supports custom test set sizes and pre-split data. It uses PCA for
        dimensionality reduction to improve performance.

        :param test_size: the proportion of the dataset that should be reserved for testing. This should be a fraction
        between 0 and 1.
        :param pre_split_data: Can be used to provide data already split into training and testing.
        :param hyperparam: hyperparameter values to be tested during hyperparameter optimization.
        :return: The function should return 1 model object and 3 numpy arrays which contain the loss, training accuracy
        and testing accuracy after each training iteration for the best model you found.
        """
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.metrics import accuracy_score
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
        import numpy as np
        import matplotlib.pyplot as plt
        import copy

        # Set default test size if not provided
        if test_size is None:
            test_size = 0.3  # Using 30% for testing as per requirements

        # Normalize the data using the function from q2
        X_standardized, scaler = self.q2(self.x)
        y = self.y

        # Apply PCA for dimensionality reduction
        # Choose n_components to preserve 95% of variance
        print("Applying PCA for dimensionality reduction...")
        pca = PCA(n_components=0.95)
        X_reduced = pca.fit_transform(X_standardized)

        print(
            f"Reduced dimensions from {X_standardized.shape[1]} to {X_reduced.shape[1]} features"
        )
        print(
            f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}"
        )

        # Create a visualization of the explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Explained Variance by PCA Components")
        plt.grid(True)
        plt.axhline(
            y=0.95, color="r", linestyle="--", label="95% Variance Threshold"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig("../visualizations/q3-3.png", dpi=300, bbox_inches="tight")

        # Subsample data for faster computation
        n_samples = min(int(len(X_reduced) * 0.1), 2000)
        indices = np.random.choice(len(X_reduced), n_samples, replace=False)
        X_subsample = X_reduced[indices]
        y_subsample = y[indices]

        # Split the data into training and testing sets if pre-split data is not provided
        if pre_split_data is None:
            # Use subsampled data for hyperparameter tuning
            X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
                X_subsample,
                y_subsample,
                test_size=test_size,
                random_state=42,
                stratify=y_subsample,
            )

            # Also create full train/test split for final model
            X_train, X_test, y_train, y_test = train_test_split(
                X_reduced, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            # Note: pre_split_data would need to already have PCA applied
            # This is a placeholder - in practice you might need to adapt this
            X_train, X_test, y_train, y_test = pre_split_data

            # Create subsamples from the pre-split data
            n_train = min(int(len(X_train) * 0.1), 1500)
            train_indices = np.random.choice(
                len(X_train), n_train, replace=False
            )
            X_train_sub, y_train_sub = (
                X_train[train_indices],
                y_train[train_indices],
            )

            n_test = min(int(len(X_test) * 0.1), 500)
            test_indices = np.random.choice(len(X_test), n_test, replace=False)
            X_test_sub, y_test_sub = X_test[test_indices], y_test[test_indices]

        # Define hyperparameters for grid search if not provided
        if hyperparam is None:
            hyperparam = {
                "hidden_layer_sizes": [(50,), (100,), (50, 25)],
                "activation": ["relu", "tanh"],
                "solver": ["adam"],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate": ["adaptive"],
                "max_iter": [100],
                "early_stopping": [True],
                "n_iter_no_change": [10],
            }

        # Perform grid search to find the best hyperparameters
        print("Starting Grid Search...")
        grid_search = GridSearchCV(
            MLPClassifier(random_state=42),
            hyperparam,
            cv=2,  # Using 2-fold CV to save time
            n_jobs=-1,
            verbose=1,
            scoring="accuracy",
        )
        # Use subsampled data for grid search
        grid_search.fit(X_train_sub, y_train_sub)

        best_params = grid_search.best_params_
        print(f"Best hyperparameters found: {best_params}")

        # Store optimal hyperparameters in the global variable
        global optimal_hyperparam
        optimal_hyperparam = best_params
        # Also store as instance attribute for easier access
        self.optimal_hyperparam = best_params

        # Initialize tracking arrays
        loss_curve = []
        train_acc_curve = []
        test_acc_curve = []

        # Set a smaller number of iterations for each mini-training
        partial_epochs = 10  # Train 10 epochs at a time
        max_epochs = 200  # Total maximum epochs

        # Create and train the model incrementally
        print("Training final model incrementally...")
        best_params_copy = copy.deepcopy(best_params)
        if "max_iter" in best_params_copy:
            del best_params_copy["max_iter"]

        model = MLPClassifier(
            **best_params_copy,
            max_iter=partial_epochs,
            random_state=42,
            warm_start=True,
            verbose=False,
        )

        # Track progress while incrementally training
        for i in range(0, max_epochs, partial_epochs):
            # Fit the model for a few epochs
            model.fit(X_train, y_train)

            # Add the latest loss to our curve
            if hasattr(model, "loss_curve_"):
                if model.loss_curve_:
                    loss_curve.append(model.loss_curve_[-1])

            # Calculate and store training accuracy on a subset
            train_indices = np.random.choice(
                len(X_train), min(1000, len(X_train)), replace=False
            )
            train_subset_x = X_train[train_indices]
            train_subset_y = y_train[train_indices]
            train_pred = model.predict(train_subset_x)
            train_acc = accuracy_score(train_subset_y, train_pred)
            train_acc_curve.append(train_acc)

            # Calculate and store test accuracy
            test_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, test_pred)
            test_acc_curve.append(test_acc)

            # Print progress
            print(
                f"Epoch {i+partial_epochs}: Loss={loss_curve[-1]:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}"
            )

            # Check for convergence or early stopping
            if test_acc > 0.95 or (loss_curve and loss_curve[-1] < 0.01):
                print("Early stopping due to good performance")
                break

        # Convert tracking lists to numpy arrays
        loss_curve = np.array(loss_curve)
        train_acc_curve = np.array(train_acc_curve)
        test_acc_curve = np.array(test_acc_curve)

        # Create visualizations
        print("Creating visualizations...")

        # Visualize the results
        plt.figure(figsize=(18, 6))

        # Plot loss curve
        plt.subplot(1, 3, 1)
        plt.plot(loss_curve)
        plt.title("Loss Curve")
        plt.xlabel("Training Iterations")
        plt.ylabel("Loss")
        plt.grid(True)

        # Plot training accuracy
        plt.subplot(1, 3, 2)
        plt.plot(train_acc_curve)
        plt.title("Training Accuracy")
        plt.xlabel("Training Iterations")
        plt.ylabel("Accuracy")
        plt.grid(True)

        # Plot testing accuracy
        plt.subplot(1, 3, 3)
        plt.plot(test_acc_curve)
        plt.title("Testing Accuracy")
        plt.xlabel("Training Iterations")
        plt.ylabel("Accuracy")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("../visualizations/q3-1.png", dpi=300, bbox_inches="tight")

        # Create a visualization of hyperparameter impact
        top_n_models = min(5, len(grid_search.cv_results_["params"]))
        results = grid_search.cv_results_

        # Sort by mean test score
        indices = np.argsort(results["mean_test_score"])[::-1][:top_n_models]

        plt.figure(figsize=(12, 6))
        plt.title(f"Top {top_n_models} Model Configurations")

        # Get scores for bar chart
        scores = [results["mean_test_score"][i] for i in indices]

        # Create bar chart with more visible differences
        bars = plt.bar(range(top_n_models), scores)

        # Add score text on top of each bar
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{score:.4f}",
                ha="center",
                va="bottom",
                rotation=0,
            )

        plt.xticks(
            range(top_n_models), [f"Model {i+1}" for i in range(top_n_models)]
        )
        plt.ylabel("Mean Validation Accuracy")

        # Adjust y-axis to better show differences between models
        min_score = min(scores) - 0.01
        max_score = max(scores) + 0.01
        plt.ylim(min_score, max_score)

        # Add parameter details as text
        for i, idx in enumerate(indices):
            params = results["params"][idx]
            param_text = (
                f"hidden_layers: {params['hidden_layer_sizes']}\n"
                f"activation: {params['activation']}\n"
                f"solver: {params['solver']}\n"
                f"alpha: {params['alpha']}\n"
                f"learning_rate: {params['learning_rate']}"
            )
            plt.annotate(
                param_text,
                xy=(i, scores[i]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.3),
                rotation=90,
            )

        plt.tight_layout()
        plt.savefig("../visualizations/q3-2.png", dpi=300, bbox_inches="tight")

        # Create object that remembers the PCA transformation
        model_with_pca = {"pca": pca, "scaler": scaler, "mlp": model}

        # Return the required values
        return model_with_pca, loss_curve, train_acc_curve, test_acc_curve

    def q4(self):
        """
        This function should study the impact of alpha on the performance and parameters of the model. For each value of
        alpha in the list below, train a separate MLPClassifier from scratch. Other hyperparameters for the model can
        be set to the best values you found in 'q3'. You can assume that the function 'q1' has been called
        prior to calling this function.

        SD: This function evaluates how different alpha values affect the MLP model's weights, biases,
        and overall performance. It trains separate models with each alpha value and visualizes the results,
        showing both performance metrics and parameter distributions.

        :return: res should be the data you visualized.
        """
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.decomposition import PCA
        import numpy as np
        import matplotlib.pyplot as plt

        # Alpha values to test
        alpha_values = [
            0,
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
            0.5,
            1,
            2,
            5,
            10,
            50,
            100,
        ]

        # Apply the same preprocessing as in q3
        X_standardized, scaler = self.q2(self.x)

        # Apply PCA for dimensionality reduction (same as in q3)
        pca = PCA(n_components=0.95)
        X_reduced = pca.fit_transform(X_standardized)
        print(
            f"Reduced dimensions from {X_standardized.shape[1]} to {X_reduced.shape[1]} features"
        )

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced, self.y, test_size=0.3, random_state=42, stratify=self.y
        )

        # Use the best hyperparameters from q3 (except alpha)
        # Access the hyperparameters through the instance variable or global variable
        if hasattr(self, "optimal_hyperparam"):
            best_params = self.optimal_hyperparam.copy()
        else:
            # Default parameters if optimal_hyperparam is not available
            best_params = {
                "hidden_layer_sizes": (100,),
                "activation": "relu",
                "solver": "adam",
                "learning_rate": "adaptive",
                "max_iter": 200,
                "early_stopping": True,
                "n_iter_no_change": 10,
            }

        # Remove alpha if present in best_params
        if "alpha" in best_params:
            del best_params["alpha"]

        # Initialize lists to store results
        train_accuracies = []
        test_accuracies = []
        weight_stats = []  # For storing statistics about weights
        bias_stats = []  # For storing statistics about biases
        models = []  # Store models for later analysis

        print("Training models with different alpha values...")
        for alpha in alpha_values:
            print(f"Training model with alpha={alpha}")

            # Create and train the model with the current alpha
            model = MLPClassifier(
                **best_params, alpha=alpha, random_state=42, verbose=False
            )

            model.fit(X_train, y_train)
            models.append(model)

            # Evaluate on training and test sets
            train_pred = model.predict(X_train)
            train_acc = accuracy_score(y_train, train_pred)
            train_accuracies.append(train_acc)

            test_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, test_pred)
            test_accuracies.append(test_acc)

            # Extract and analyze weights and biases
            weight_magnitudes = []
            for i, w in enumerate(model.coefs_):
                # Flatten weights and calculate statistics
                flat_weights = w.flatten()
                weight_magnitudes.extend(np.abs(flat_weights))

            bias_magnitudes = []
            for i, b in enumerate(model.intercepts_):
                # Flatten biases and calculate statistics
                flat_biases = b.flatten()
                bias_magnitudes.extend(np.abs(flat_biases))

            # Store statistics
            weight_stats.append(
                {
                    "mean": np.mean(weight_magnitudes),
                    "std": np.std(weight_magnitudes),
                    "max": np.max(weight_magnitudes),
                    "min": np.min(weight_magnitudes),
                    "median": np.median(weight_magnitudes),
                }
            )

            bias_stats.append(
                {
                    "mean": np.mean(bias_magnitudes),
                    "std": np.std(bias_magnitudes),
                    "max": np.max(bias_magnitudes),
                    "min": np.min(bias_magnitudes),
                    "median": np.median(bias_magnitudes),
                }
            )

            print(
                f"  Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}"
            )
            print(
                f"  Mean weight magnitude: {weight_stats[-1]['mean']:.4f}, Mean bias magnitude: {bias_stats[-1]['mean']:.4f}"
            )

        # Create visualizations
        print("Creating visualizations...")

        # 1. Plot accuracy vs alpha
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.semilogx(
            alpha_values, train_accuracies, "o-", label="Train Accuracy"
        )
        plt.semilogx(alpha_values, test_accuracies, "s-", label="Test Accuracy")
        plt.xlabel("Alpha (log scale)")
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy vs Alpha")
        plt.grid(True)
        plt.legend()

        # 2. Plot weight statistics vs alpha
        plt.subplot(2, 2, 2)
        plt.semilogx(
            alpha_values, [s["mean"] for s in weight_stats], "o-", label="Mean"
        )
        plt.semilogx(
            alpha_values,
            [s["median"] for s in weight_stats],
            "s-",
            label="Median",
        )
        plt.xlabel("Alpha (log scale)")
        plt.ylabel("Weight Magnitude")
        plt.title("Weight Statistics vs Alpha")
        plt.grid(True)
        plt.legend()

        # 3. Plot weight standard deviation vs alpha
        plt.subplot(2, 2, 3)
        plt.semilogx(alpha_values, [s["std"] for s in weight_stats], "o-")
        plt.xlabel("Alpha (log scale)")
        plt.ylabel("Standard Deviation")
        plt.title("Weight Variability vs Alpha")
        plt.grid(True)

        # 4. Plot bias statistics vs alpha
        plt.subplot(2, 2, 4)
        plt.semilogx(
            alpha_values, [s["mean"] for s in bias_stats], "o-", label="Mean"
        )
        plt.semilogx(
            alpha_values,
            [s["median"] for s in bias_stats],
            "s-",
            label="Median",
        )
        plt.xlabel("Alpha (log scale)")
        plt.ylabel("Bias Magnitude")
        plt.title("Bias Statistics vs Alpha")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig("../visualizations/q4-1.png", dpi=300, bbox_inches="tight")

        # 5. Create a visualization of weight distributions for selected alpha values
        selected_indices = [
            0,
            3,
            7,
            12,
        ]  # Corresponding to alpha = 0, 0.01, 1, 100
        selected_alphas = [alpha_values[i] for i in selected_indices]
        selected_models = [models[i] for i in selected_indices]

        plt.figure(figsize=(16, 10))
        for i, (alpha, model) in enumerate(
            zip(selected_alphas, selected_models)
        ):
            # Get all weights from the first layer
            weights = model.coefs_[0].flatten()

            plt.subplot(2, 2, i + 1)
            plt.hist(weights, bins=50, alpha=0.7)
            plt.title(f"Weight Distribution (alpha={alpha})")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")
            plt.grid(True)

            # Add statistics annotations
            plt.annotate(
                f"Mean: {np.mean(weights):.4f}\nStd: {np.std(weights):.4f}\n"
                f"Max: {np.max(weights):.4f}\nMin: {np.min(weights):.4f}",
                xy=(0.05, 0.95),
                xycoords="axes fraction",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
                verticalalignment="top",
            )

        plt.tight_layout()
        plt.savefig("../visualizations/q4-2.png", dpi=300, bbox_inches="tight")

        # 6. Visualize overfitting vs regularization with two representative alphas
        # Choose a low alpha (minimal regularization) and high alpha (strong regularization)
        low_alpha_idx = 0  # alpha = 0
        high_alpha_idx = 7  # alpha = 1

        # Get models
        low_alpha_model = models[low_alpha_idx]
        high_alpha_model = models[high_alpha_idx]

        # Generate predictions across a range of epochs
        epoch_range = range(1, 11)  # 10 measurement points
        low_alpha_train_acc = []
        low_alpha_test_acc = []
        high_alpha_train_acc = []
        high_alpha_test_acc = []

        best_params_for_epochs = best_params.copy()
        if "max_iter" in best_params_for_epochs:
            del best_params_for_epochs["max_iter"]

        # Train models with incremental number of iterations
        for n_iter in epoch_range:
            # Low alpha model
            model_low = MLPClassifier(
                **best_params_for_epochs,
                alpha=alpha_values[low_alpha_idx],
                max_iter=n_iter * 20,  # Scale iterations
                random_state=42,
            )
            model_low.fit(X_train, y_train)

            low_alpha_train_acc.append(
                accuracy_score(y_train, model_low.predict(X_train))
            )
            low_alpha_test_acc.append(
                accuracy_score(y_test, model_low.predict(X_test))
            )

            # High alpha model
            model_high = MLPClassifier(
                **best_params_for_epochs,  # Use the version without max_iter
                alpha=alpha_values[high_alpha_idx],
                max_iter=n_iter * 20,  # Scale iterations
                random_state=42,
            )
            model_high.fit(X_train, y_train)

            high_alpha_train_acc.append(
                accuracy_score(y_train, model_high.predict(X_train))
            )
            high_alpha_test_acc.append(
                accuracy_score(y_test, model_high.predict(X_test))
            )

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epoch_range, low_alpha_train_acc, "o-", label="Train")
        plt.plot(epoch_range, low_alpha_test_acc, "s-", label="Test")
        plt.title(f"Learning Curve (alpha={alpha_values[low_alpha_idx]})")
        plt.xlabel("Epoch Group (x20 iterations)")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epoch_range, high_alpha_train_acc, "o-", label="Train")
        plt.plot(epoch_range, high_alpha_test_acc, "s-", label="Test")
        plt.title(f"Learning Curve (alpha={alpha_values[high_alpha_idx]})")
        plt.xlabel("Epoch Group (x20 iterations)")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig("../visualizations/q4-3.png", dpi=300, bbox_inches="tight")

        # Return the collected data for visualization
        res = {
            "alpha_values": alpha_values,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies,
            "weight_stats": weight_stats,
            "bias_stats": bias_stats,
        }

        return res

    def q5(self):
        """
        This function should perform hypothesis testing to study the impact of using CV with and without Stratification
        on the performance of MLPClassifier. Set other model hyperparameters to the best values obtained in the previous
        questions. Use 5-fold cross validation for this question. You can assume that the function 'q1' has been called
        prior to calling this function.

        :return: The function should return 4 items - the final testing accuracy for both methods of CV, p-value of the
        test and a string representing the result of hypothesis testing. The string can have only two possible values -
        'Splitting method impacted performance' and 'Splitting method had no effect'.
        """

        res1 = 0
        res2 = 0
        res3 = 0
        res4 = ""

        return res1, res2, res3, res4

    def q6(self):
        """
        This function should perform unsupervised learning using LocallyLinearEmbedding in Sklearn. You can assume that
        the function 'q1' has been called prior to calling this function.

        :return: The function should return the data you visualize.
        """

        res = np.zeros(1)

        return res
