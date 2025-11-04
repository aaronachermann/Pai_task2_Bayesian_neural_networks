import abc
import collections
import enum
import math
import pathlib
import typing
import warnings

import numpy as np
import torch
import torch.optim
import torch.utils.data
import tqdm
from matplotlib import pyplot as plt

from util import paint_reliability_diagram, compute_cost, seed_setup, calculate_calibration_curve

ENABLE_EXTENDED_ANALYSIS = False
"""
Set `ENABLE_EXTENDED_ANALYSIS` to `True` in order to generate additional plots on validation data.
"""

USE_PRETRAINED_WEIGHTS = True
"""
If `USE_PRETRAINED_WEIGHTS` is `True`, then MAP inference uses provided pretrained weights.
You should not modify MAP training or the CNN architecture before passing the baseline.
If you set the constant to `False` (to further experiment),
this solution always performs MAP inference before running your SWAG implementation.
Note that MAP inference can take a long time.
"""


def main():
    raise RuntimeError(
        "This main() method is for illustrative purposes only"
        " and will NEVER be called when running your solution to generate your submission file!\n"
        "The checker always directly interacts with your SWAInferenceHandler class and run_evaluation method.\n"
        "You can remove this exception for local testing, but be aware that any changes to the main() method"
        " are ignored when generating your submission file."
    )

    data_location = pathlib.Path.cwd()
    model_location = pathlib.Path.cwd()
    output_location = pathlib.Path.cwd()

    # Load training data
    training_images = torch.from_numpy(np.load(data_location / "train_xs.npz")["train_xs"])
    training_metadata = np.load(data_location / "train_ys.npz")
    training_labels = torch.from_numpy(training_metadata["train_ys"])
    training_snow_labels = torch.from_numpy(training_metadata["train_is_snow"])
    training_cloud_labels = torch.from_numpy(training_metadata["train_is_cloud"])
    training_dataset = torch.utils.data.TensorDataset(training_images, training_snow_labels, training_cloud_labels, training_labels)

    # Load validation data
    validation_images = torch.from_numpy(np.load(data_location / "val_xs.npz")["val_xs"])
    validation_metadata = np.load(data_location / "val_ys.npz")
    validation_labels = torch.from_numpy(validation_metadata["val_ys"])
    validation_snow_labels = torch.from_numpy(validation_metadata["val_is_snow"])
    validation_cloud_labels = torch.from_numpy(validation_metadata["val_is_cloud"])
    validation_dataset = torch.utils.data.TensorDataset(validation_images, validation_snow_labels, validation_cloud_labels, validation_labels)

    # Fix all randomness
    seed_setup()

    # Build and run the actual solution
    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )
    swag_inference = SWAInferenceHandler(
        train_xs=training_dataset.tensors[0],
        model_dir=model_location,
    )
    swag_inference.train_model(training_loader)
    swag_inference.run_calibration(validation_dataset)

    # fork_rng ensures that the evaluation does not change the rng state.
    # That way, you should get exactly the same results even if you remove evaluation
    # to save computational time when developing the task
    # (as long as you ONLY use torch randomness, and not e.g. random or numpy.random).
    with torch.random.fork_rng():
        run_evaluation(swag_inference, validation_dataset, ENABLE_EXTENDED_ANALYSIS, output_location)


class InferenceMode(enum.Enum):
    """
    Inference mode switch for your implementation.
    `MAP` simply predicts the most likely class using pretrained MAP weights.
    `SWAG_DIAGONAL` and `SWAG_FULL` correspond to SWAG-diagonal and the full SWAG method, respectively.
    """
    MAP = 0
    SWAG_DIAGONAL = 1
    SWAG_FULL = 2


class SWAInferenceHandler(object):
    """
    Your implementation of SWA-Gaussian.
    This class is used to run and evaluate your solution.
    You must preserve all methods and signatures of this class.
    However, you can add new methods if you want.

    We provide basic functionality and some helper methods.
    You can pass the baseline by only modifying methods marked with TODO.
    However, we encourage you to skim other methods in order to gain a better understanding of SWAG.
    """

    def __init__(
        self,
        train_xs: torch.Tensor,
        model_dir: pathlib.Path,
        # TODO(1): change inference_mode to InferenceMode.SWAG_DIAGONAL
        # TODO(2): change inference_mode to InferenceMode.SWAG_FULL
        inference_mode: InferenceMode = InferenceMode.SWAG_DIAGONAL,
        # TODO(2): optionally add/tweak hyperparameters
        swag_training_epochs: int = 50,
        swag_lr: float = 0.045,
        swag_update_interval: int = 1,
        max_rank_deviation_matrix: int = 15,
        num_bma_samples: int = 50,
    ):
        """
        :param train_xs: Training images (for storage only)
        :param model_dir: Path to directory containing pretrained MAP weights
        :param inference_mode: Control which inference mode (MAP, SWAG-diagonal, full SWAG) to use
        :param swag_training_epochs: Total number of gradient descent epochs for SWAG
        :param swag_lr: Learning rate for SWAG gradient descent
        :param swag_update_interval: Frequency (in epochs) for updating SWAG statistics during gradient descent
        :param max_rank_deviation_matrix: Rank of deviation matrix for full SWAG
        :param num_bma_samples: Number of networks to sample for Bayesian model averaging during prediction
        """

        self.model_dir = model_dir
        self.inference_mode = inference_mode
        self.swag_training_epochs = swag_training_epochs
        self.swag_lr = swag_lr
        self.swag_update_interval = swag_update_interval
        self.max_rank_deviation_matrix = max_rank_deviation_matrix
        self.num_bma_samples = num_bma_samples

        # Network used to perform SWAG.
        # Note that all operations in this class modify this network IN-PLACE!
        self.network = CNN(in_channels=3, out_classes=6)

        # Store training dataset to recalculate batch normalization statistics during SWAG inference
        self.training_dataset = torch.utils.data.TensorDataset(train_xs)

        # SWAG-diagonal
        # TODO(1): create attributes for SWAG-diagonal
        #  Hint: self._create_weight_copy() creates an all-zero copy of the weights
        #  as a dictionary that maps from weight name to values.
        #  Hint: you never need to consider the full vector of weights,
        #  but can always act on per-layer weights (in the format that _create_weight_copy() returns)
        self.swag_mean_weights = self._create_weight_copy()  # θ_SWA: first moment
        self.swag_second_moment = self._create_weight_copy()  # θ²: second moment
        self.num_swag_models = 0  # number of models collected

        # Full SWAG
        # TODO(2): create attributes for SWAG-full
        #  Hint: check collections.deque
        self.deviation_matrix_buffer = {
            name: collections.deque(maxlen=max_rank_deviation_matrix)
            for name, _ in self.network.named_parameters()
        }
        self.swag_running_mean = self._create_weight_copy()  # θ_running for deviation calculation

        # Calibration, prediction, and other attributes
        # TODO(2): create additional attributes, e.g., for calibration
        self._calibration_threshold = None  # this is an example, feel free to be creative
        self._temperature = 1.5  # Temperature scaling for better calibration

    def update_swag_statistics(self) -> None:
        """
        Update SWAG statistics with the current weights of self.network.
        """

        # Create a copy of the current network weights
        copied_params = {name: param.detach() for name, param in self.network.named_parameters()}

        # SWAG-diagonal
        for name, param in copied_params.items():
            # TODO(1): update SWAG-diagonal attributes for weight `name` using `copied_params` and `param`
            if self.num_swag_models == 0:
                # First update: initialize with current weights
                self.swag_mean_weights[name] = param.clone()
                self.swag_second_moment[name] = (param.clone() ** 2)
            else:
                # Running average update
                # θ_SWA = (n·θ_SWA + θ_new) / (n+1)
                self.swag_mean_weights[name] = (self.num_swag_models * self.swag_mean_weights[name] + param) / (self.num_swag_models + 1)
                # θ² = (n·θ² + θ²_new) / (n+1)
                self.swag_second_moment[name] = (self.num_swag_models * self.swag_second_moment[name] + param ** 2) / (self.num_swag_models + 1)

        # Full SWAG
        if self.inference_mode == InferenceMode.SWAG_FULL:
            # TODO(2): update full SWAG attributes for weight `name` using `copied_params` and `param`
            for name, param in copied_params.items():
                # Calculate deviation: D_i = θ_current - θ_running
                deviation = param - self.swag_running_mean[name]
                # Append deviation to deque buffer
                self.deviation_matrix_buffer[name].append(deviation.clone())
                # Update running mean for next iteration
                self.swag_running_mean[name] = self.swag_mean_weights[name].clone()

    def fit_swag_model(self, loader: torch.utils.data.DataLoader) -> None:
        """
        Fit SWAG on top of the pretrained network self.network.
        This method should perform gradient descent with occasional SWAG updates
        by calling self.update_swag_statistics().
        """

        # We use SGD with momentum and weight decay to perform SWA.
        # See the paper on how weight decay corresponds to a type of prior.
        # Feel free to play around with optimization hyperparameters.
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.swag_lr,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )
        loss_fn = torch.nn.CrossEntropyLoss(
            reduction="mean",
        )
        # TODO(2): Update SWAGScheduler instantiation if you decided to implement a custom schedule.
        #  By default, this scheduler just keeps the initial learning rate given to `optimizer`.
        lr_scheduler = SWAGScheduler(
            optimizer,
            epochs=self.swag_training_epochs,
            steps_per_epoch=len(loader),
        )

        # TODO(1): Perform initialization for SWAG fitting
        # Initialize SWAG statistics
        self.num_swag_models = 0

        self.network.train()
        with tqdm.trange(self.swag_training_epochs, desc="Running gradient descent for SWA") as pbar:
            progress_dict = {}
            for epoch in pbar:
                avg_loss = 0.0
                avg_accuracy = 0.0
                num_samples = 0
                for batch_images, batch_snow_labels, batch_cloud_labels, batch_labels in loader:
                    optimizer.zero_grad()
                    predictions = self.network(batch_images)
                    batch_loss = loss_fn(input=predictions, target=batch_labels)
                    batch_loss.backward()
                    optimizer.step()
                    progress_dict["lr"] = lr_scheduler.get_last_lr()[0]
                    lr_scheduler.step()

                    # Calculate cumulative average training loss and accuracy
                    avg_loss = (batch_images.size(0) * batch_loss.item() + num_samples * avg_loss) / (
                        num_samples + batch_images.size(0)
                    )
                    avg_accuracy = (
                        torch.sum(predictions.argmax(dim=-1) == batch_labels).item()
                        + num_samples * avg_accuracy
                    ) / (num_samples + batch_images.size(0))
                    num_samples += batch_images.size(0)
                    progress_dict["avg. epoch loss"] = avg_loss
                    progress_dict["avg. epoch accuracy"] = avg_accuracy
                    pbar.set_postfix(progress_dict)

                # TODO(1): Implement periodic SWAG updates using the attributes defined in __init__
                # Update SWAG statistics at specified intervals
                if (epoch + 1) % self.swag_update_interval == 0:
                    self.update_swag_statistics()
                    self.num_swag_models += 1

    def run_calibration(self, validation_data: torch.utils.data.Dataset) -> None:
        """
        Calibrate your predictions using a small validation set.
        validation_data contains well-defined and ambiguous samples,
        where you can identify the latter by having label -1.
        """
        if self.inference_mode == InferenceMode.MAP:
            # In MAP mode, simply predict argmax and do nothing else
            self._calibration_threshold = 0.0
            return

        val_images = validation_data.tensors[0]
        val_labels = validation_data.tensors[3]

        # Get SWAG predictions
        print("Running calibration on validation set...")
        val_probs = self.predict_probs(val_images)
        val_max_probs, val_pred_labels = torch.max(val_probs, dim=-1)

        # Separate ambiguous and non-ambiguous samples
        ambiguous_mask = val_labels == -1
        non_ambiguous_mask = ~ambiguous_mask

        # Strategy: Find threshold that balances three costs:
        # 1. Predicting wrong class (cost=3) - worst
        # 2. Predicting "don't know" (cost=1) - medium
        # 3. Predicting correct class (cost=0) - best

        # Test many thresholds and compute FULL cost (including ECE)
        threshold_candidates = torch.linspace(0.4, 0.9, 100)
        best_total_cost = float('inf')
        best_threshold = 0.666

        for threshold in threshold_candidates:
            # Apply threshold
            thresholded_preds = torch.where(
                val_max_probs >= threshold,
                val_pred_labels,
                torch.ones_like(val_pred_labels) * -1
            )

            # Compute prediction cost
            pred_cost = compute_cost(thresholded_preds, val_labels).item()

            # Compute ECE for these predictions
            # Note: we need to recalculate "effective probabilities" after thresholding
            cal_data = calculate_calibration_curve(
                val_probs.numpy(),
                val_labels.numpy(),
                num_bins=20
            )
            ece = cal_data['ece']

            # Total cost with ECE penalty
            total_cost = pred_cost + max(ece - 0.1, 0)

            if total_cost < best_total_cost:
                best_total_cost = total_cost
                best_threshold = threshold.item()

        self._calibration_threshold = best_threshold
        print(f"✓ Selected threshold: {best_threshold:.4f} (validation cost: {best_total_cost:.4f})")

        # Detailed validation analysis
        print("\n" + "="*50)
        print("VALIDATION SET ANALYSIS")
        print("="*50)

        val_probs_final = self.predict_probs(val_images)
        val_preds_final = self.label_prediction(val_probs_final)

        # Count predictions
        num_dont_know = (val_preds_final == -1).sum().item()
        num_predictions = (val_preds_final != -1).sum().item()

        print(f"Predictions: {num_predictions}/140 ({num_predictions/140*100:.1f}%)")
        print(f"Don't know:  {num_dont_know}/140 ({num_dont_know/140*100:.1f}%)")

        # Accuracy breakdown
        non_ambig_mask = val_labels != -1
        correct = (val_preds_final[non_ambig_mask] == val_labels[non_ambig_mask]).sum().item()
        total_non_ambig = non_ambig_mask.sum().item()

        print(f"Accuracy (non-ambiguous): {correct}/{total_non_ambig} ({correct/total_non_ambig*100:.1f}%)")

        # Cost breakdown
        cost = compute_cost(val_preds_final, val_labels).item()
        print(f"Validation cost: {cost:.4f}")
        print("="*50 + "\n")

        # Fine-tune temperature on validation set
        print("Tuning temperature scaling (using 20 samples for speed)...")
        best_ece = float('inf')
        best_temp = 1.5

        # Temporarily reduce samples for faster calibration
        old_num_samples = self.num_bma_samples
        self.num_bma_samples = 20  # Reduced for speed

        for temp in torch.linspace(1.0, 2.5, 10):  # Only 10 temperatures
            self._temperature = temp.item()
            val_probs_temp = self.predict_probs(val_images)

            cal_data = calculate_calibration_curve(
                val_probs_temp.numpy(),
                val_labels.numpy(),
                num_bins=20
            )

            if cal_data['ece'] < best_ece:
                best_ece = cal_data['ece']
                best_temp = temp.item()

        # Restore original number of samples for actual predictions
        self.num_bma_samples = old_num_samples

        self._temperature = best_temp
        print(f"✓ Selected temperature: {best_temp:.4f} (ECE: {best_ece:.4f})\n")

    def _apply_temperature_scaling(
        self,
        logits: torch.Tensor,
        temperature: float = 1.5
    ) -> torch.Tensor:
        """
        Apply temperature scaling to improve calibration.
        Temperature > 1 makes predictions less confident (broader distribution).
        """
        return torch.softmax(logits / temperature, dim=-1)

    def predict_probabilities_swag(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Perform Bayesian model averaging using your SWAG statistics and predict
        probabilities for all samples in the loader.
        Outputs should be a Nx6 tensor, where N is the number of samples in loader,
        and all rows of the output should sum to 1.
        That is, output row i column j should be your predicted p(y=j | x_i).
        """

        self.network.eval()

        # Perform Bayesian model averaging:
        # Instead of sampling self.num_bma_samples networks (using self.sample_parameters())
        # for each datapoint, you can save time by sampling self.num_bma_samples networks,
        # and perform inference with each network on all samples in loader.
        model_predictions = []
        for _ in tqdm.trange(self.num_bma_samples, desc="Performing Bayesian model averaging"):
            # TODO(1): Sample new parameters for self.network from the SWAG approximate posterior
            self.sample_parameters()

            # TODO(1): Perform inference for all samples in `loader` using current model sample,
            #  and add the predictions to model_predictions
            batch_predictions = []
            for (batch_images,) in loader:
                # Get model predictions (logits)
                logits = self.network(batch_images)
                # Convert to probabilities using temperature scaling
                probabilities = self._apply_temperature_scaling(logits, temperature=self._temperature)
                batch_predictions.append(probabilities)

            # Concatenate all batch predictions for this model sample
            sample_predictions = torch.cat(batch_predictions, dim=0)
            model_predictions.append(sample_predictions)

        assert len(model_predictions) == self.num_bma_samples
        assert all(
            isinstance(sample_predictions, torch.Tensor)
            and sample_predictions.dim() == 2  # N x C
            and sample_predictions.size(1) == 6
            for sample_predictions in model_predictions
        )

        # TODO(1): Average predictions from different model samples into bma_probabilities
        # Stack all predictions: shape will be (num_samples, N, C)
        stacked_predictions = torch.stack(model_predictions, dim=0)
        # Average across model samples (dim=0) to get final BMA probabilities
        bma_probabilities = torch.mean(stacked_predictions, dim=0)

        assert bma_probabilities.dim() == 2 and bma_probabilities.size(1) == 6  # N x C
        return bma_probabilities

    def sample_parameters(self) -> None:
        """
        Sample a new network from the approximate SWAG posterior.
        For simplicity, this method directly modifies self.network in-place.
        Hence, after calling this method, self.network corresponds to a new posterior sample.
        """

        # Instead of acting on a full vector of parameters, all operations can be done on per-layer parameters.
        for name, param in self.network.named_parameters():
            # SWAG-diagonal part
            z_diag = torch.randn(param.size())
            # TODO(1): Sample parameter values for SWAG-diagonal
            mean_weights = self.swag_mean_weights[name]
            # std_weights = sqrt(θ² - θ²_SWA) where θ²_SWA = (θ_SWA)²
            variance = self.swag_second_moment[name] - (self.swag_mean_weights[name] ** 2)
            # Ensure variance is non-negative (numerical stability)
            variance = torch.clamp(variance, min=1e-8)
            std_weights = torch.sqrt(variance)
            assert mean_weights.size() == param.size() and std_weights.size() == param.size()

            # Diagonal part: θ_sample = θ_SWA + (1/√2)·σ_diag·z
            sampled_weight = mean_weights + (1.0 / math.sqrt(2.0)) * std_weights * z_diag

            # Full SWAG part
            if self.inference_mode == InferenceMode.SWAG_FULL:
                # TODO(2): Sample parameter values for full SWAG
                K = len(self.deviation_matrix_buffer[name])
                if K > 0:
                    # Sample z2 ~ N(0, I_K)
                    z2 = torch.randn(K)
                    # Calculate low-rank component: (1/√(2(K-1)))·D·z2
                    low_rank_component = torch.zeros_like(param)
                    for i, deviation in enumerate(self.deviation_matrix_buffer[name]):
                        low_rank_component += deviation * z2[i]
                    if K > 1:
                        low_rank_component = low_rank_component / math.sqrt(2.0 * (K - 1))
                    else:
                        low_rank_component = low_rank_component / math.sqrt(2.0)
                    sampled_weight += low_rank_component

            # Modify weight value in-place; directly changing self.network
            param.data = sampled_weight

        # TODO(1): Don't forget to update batch normalization statistics using self._update_batchnorm_statistics()
        #  in the appropriate place!
        # Update batch normalization statistics for the newly sampled network
        self._update_batchnorm_statistics()

    def label_prediction(self, predicted_probabilities: torch.Tensor) -> torch.Tensor:
        """
        Predict labels in {0, 1, 2, 3, 4, 5} or "don't know" as -1
        based on your model's predicted probabilities.
        The parameter predicted_probabilities is an Nx6 tensor containing predicted probabilities
        as returned by predict_probs(...).
        The output should be a N-dimensional long tensor, containing values in {-1, 0, 1, 2, 3, 4, 5}.
        """

        # label_probabilities contains the per-row maximum values in predicted_probabilities,
        # max_likelihood_labels the corresponding column index (equivalent to class).
        label_probabilities, max_likelihood_labels = torch.max(predicted_probabilities, dim=-1)
        num_samples = predicted_probabilities.size(0)

        if self.inference_mode == InferenceMode.MAP:
            return max_likelihood_labels

        # Calculate multiple uncertainty metrics

        # 1. Max probability (confidence)
        confidence = label_probabilities

        # 2. Entropy (distributional uncertainty)
        epsilon = 1e-10
        entropy = -torch.sum(
            predicted_probabilities * torch.log(predicted_probabilities + epsilon),
            dim=-1
        )
        max_entropy = math.log(6)
        normalized_entropy = entropy / max_entropy

        # 3. Margin (difference between top-2 predictions)
        sorted_probs, _ = torch.sort(predicted_probabilities, dim=-1, descending=True)
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]  # Higher margin = more certain

        # Combined uncertainty score (0 = certain, 1 = uncertain)
        uncertainty = (
            0.5 * (1 - confidence) +           # Low confidence → uncertain
            0.3 * normalized_entropy +          # High entropy → uncertain
            0.2 * (1 - margin)                  # Low margin → uncertain
        )

        # Decision rule with adaptive threshold
        # For very certain predictions (low uncertainty), use lower threshold
        # For borderline predictions, use calibration threshold
        certainty_bonus = 0.15  # Allow some confident predictions through
        effective_threshold = torch.clamp(
            self._calibration_threshold - certainty_bonus * (1 - uncertainty),
            min=0.3,  # Never go below 0.3
            max=0.95  # Never require > 0.95 confidence
        )

        return torch.where(
            confidence >= effective_threshold,
            max_likelihood_labels,
            torch.ones_like(max_likelihood_labels) * -1,
        )

    def _create_weight_copy(self) -> typing.Dict[str, torch.Tensor]:
        """Create an all-zero copy of the network weights as a dictionary that maps name -> weight"""
        return {
            name: torch.zeros_like(param, requires_grad=False)
            for name, param in self.network.named_parameters()
        }

    def train_model(
        self,
        loader: torch.utils.data.DataLoader,
    ) -> None:
        """
        Perform full SWAG fitting procedure.
        If `PRETRAINED_WEIGHTS_FILE` is `True`, this method skips the MAP inference part,
        and uses pretrained weights instead.

        Note that MAP inference can take a very long time.
        You should hence only perform MAP inference yourself after passing the baseline
        using the given CNN architecture and pretrained weights.
        """

        # MAP inference to obtain initial weights
        PRETRAINED_WEIGHTS_FILE = self.model_dir / "map_weights.pt"
        if USE_PRETRAINED_WEIGHTS:
            self.network.load_state_dict(torch.load(PRETRAINED_WEIGHTS_FILE))
            print("Loaded pretrained MAP weights from", PRETRAINED_WEIGHTS_FILE)
        else:
            self.fit_map_model(loader)

        # SWAG
        if self.inference_mode in (InferenceMode.SWAG_DIAGONAL, InferenceMode.SWAG_FULL):
            self.fit_swag_model(loader)

    def fit_map_model(self, loader: torch.utils.data.DataLoader) -> None:
        """
        MAP inference procedure to obtain initial weights of self.network.
        This is the exact procedure that was used to obtain the pretrained weights we provide.
        """
        map_training_epochs = 140
        initial_learning_rate = 0.01
        reduced_learning_rate = 0.0001
        start_decay_epoch = 50
        decay_factor = reduced_learning_rate / initial_learning_rate

        # Create optimizer, loss, and a learning rate scheduler that aids convergence
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=initial_learning_rate,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )
        loss_fn = torch.nn.CrossEntropyLoss(
            reduction="mean",
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0),
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=decay_factor,
                    total_iters=(map_training_epochs - start_decay_epoch) * len(loader),
                ),
            ],
            milestones=[start_decay_epoch * len(loader)],
        )

        # Put network into training mode
        # Batch normalization layers are only updated if the network is in training mode,
        # and are replaced by a moving average if the network is in evaluation mode.
        self.network.train()
        with tqdm.trange(map_training_epochs, desc="Fitting initial MAP weights") as pbar:
            progress_dict = {}
            # Perform the specified number of MAP epochs
            for epoch in pbar:
                avg_loss = 0.0
                avg_accuracy = 0.0
                num_samples = 0
                # Iterate over batches of randomly shuffled training data
                for batch_images, _, _, batch_labels in loader:
                    # Training step
                    optimizer.zero_grad()
                    predictions = self.network(batch_images)
                    batch_loss = loss_fn(input=predictions, target=batch_labels)
                    batch_loss.backward()
                    optimizer.step()

                    # Save learning rate that was used for step, and calculate new one
                    progress_dict["lr"] = lr_scheduler.get_last_lr()[0]
                    with warnings.catch_warnings():
                        # Suppress annoying warning (that we cannot control) inside PyTorch
                        warnings.simplefilter("ignore")
                        lr_scheduler.step()

                    # Calculate cumulative average training loss and accuracy
                    avg_loss = (batch_images.size(0) * batch_loss.item() + num_samples * avg_loss) / (
                        num_samples + batch_images.size(0)
                    )
                    avg_accuracy = (
                        torch.sum(predictions.argmax(dim=-1) == batch_labels).item()
                        + num_samples * avg_accuracy
                    ) / (num_samples + batch_images.size(0))
                    num_samples += batch_images.size(0)

                    progress_dict["avg. epoch loss"] = avg_loss
                    progress_dict["avg. epoch accuracy"] = avg_accuracy
                    pbar.set_postfix(progress_dict)

    def predict_probs(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities for the given images xs.
        This method returns an NxC float tensor,
        where row i column j corresponds to the probability that y_i is class j.

        This method uses different strategies depending on self.inference_mode.
        """
        self.network = self.network.eval()

        # Create a loader that we can deterministically iterate many times if necessary
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(xs),
            batch_size=32,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        with torch.no_grad():  # save memory by not tracking gradients
            if self.inference_mode == InferenceMode.MAP:
                return self.predict_probabilities_map(loader)
            else:
                return self.predict_probabilities_swag(loader)

    def predict_probabilities_map(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Predict probabilities assuming that self.network is a MAP estimate.
        This simply performs a forward pass for every batch in `loader`,
        concatenates all results, and applies a row-wise softmax.
        """
        all_predictions = []
        for (batch_images,) in loader:
            all_predictions.append(self.network(batch_images))

        all_predictions = torch.cat(all_predictions)
        return torch.softmax(all_predictions, dim=-1)

    def _update_batchnorm_statistics(self) -> None:
        """
        Reset and fit batch normalization statistics using the training dataset self.training_dataset.
        We provide this method for you for convenience.
        See the SWAG paper for why this is required.

        Batch normalization usually uses an exponential moving average, controlled by the `momentum` parameter.
        However, we are not training but want the statistics for the full training dataset.
        Hence, setting `momentum` to `None` tracks a cumulative average instead.
        The following code stores original `momentum` values, sets all to `None`,
        and restores the previous hyperparameters after updating batchnorm statistics.
        """

        original_momentum_values = dict()
        for module in self.network.modules():
            # Only need to handle batchnorm modules
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                continue

            # Store old momentum value before removing it
            original_momentum_values[module] = module.momentum
            module.momentum = None

            # Reset batch normalization statistics
            module.reset_running_stats()

        loader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        self.network.train()
        for (batch_images,) in loader:
            self.network(batch_images)
        self.network.eval()

        # Restore old `momentum` hyperparameter values
        for module, momentum in original_momentum_values.items():
            module.momentum = momentum


class SWAGScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Custom learning rate scheduler that calculates a different learning rate each gradient descent step.
    The default implementation keeps the original learning rate constant, i.e., does nothing.
    You can implement a custom schedule inside calculate_lr,
    and add+store additional attributes in __init__.
    You should not change any other parts of this class.
    """

    def calculate_lr(self, current_epoch: float, previous_lr: float) -> float:
        """
        Calculate the learning rate for the epoch given by current_epoch.
        current_epoch is the fractional epoch of SWA fitting, starting at 0.
        That is, an integer value x indicates the start of epoch (x+1),
        and non-integer values x.y correspond to steps in between epochs (x+1) and (x+2).
        previous_lr is the previous learning rate.

        This method should return a single float: the new learning rate.
        """
        # TODO(2): Implement a custom schedule if desired
        return previous_lr

    # TODO(2): Add and store additional arguments if you decide to implement a custom scheduler
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        steps_per_epoch: int,
    ):
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        super().__init__(optimizer, last_epoch=-1, verbose=False)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )
        return [
            self.calculate_lr(self.last_epoch / self.steps_per_epoch, group["lr"])
            for group in self.optimizer.param_groups
        ]


def run_evaluation(
    swag_inference: SWAInferenceHandler,
    eval_dataset: torch.utils.data.Dataset,
    extended_evaluation: bool,
    output_location: pathlib.Path,
) -> None:
    """
    Run evaluation with your model.
    Feel free to change or extend this code.
    :param swag_inference: Trained model to evaluate
    :param eval_dataset: Validation dataset
    :param: extended_evaluation: If True, generates additional plots
    :param output_location: Directory into which extended evaluation plots are saved
    """

    print("Evaluating model on validation data")

    # We ignore is_snow and is_cloud here, but feel free to use them as well
    images, snow_labels, cloud_labels, labels = eval_dataset.tensors

    # Predict class probabilities on test data,
    # most likely classes (according to the max predicted probability),
    # and classes as predicted by your SWAG implementation.
    all_pred_probabilities = swag_inference.predict_probs(images)
    max_pred_probabilities, argmax_pred_labels = torch.max(all_pred_probabilities, dim=-1)
    predicted_labels = swag_inference.label_prediction(all_pred_probabilities)

    # Create a mask that ignores ambiguous samples (those with class -1)
    non_ambiguous_mask = labels != -1

    # Calculate three kinds of accuracy:
    # 1. Overall accuracy, counting "don't know" (-1) as its own class
    # 2. Accuracy on all samples that have a known label. Predicting -1 on those counts as wrong here.
    # 3. Accuracy on all samples that have a known label w.r.t. the class with the highest predicted probability.
    overall_accuracy = torch.mean((predicted_labels == labels).float()).item()
    non_ambiguous_accuracy = torch.mean((predicted_labels[non_ambiguous_mask] == labels[non_ambiguous_mask]).float()).item()
    non_ambiguous_argmax_accuracy = torch.mean(
        (argmax_pred_labels[non_ambiguous_mask] == labels[non_ambiguous_mask]).float()
    ).item()
    print(f"Accuracy (raw): {overall_accuracy:.4f}")
    print(f"Accuracy (non-ambiguous only, your predictions): {non_ambiguous_accuracy:.4f}")
    print(f"Accuracy (non-ambiguous only, predicting most-likely class): {non_ambiguous_argmax_accuracy:.4f}")

    # Determine which threshold would yield the smallest cost on the validation data
    # Note that this threshold does not necessarily generalize to the test set!
    # However, it can help you judge your method's calibration.
    threshold_values = [0.0] + list(torch.unique(max_pred_probabilities, sorted=True))
    costs = []
    for threshold in threshold_values:
        thresholded_predictions = torch.where(max_pred_probabilities <= threshold, -1 * torch.ones_like(predicted_labels), predicted_labels)
        costs.append(compute_cost(thresholded_predictions, labels).item())
    best_threshold_index = np.argmin(costs)
    print(f"Best cost {costs[best_threshold_index]} at threshold {threshold_values[best_threshold_index]}")
    print("Note that this threshold does not necessarily generalize to the test set!")

    # Calculate ECE and plot the calibration curve
    calibration_data = calculate_calibration_curve(all_pred_probabilities.numpy(), labels.numpy(), num_bins=20)
    print("Validation ECE:", calibration_data["ece"])

    if extended_evaluation:
        print("Plotting reliability diagram")
        fig = paint_reliability_diagram(calibration_data)
        fig.savefig(output_location / "reliability_diagram.pdf")

        sorted_confidence_indices = torch.argsort(max_pred_probabilities)

        # Plot samples your model is most confident about
        print("Plotting most confident validation set predictions")
        most_confident_indices = sorted_confidence_indices[-10:]
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_index = most_confident_indices[5 * row // 2 + col]
                ax[row, col].imshow(images[sample_index].permute(1, 2, 0).numpy())
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f"pred. {predicted_labels[sample_index]}, true {labels[sample_index]}")
                bar_colors = ["C0"] * 6
                if labels[sample_index] >= 0:
                    bar_colors[labels[sample_index]] = "C1"
                ax[row + 1, col].bar(
                    np.arange(6), all_pred_probabilities[sample_index].numpy(), tick_label=np.arange(6), color=bar_colors
                )
        fig.suptitle("Most confident predictions", size=20)
        fig.savefig(output_location / "examples_most_confident.pdf")

        # Plot samples your model is least confident about
        print("Plotting least confident validation set predictions")
        least_confident_indices = sorted_confidence_indices[:10]
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_index = least_confident_indices[5 * row // 2 + col]
                ax[row, col].imshow(images[sample_index].permute(1, 2, 0).numpy())
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f"pred. {predicted_labels[sample_index]}, true {labels[sample_index]}")
                bar_colors = ["C0"] * 6
                if labels[sample_index] >= 0:
                    bar_colors[labels[sample_index]] = "C1"
                ax[row + 1, col].bar(
                    np.arange(6), all_pred_probabilities[sample_index].numpy(), tick_label=np.arange(6), color=bar_colors
                )
        fig.suptitle("Least confident predictions", size=20)
        fig.savefig(output_location / "examples_least_confident.pdf")


class CNN(torch.nn.Module):
    """
    Small convolutional neural network used in this task.
    You should not modify this class before passing the baseline.

    Note that if you change the architecture of this network,
    you need to re-run MAP inference and cannot use the provided pretrained weights anymore.
    Hence, you need to set `USE_PRETRAINED_INIT = False` at the top of this file.
    """
    def __init__(
        self,
        in_channels: int,
        out_classes: int,
    ):
        super().__init__()

        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=5),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.pool1 = torch.nn.MaxPool2d((2, 2), stride=(2, 2))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.pool2 = torch.nn.MaxPool2d((2, 2), stride=(2, 2))

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
        )

        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.linear = torch.nn.Linear(64, out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool1(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool2(x)
        x = self.layer5(x)

        # Average features over both spatial dimensions, and remove the now superfluous dimensions
        x = self.global_pool(x).squeeze(-1).squeeze(-1)

        log_softmax = self.linear(x)

        return log_softmax


if __name__ == "__main__":
    main()
