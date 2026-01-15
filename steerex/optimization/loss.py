"""Loss components for steering optimization."""

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from steerex.backends.base import ModelBackend
    from steerex.steering.base import SteeringMode


class LossComponent(ABC):
    """
    Abstract base class for loss components.

    Loss components compute a scalar loss from model logits
    and target tokens. They can be composed to create complex
    loss functions.
    """

    @abstractmethod
    def compute(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        prompt_len: int,
        coldness: float = 1.0,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """
        Compute loss contribution.

        Args:
            logits: Model output logits (seq_len, vocab_size).
            target_ids: Full sequence token IDs (seq_len,).
            prompt_len: Number of prompt tokens (loss only on rest).
            coldness: Inverse temperature for softmax.
            eps: Small constant for numerical stability.

        Returns:
            Scalar loss tensor.
        """
        ...


class PromotionLoss(LossComponent):
    """
    Increase probability of target tokens.

    Loss = -sum(log P(target_token))

    Used for dst_completions (things we want the model to say).
    """

    def __init__(self, normalize_by_length: bool = False, eps: float = 1e-6):
        """
        Args:
            normalize_by_length: If True, divide by completion length.
            eps: Small constant for numerical stability.
        """
        self.normalize_by_length = normalize_by_length
        self.eps = eps

    def compute(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        prompt_len: int,
        coldness: float = 1.0,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """Compute negative log probability of target tokens (vectorized).

        Note: Uses self.eps (set at initialization) for numerical stability,
        ignoring the eps parameter which exists for interface compatibility.
        """
        total_len = len(target_ids)
        completion_len = total_len - prompt_len

        # Handle empty completion
        if completion_len <= 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        probs = torch.softmax(logits * coldness, dim=-1)

        # Get completion token IDs (tokens we're predicting)
        completion_ids = target_ids[prompt_len:total_len]

        # Get probabilities for those tokens from the previous position's logits
        # logits[i-1] predicts token[i], so we need logits[prompt_len-1:total_len-1]
        completion_logit_probs = probs[prompt_len - 1 : total_len - 1]

        # Gather the probabilities for the actual target tokens
        # completion_ids shape: [completion_len]
        # completion_logit_probs shape: [completion_len, vocab_size]
        target_probs = completion_logit_probs.gather(
            dim=-1, index=completion_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Compute negative log probability sum (vectorized)
        loss = -torch.log(target_probs + self.eps).sum()

        if self.normalize_by_length:
            loss = loss / completion_len

        return loss


class SuppressionLoss(LossComponent):
    """
    Decrease probability of target tokens.

    If use_one_minus=True:
        Loss = -sum(log(1 - P(target_token)))
    Else:
        Loss = sum(log P(target_token))

    Used for src_completions (things we want the model NOT to say).
    """

    def __init__(
        self,
        use_one_minus: bool = True,
        normalize_by_length: bool = False,
        eps: float = 1e-6,
        max_prob: float = 0.9999,
    ):
        """
        Args:
            use_one_minus: Use log(1-p) vs -log(p).
            normalize_by_length: If True, divide by completion length.
            eps: Small constant for numerical stability.
            max_prob: Clamp probability to this max to prevent log(0) in use_one_minus mode.
        """
        self.use_one_minus = use_one_minus
        self.normalize_by_length = normalize_by_length
        self.eps = eps
        self.max_prob = max_prob

    def compute(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        prompt_len: int,
        coldness: float = 1.0,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """Compute suppression loss (vectorized).

        Note: Uses self.eps (set at initialization) for numerical stability,
        ignoring the eps parameter which exists for interface compatibility.
        """
        total_len = len(target_ids)
        completion_len = total_len - prompt_len

        # Handle empty completion
        if completion_len <= 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        probs = torch.softmax(logits * coldness, dim=-1)

        # Get completion token IDs (tokens we're predicting)
        completion_ids = target_ids[prompt_len:total_len]

        # Get probabilities for those tokens from the previous position's logits
        completion_logit_probs = probs[prompt_len - 1 : total_len - 1]

        # Gather the probabilities for the actual target tokens
        target_probs = completion_logit_probs.gather(
            dim=-1, index=completion_ids.unsqueeze(-1)
        ).squeeze(-1)

        if self.use_one_minus:
            # Clamp prob to prevent log(0) when prob→1
            # This bounds the gradient to 1/(1-max_prob) ≈ 10000 instead of infinity
            target_probs_clamped = torch.clamp(target_probs, max=self.max_prob)
            loss = -torch.log(1 - target_probs_clamped + self.eps).sum()
        else:
            loss = torch.log(target_probs + self.eps).sum()

        if self.normalize_by_length:
            loss = loss / completion_len

        return loss


class SatisficingLoss(LossComponent):
    """
    Penalize squared difference from target loss.

    Loss = (actual_loss - target_loss)^2

    Used when you want to achieve a specific loss value,
    not just minimize/maximize.
    """

    def __init__(
        self,
        base_loss: LossComponent,
        target_loss: float,
    ):
        """
        Args:
            base_loss: The underlying loss component.
            target_loss: The target value to achieve.
        """
        self.base_loss = base_loss
        self.target_loss = target_loss

    def compute(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        prompt_len: int,
        coldness: float = 1.0,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """Compute squared difference from target."""
        actual_loss = self.base_loss.compute(
            logits, target_ids, prompt_len, coldness, eps
        )
        return (actual_loss - self.target_loss) ** 2


class CompositeLoss(LossComponent):
    """
    Combine multiple loss components.

    Loss = sum(component.compute(...) for component in components)
    """

    def __init__(self, *components: LossComponent):
        """
        Args:
            *components: Loss components to combine.
        """
        self.components = list(components)

    def add(self, component: LossComponent) -> "CompositeLoss":
        """Add a component and return self for chaining."""
        self.components.append(component)
        return self

    def compute(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        prompt_len: int,
        coldness: float = 1.0,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """Compute sum of all component losses."""
        total = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        for component in self.components:
            total = total + component.compute(
                logits, target_ids, prompt_len, coldness, eps
            )
        return total


class WeightedLoss(LossComponent):
    """
    Apply a weight to a loss component.

    Loss = weight * base_loss.compute(...)
    """

    def __init__(self, base_loss: LossComponent, weight: float):
        """
        Args:
            base_loss: The underlying loss component.
            weight: Multiplier for the loss.
        """
        self.base_loss = base_loss
        self.weight = weight

    def compute(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        prompt_len: int,
        coldness: float = 1.0,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """Compute weighted loss."""
        return self.weight * self.base_loss.compute(
            logits, target_ids, prompt_len, coldness, eps
        )


# =============================================================================
# Regularization Components
# =============================================================================


class RegularizerComponent(ABC):
    """
    Abstract base class for regularization losses.

    Unlike LossComponent which operates on model logits,
    RegularizerComponent operates directly on steering parameters
    to add regularization terms to the optimization objective.
    """

    @abstractmethod
    def compute(self, steering_mode: "SteeringMode") -> torch.Tensor:
        """
        Compute regularization loss from steering parameters.

        Args:
            steering_mode: The steering mode containing the vector to regularize.

        Returns:
            Scalar loss tensor.
        """
        ...


class ManifoldLoss(RegularizerComponent):
    """
    Regularize steering vector to stay on the manifold of natural activations.

    Uses PCA to capture the principal subspace of natural model activations,
    then penalizes the component of the steering vector that lies outside
    this subspace (reconstruction error).

    L_manifold = ||v - v_recon||^2

    where v_recon = PCA.inverse_transform(PCA.transform(v))

    This encourages steering vectors that look like natural activation
    differences rather than arbitrary perturbations ("wormholes").

    Example:
        >>> # Initialize with PCA from natural activations
        >>> manifold_loss = ManifoldLoss.from_alpaca(
        ...     backend=backend,
        ...     tokenizer=tokenizer,
        ...     layer=16,
        ...     n_samples=100,
        ...     num_activations_per_sample=10,
        ...     explained_variance_threshold=0.95,
        ... )
        >>>
        >>> # Use in optimizer
        >>> optimizer = SteeringOptimizer(
        ...     backend, steering, config,
        ...     regularizer=manifold_loss,
        ...     regularizer_weight=0.1,
        ... )
    """

    def __init__(
        self,
        pca_components: torch.Tensor,
        pca_mean: torch.Tensor,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        n_components: Optional[int] = None,
        explained_variance_ratio: Optional[float] = None,
    ):
        """
        Initialize ManifoldLoss with pre-computed PCA components.

        Args:
            pca_components: PCA component matrix [n_components, hidden_dim].
            pca_mean: Mean vector used in PCA [hidden_dim].
            device: Device for computation.
            dtype: Data type for tensors.
            n_components: Number of PCA components kept (for info).
            explained_variance_ratio: Total variance explained (for info).
        """
        self.pca_components = pca_components.to(device=device, dtype=dtype)
        self.pca_mean = pca_mean.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.n_components = n_components or pca_components.shape[0]
        self.explained_variance_ratio = explained_variance_ratio

    @classmethod
    def from_activations(
        cls,
        activations: torch.Tensor,
        explained_variance_threshold: float = 0.95,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> "ManifoldLoss":
        """
        Create ManifoldLoss by fitting PCA to activation matrix.

        Args:
            activations: Activation matrix [n_samples, hidden_dim].
            explained_variance_threshold: Keep components explaining this
                fraction of variance (e.g., 0.95 for 95%).
            device: Device for the loss computation.
            dtype: Data type for tensors.

        Returns:
            Configured ManifoldLoss instance.
        """
        import numpy as np
        from sklearn.decomposition import PCA

        # Convert to numpy for sklearn PCA
        X = activations.cpu().float().numpy()

        # Fit full PCA first to determine n_components
        pca_full = PCA()
        pca_full.fit(X)

        # Find number of components for threshold
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumsum, explained_variance_threshold) + 1)
        n_components = min(n_components, X.shape[1], X.shape[0])

        # Refit with optimal n_components
        pca = PCA(n_components=n_components)
        pca.fit(X)

        # Extract components and mean
        pca_components = torch.from_numpy(pca.components_)  # [n_components, hidden_dim]
        pca_mean = torch.from_numpy(pca.mean_)  # [hidden_dim]
        explained_variance_ratio = float(cumsum[n_components - 1])

        return cls(
            pca_components=pca_components,
            pca_mean=pca_mean,
            device=device,
            dtype=dtype,
            n_components=n_components,
            explained_variance_ratio=explained_variance_ratio,
        )

    @classmethod
    def from_alpaca(
        cls,
        backend: "ModelBackend",
        tokenizer,
        layer: int,
        n_samples: int = 100,
        num_activations_per_sample: int = 10,
        explained_variance_threshold: float = 0.95,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> "ManifoldLoss":
        """
        Create ManifoldLoss by extracting activations from Alpaca dataset.

        This is the main constructor for typical usage. It:
        1. Loads n_samples from the Alpaca dataset
        2. Formats each as chat (instruction + input -> output)
        3. Runs through the model and extracts activations at layer
        4. Randomly samples num_activations_per_sample from Assistant response tokens
        5. Fits PCA to the collected activations

        Args:
            backend: Model backend for forward passes.
            tokenizer: Tokenizer with chat template support.
            layer: Layer index to extract activations from.
            n_samples: Number of Alpaca samples to use.
            num_activations_per_sample: Number of activation vectors to
                sample per Alpaca example (from Assistant response tokens).
            explained_variance_threshold: Variance threshold for PCA.
            seed: Random seed for reproducibility.
            cache_dir: Optional cache directory for dataset.
            verbose: Whether to show progress.

        Returns:
            Configured ManifoldLoss instance.
        """
        import random

        import numpy as np
        from datasets import load_dataset

        random.seed(seed)
        np.random.seed(seed)

        # Load Alpaca dataset
        dataset = load_dataset("tatsu-lab/alpaca", split="train", cache_dir=cache_dir)
        if n_samples < len(dataset):
            indices = random.sample(range(len(dataset)), n_samples)
        else:
            indices = list(range(len(dataset)))

        all_activations = []

        if verbose:
            try:
                from tqdm import tqdm

                iterator = tqdm(indices, desc="Extracting activations for PCA")
            except ImportError:
                iterator = indices
        else:
            iterator = indices

        for idx in iterator:
            row = dataset[idx]

            # Format as chat conversation with response
            messages = _format_alpaca_as_chat(row)

            # Apply chat template
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Find where assistant response starts
            prompt_messages = messages[:-1] + [{"role": "assistant", "content": ""}]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            prompt_ids = backend.tokenize(prompt_text)
            full_ids = backend.tokenize(full_text)

            prompt_len = prompt_ids.shape[1]
            full_len = full_ids.shape[1]

            # Skip if no response tokens
            response_len = full_len - prompt_len
            if response_len <= 0:
                continue

            # Extract activations at layer
            activations = _extract_layer_activations(backend, full_ids, layer)

            # Sample from response tokens only
            response_activations = activations[prompt_len:]

            # Randomly sample num_activations_per_sample
            n_to_sample = min(num_activations_per_sample, response_activations.shape[0])
            if n_to_sample <= 0:
                continue

            sample_indices = random.sample(range(response_activations.shape[0]), n_to_sample)

            for i in sample_indices:
                all_activations.append(response_activations[i].cpu())

        if len(all_activations) == 0:
            raise ValueError("No activations collected. Check dataset and model.")

        # Stack into matrix
        activation_matrix = torch.stack(all_activations)

        if verbose:
            print(
                f"Collected {activation_matrix.shape[0]} activations "
                f"with dim {activation_matrix.shape[1]}"
            )

        return cls.from_activations(
            activations=activation_matrix,
            explained_variance_threshold=explained_variance_threshold,
            device=backend.get_device(),
            dtype=backend.get_dtype(),
        )

    def compute(self, steering_mode: "SteeringMode") -> torch.Tensor:
        """
        Compute manifold regularization loss.

        Projects the steering vector to PCA subspace and back,
        then computes squared reconstruction error.

        Args:
            steering_mode: Steering mode containing the vector.

        Returns:
            Scalar reconstruction error loss.
        """
        # Get the steering vector (need gradient to flow through)
        # Use parameters() to get the actual tensor, not detached copy
        params = steering_mode.parameters()
        if not params:
            raise ValueError("Steering mode has no parameters")

        # First parameter is the vector for VectorSteering
        vector = params[0]
        v = vector.to(device=self.device, dtype=self.dtype)

        # Project to PCA space and back
        v_centered = v - self.pca_mean
        v_proj = v_centered @ self.pca_components.T  # [n_components]
        v_recon_centered = v_proj @ self.pca_components  # [hidden_dim]
        v_recon = v_recon_centered + self.pca_mean

        # Reconstruction error
        error = v - v_recon
        loss = (error**2).sum()

        return loss

    def get_reconstruction(self, vector: torch.Tensor) -> torch.Tensor:
        """
        Get the reconstructed vector (for analysis).

        Args:
            vector: Input vector [hidden_dim].

        Returns:
            Reconstructed vector [hidden_dim].
        """
        v = vector.to(device=self.device, dtype=self.dtype)
        v_centered = v - self.pca_mean
        v_proj = v_centered @ self.pca_components.T
        v_recon_centered = v_proj @ self.pca_components
        return v_recon_centered + self.pca_mean

    def get_reconstruction_error(self, vector: torch.Tensor) -> float:
        """
        Get reconstruction error for a vector (for analysis).

        Args:
            vector: Input vector [hidden_dim].

        Returns:
            Reconstruction error as scalar.
        """
        v_recon = self.get_reconstruction(vector)
        return float(((vector - v_recon) ** 2).sum().item())


# =============================================================================
# Helper Functions
# =============================================================================


def _format_alpaca_as_chat(row: dict) -> list:
    """Format Alpaca row as chat messages including response."""
    user_content = row["instruction"]
    if row.get("input"):
        user_content += f"\n\nInput: {row['input']}"

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": row["output"]},
    ]


def _extract_layer_activations(
    backend: "ModelBackend",
    input_ids: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """Extract activations at a specific layer."""
    captured = []

    def capture_hook(module, args):
        hidden_states = args[0]
        captured.append(hidden_states.detach().clone())
        return args

    with backend.hooks_context([(layer, capture_hook)]):
        _ = backend.get_logits(input_ids)

    return captured[0].squeeze(0)  # [seq_len, hidden_dim]
