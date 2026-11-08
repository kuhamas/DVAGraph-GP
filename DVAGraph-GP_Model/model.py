"""Mathematical implementation of DVAGraph-GP.

This module implements the architecture specified by the current manuscript:

    DCF-VAE encoder:
        D -> 256 -> MaxPool(2) -> three parallel Conv1D branches
        (dilation 2, 4, 6; 64 channels each) -> 1x1 fusion (192 -> 32)
        -> flatten (4096) -> (mu, logvar), each 64-dimensional.

    DCF-VAE decoder (training only):
        64 -> 512 -> 512 -> D, GELU after the first two layers and a
        linear output layer.

    DAGSC:
        a sparse graph over the 64 latent dimensions, built from column-wise
        cosine similarities, ELU, off-diagonal row softmax, row-wise top-8,
        max-symmetrization, and unit self-loops; followed by degree-2
        Chebyshev filtering with three trainable scalar coefficients.

    Regression:
        global mean pooling over the 64 graph-enhanced latent dimensions,
        followed by one Linear(1, 1) layer.

Training uses a sampled latent Z for reconstruction and the dynamic minibatch
prediction graph. Evaluation uses posterior means and a graph frozen from the
relevant training partition. No data preprocessing or cross-validation logic is
implemented here; those remain responsibilities of the training pipeline.

PyTorch defaults are retained where the manuscript does not specify parameter
initialization or bias flags: Linear/Conv1D layers use biases, LayerNorm uses
its learnable affine parameters, and module weights use PyTorch's default
initializers. Chebyshev coefficients are initialized to one. Top-k ties are
resolved stably by ascending latent-node index.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Fixed architectural quantities specified by the manuscript
# ---------------------------------------------------------------------------
LATENT_DIM = 64
INITIAL_EMBEDDING_DIM = 256
POOLED_LENGTH = 128
BRANCH_CHANNELS = 64
BRANCH_KERNEL_SIZE = 3
DILATION_RATES = (2, 4, 6)
FUSION_CHANNELS = 32
DROPOUT_RATE = 0.20
DECODER_HIDDEN_DIM = 512
GRAPH_TOP_K = 8
CHEBYSHEV_DEGREE = 2  # R = 2; terms T_0, T_1, and T_2
N_CHEBYSHEV_TERMS = CHEBYSHEV_DEGREE + 1
NORM_EPS = 1e-8
LAMBDA_PRED = 1.0
LAMBDA_RECON = 0.25
LAMBDA_KL = 0.001



class DCFVAE(nn.Module):
    """Dilated convolutional fusion variational autoencoder."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if int(input_dim) < 1:
            raise ValueError("input_dim (D) must be a positive integer.")

        self.input_dim = int(input_dim)

        # Eq. (1): D -> 256, followed by reshape and MaxPool(2, stride=2).
        self.initial_projection = nn.Linear(self.input_dim, INITIAL_EMBEDDING_DIM)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Eqs. (2)-(3): three parallel same-length dilated Conv1D branches.
        # For kernel=3, padding=dilation preserves the 128-position length.
        self.dilated_branches = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=1,
                    out_channels=BRANCH_CHANNELS,
                    kernel_size=BRANCH_KERNEL_SIZE,
                    dilation=dilation,
                    padding=dilation,
                    bias=True,
                )
                for dilation in DILATION_RATES
            ]
        )
        # LayerNorm is applied over channels at each position independently.
        self.branch_norms = nn.ModuleList(
            [nn.LayerNorm(BRANCH_CHANNELS) for _ in DILATION_RATES]
        )

        # Eq. (4): concatenate 3 x 64 channels, then 1x1 convolution 192 -> 32.
        self.fusion_projection = nn.Conv1d(
            in_channels=BRANCH_CHANNELS * len(DILATION_RATES),
            out_channels=FUSION_CHANNELS,
            kernel_size=1,
            bias=True,
        )
        self.fusion_dropout = nn.Dropout(p=DROPOUT_RATE)

        # Eq. (5): flatten 32 x 128 = 4096; independent posterior projections.
        self.fc_mu = nn.Linear(FUSION_CHANNELS * POOLED_LENGTH, LATENT_DIM)
        self.fc_logvar = nn.Linear(FUSION_CHANNELS * POOLED_LENGTH, LATENT_DIM)

        # Eqs. (10)-(12): approved three-layer decoder, linear final output.
        self.decoder_fc1 = nn.Linear(LATENT_DIM, DECODER_HIDDEN_DIM)
        self.decoder_fc2 = nn.Linear(DECODER_HIDDEN_DIM, DECODER_HIDDEN_DIM)
        self.decoder_fc3 = nn.Linear(DECODER_HIDDEN_DIM, self.input_dim)

    def _encode_features(
        self, x: torch.Tensor, return_details: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if x.ndim != 2:
            raise ValueError(f"Expected genotype input [B, D], received {tuple(x.shape)}.")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Input marker dimension mismatch: expected D={self.input_dim}, "
                f"received D={x.shape[1]}."
            )

        # Eq. (1): learned 256-dimensional embedding, then [B, 1, 256].
        initial = F.gelu(self.initial_projection(x))
        pooled = self.max_pool(initial.unsqueeze(1))
        if pooled.shape[1:] != (1, POOLED_LENGTH):
            raise RuntimeError(
                "Initial projection/pooling must produce [B, 1, 128]; "
                f"received {tuple(pooled.shape)}."
            )

        # Eqs. (2)-(3): Conv1D -> GELU -> channel-wise LayerNorm.
        branch_outputs = []
        for conv, norm in zip(self.dilated_branches, self.branch_norms):
            branch = F.gelu(conv(pooled))                  # [B, 64, 128]
            branch = norm(branch.transpose(1, 2))          # [B, 128, 64]
            branch = branch.transpose(1, 2).contiguous()   # [B, 64, 128]
            branch_outputs.append(branch)

        # Eq. (4): [B, 192, 128] -> [B, 32, 128].
        concatenated = torch.cat(branch_outputs, dim=1)
        fused_pre_dropout = F.gelu(self.fusion_projection(concatenated))
        fused = self.fusion_dropout(fused_pre_dropout)

        # Eq. (5): [B, 32, 128] -> [B, 4096].
        flattened = fused.flatten(start_dim=1)
        if flattened.shape[1] != FUSION_CHANNELS * POOLED_LENGTH:
            raise RuntimeError(
                "Flattened encoder width must be 4096; "
                f"received {flattened.shape[1]}."
            )

        details = None
        if return_details:
            details = {
                "initial_embedding": initial,
                "pooled_embedding": pooled,
                "branch_1_d2": branch_outputs[0],
                "branch_2_d4": branch_outputs[1],
                "branch_3_d6": branch_outputs[2],
                "concatenated_branches": concatenated,
                "fused_pre_dropout": fused_pre_dropout,
                "fused": fused,
                "flattened": flattened,
            }
        return flattened, details

    def encode(
        self, x: torch.Tensor, return_details: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        flattened, details = self._encode_features(x, return_details=return_details)
        mu = self.fc_mu(flattened)
        logvar = self.fc_logvar(flattened)
        return mu, logvar, details

    def encode_mu(
        self, x: torch.Tensor, return_details: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]]:
        """Deterministic posterior-mean path used for graph freezing/inference."""
        flattened, details = self._encode_features(x, return_details=return_details)
        mu = self.fc_mu(flattened)
        if return_details:
            return mu, details
        return mu

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Eqs. (10)-(12): 64 -> 512 -> 512 -> D; final layer is linear."""
        hidden_1 = F.gelu(self.decoder_fc1(z))
        hidden_2 = F.gelu(self.decoder_fc2(hidden_1))
        return self.decoder_fc3(hidden_2)


class AdaptiveFeatureGraph(nn.Module):
    """Build the 64-node latent-feature graph specified in Eqs. (13)-(18)."""

    def __init__(self) -> None:
        super().__init__()

    def build(
        self, z: torch.Tensor, return_details: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if z.ndim != 2 or z.shape[1] != LATENT_DIM:
            raise ValueError(
                f"Graph input must have shape [B, {LATENT_DIM}], "
                f"received {tuple(z.shape)}."
            )

        # Eq. (13): normalize each latent-feature column over individuals.
        column_norms = torch.linalg.vector_norm(z, ord=2, dim=0, keepdim=True)
        z_bar = z / column_norms.clamp_min(NORM_EPS)

        # Eq. (14): cosine-similarity matrix among the 64 latent dimensions.
        similarity = z_bar.transpose(0, 1) @ z_bar

        # Eq. (15): ELU similarities, excluding self-connections.
        diagonal = torch.eye(LATENT_DIM, dtype=torch.bool, device=z.device)
        logits = F.elu(similarity).masked_fill(diagonal, float("-inf"))

        # Eq. (16): row-wise softmax over the 63 off-diagonal candidates.
        probabilities = torch.softmax(logits, dim=1)

        # Eq. (17): retain exactly eight off-diagonal values per row.
        # Stable sorting resolves exact ties by ascending node index.
        selected_indices = torch.argsort(
            probabilities, dim=1, descending=True, stable=True
        )[:, :GRAPH_TOP_K]
        selected_mask = torch.zeros_like(probabilities)
        selected_mask.scatter_(1, selected_indices, 1.0)
        adjacency_topk = probabilities * selected_mask

        # Eq. (18): max-symmetrization followed by unit-weight self-loops.
        adjacency = torch.maximum(adjacency_topk, adjacency_topk.transpose(0, 1))
        adjacency = adjacency + torch.eye(
            LATENT_DIM, dtype=z.dtype, device=z.device
        )

        if return_details:
            return adjacency, {
                "normalized_latent_columns": z_bar,
                "similarity": similarity,
                "softmax_probabilities": probabilities,
                "topk_adjacency": adjacency_topk,
                "adjacency": adjacency,
            }
        return adjacency


class ChebyshevFeatureGraphConvolution(nn.Module):
    """Degree-2 Chebyshev filter specified in Eqs. (19)-(24)."""

    def __init__(self) -> None:
        super().__init__()
        # Three trainable scalar coefficients: theta_0, theta_1, theta_2.
        # The manuscript does not specify initialization; initialize to one.
        self.theta = nn.Parameter(torch.ones(N_CHEBYSHEV_TERMS))

    @staticmethod
    def scaled_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.shape != (LATENT_DIM, LATENT_DIM):
            raise ValueError(
                f"Adjacency must be [{LATENT_DIM}, {LATENT_DIM}], "
                f"received {tuple(adjacency.shape)}."
            )

        # Eqs. (19)-(21): D, L, and normalized Laplacian.
        degree = adjacency.sum(dim=1)
        inv_sqrt_degree = degree.clamp_min(NORM_EPS).rsqrt()
        normalized_adjacency = (
            inv_sqrt_degree[:, None]
            * adjacency
            * inv_sqrt_degree[None, :]
        )
        identity = torch.eye(LATENT_DIM, dtype=adjacency.dtype, device=adjacency.device)
        normalized_laplacian = identity - normalized_adjacency

        # lambda_max ~= 2, hence L_hat = (2/2)L_tilde - I = L_tilde - I.
        return normalized_laplacian - identity

    def forward(
        self,
        z: torch.Tensor,
        adjacency: torch.Tensor,
        return_terms: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        if z.ndim != 2 or z.shape[1] != LATENT_DIM:
            raise ValueError(
                f"Chebyshev input must be [B, {LATENT_DIM}], received {tuple(z.shape)}."
            )

        scaled_laplacian = self.scaled_laplacian(adjacency)

        # Eqs. (22)-(23): T0, T1, T2, with right multiplication by L_hat.
        t0 = z
        t1 = z @ scaled_laplacian
        t2 = 2.0 * (t1 @ scaled_laplacian) - t0

        # Eq. (24): scalar-coefficient polynomial filter; no extra projection,
        # activation, residual concatenation, or graph-convolution dropout.
        filtered = self.theta[0] * t0 + self.theta[1] * t1 + self.theta[2] * t2

        if return_terms:
            return filtered, (t0, t1, t2)
        return filtered


class DVAGraphGP(nn.Module):
    """Full DVAGraph-GP model following the current mathematical specification.

    Training mode:
        sampled Z -> decoder/reconstruction and dynamic minibatch graph ->
        Chebyshev filter -> global mean pooling -> Linear(1, 1).

    Evaluation mode:
        posterior mean mu -> frozen training-derived graph -> Chebyshev filter
        -> global mean pooling -> Linear(1, 1). The decoder, log-variance head,
        sampling, reconstruction loss, and KL loss are skipped.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.vae = DCFVAE(self.input_dim)
        self.graph_learner = AdaptiveFeatureGraph()
        self.graph_convolution = ChebyshevFeatureGraphConvolution()

        # Eqs. (25)-(26): mean over 64 nodes gives [B, 1], then FC_final 1 -> 1.
        self.regressor = nn.Linear(1, 1)

        self.register_buffer("frozen_adjacency", torch.empty(0, 0), persistent=True)

    @property
    def has_frozen_graph(self) -> bool:
        return self.frozen_adjacency.shape == (LATENT_DIM, LATENT_DIM)

    def clear_frozen_graph(self) -> None:
        self.frozen_adjacency = torch.empty(
            0, 0,
            dtype=next(self.parameters()).dtype,
            device=next(self.parameters()).device,
        )

    @torch.no_grad()
    def freeze_graph_from_training_data(
        self,
        x_train: torch.Tensor,
        batch_size: int = 256,
    ) -> torch.Tensor:
        """Freeze A_adp from posterior means of training-partition samples only.

        Call after fitting the model on the relevant training partition and
        before validation/test prediction. The supplied matrix must already
        have undergone that partition's fitted genotype preprocessing.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        parameter = next(self.parameters())
        x_tensor = torch.as_tensor(x_train)
        if x_tensor.ndim != 2 or x_tensor.shape[1] != self.input_dim:
            raise ValueError(
                f"x_train must have shape [N, {self.input_dim}], "
                f"received {tuple(x_tensor.shape)}."
            )
        if x_tensor.shape[0] < 1:
            raise ValueError("Cannot freeze a graph from an empty training matrix.")

        was_training = self.training
        self.eval()
        means = []
        try:
            for start in range(0, x_tensor.shape[0], batch_size):
                xb = x_tensor[start : start + batch_size].to(
                    device=parameter.device, dtype=parameter.dtype
                )
                means.append(self.vae.encode_mu(xb).detach())
            mu_train = torch.cat(means, dim=0)
            adjacency = self.graph_learner.build(mu_train).detach()
            self.frozen_adjacency = adjacency
        finally:
            self.train(was_training)

        return self.frozen_adjacency

    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ) -> Dict[str, Any]:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected x with shape [B, {self.input_dim}], received {tuple(x.shape)}."
            )
        x = x.to(dtype=next(self.parameters()).dtype)

        if self.training:
            # Eqs. (7)-(9): sample Z during training.
            mu, logvar, encoder_details = self.vae.encode(
                x, return_details=return_details
            )
            z = self.vae.reparameterize(mu, logvar)
            reconstruction = self.vae.decode(z)
            reconstruction_loss = F.mse_loss(reconstruction, x, reduction="mean")
            kl_loss = -0.5 * torch.mean(
                1.0 + logvar - mu.pow(2) - torch.exp(logvar)
            )

            # Algorithm 3: training graph and graph filtering use sampled Z_b.
            adjacency = self.graph_learner.build(z)
            graph_source = "dynamic_training_minibatch_sampled_Z"
        else:
            # Deterministic validation/test path: posterior mean and frozen graph.
            if not self.has_frozen_graph:
                raise RuntimeError(
                    "Evaluation requires a frozen adjacency built only from the "
                    "relevant training partition via freeze_graph_from_training_data()."
                )
            if return_details:
                mu, encoder_details = self.vae.encode_mu(x, return_details=True)
            else:
                mu = self.vae.encode_mu(x, return_details=False)
                encoder_details = None
            z = mu
            logvar = None
            reconstruction = None
            reconstruction_loss = None
            kl_loss = None
            adjacency = self.frozen_adjacency
            graph_source = "frozen_training_partition_posterior_means"

        graph_features = self.graph_convolution(z, adjacency)

        # Eq. (25): global mean pooling over the 64 latent-dimension nodes.
        pooled = graph_features.mean(dim=1, keepdim=True)  # [B, 1]

        # Eq. (26): final fully connected regression layer, [B, 1] -> [B, 1].
        prediction = self.regressor(pooled)

        output: Dict[str, Any] = {
            "prediction": prediction,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "reconstruction": reconstruction,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "adjacency": adjacency,
            "graph_features": graph_features,
            "pooled": pooled,
            "encoder_details": None,
        }
        if return_details:
            output["encoder_details"] = encoder_details
        # A string is returned separately to keep tensor outputs easy to inspect.
        output["graph_source"] = graph_source
        return output


def dvagraph_loss(
    output: Dict[str, Any],
    y: torch.Tensor,
    lambda_pred: float = LAMBDA_PRED,
    lambda_recon: float = LAMBDA_RECON,
    lambda_kl: float = LAMBDA_KL,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute Eqs. (27)-(30); intended for model training mode only."""
    prediction = output.get("prediction")
    reconstruction_loss = output.get("reconstruction_loss")
    kl_loss = output.get("kl_loss")
    if prediction is None or reconstruction_loss is None or kl_loss is None:
        raise ValueError("Training losses require a training-mode model output.")

    if y.ndim == 1:
        y = y.unsqueeze(1)
    if y.shape != prediction.shape:
        raise ValueError(
            f"Phenotype target must match prediction shape {tuple(prediction.shape)}; "
            f"received {tuple(y.shape)}."
        )
    y = y.to(device=prediction.device, dtype=prediction.dtype)

    prediction_loss = F.mse_loss(prediction, y, reduction="mean")
    total = (
        lambda_pred * prediction_loss
        + lambda_recon * reconstruction_loss
        + lambda_kl * kl_loss
    )
    components = {
        "prediction_loss": prediction_loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kl_loss,
        "weighted_prediction_loss": lambda_pred * prediction_loss,
        "weighted_reconstruction_loss": lambda_recon * reconstruction_loss,
        "weighted_kl_loss": lambda_kl * kl_loss,
        "total_loss": total,
    }
    return total, components


__all__ = [
    "LATENT_DIM",
    "GRAPH_TOP_K",
    "CHEBYSHEV_DEGREE",
    "LAMBDA_PRED",
    "LAMBDA_RECON",
    "LAMBDA_KL",
    "DCFVAE",
    "AdaptiveFeatureGraph",
    "ChebyshevFeatureGraphConvolution",
    "DVAGraphGP",
    "dvagraph_loss",
]
