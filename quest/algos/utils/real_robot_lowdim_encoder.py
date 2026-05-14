import torch.nn as nn

from quest.algos.utils.mlp_proj import MLPProj


SEQUENCE_KEYS = {
    "left_force_history",
    "right_force_history",
}


class ResidualTemporalConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd for same-padding conv, got {kernel_size}")
        self.residual_proj = None
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        self.conv = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        residual = x if self.residual_proj is None else self.residual_proj(x)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x)
        x = self.activation(x)
        return x + residual


class SequenceConvEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        history_len=10,
        input_dim=None,
        conv_channels=64,
        cnn_features=None,
        kernel_size=3,
        dropout=0.0,
    ):
        super().__init__()
        if input_dim is None:
            if input_size % history_len != 0:
                raise ValueError(
                    f"input_size={input_size} must be divisible by history_len={history_len} "
                    "when input_dim is not set"
                )
            input_dim = input_size // history_len
        expected_size = history_len * input_dim
        if input_size != expected_size:
            raise ValueError(
                f"Sequence input_size={input_size} does not match "
                f"history_len * input_dim = {history_len} * {input_dim} = {expected_size}"
            )

        self.history_len = history_len
        self.input_dim = input_dim
        self.out_channels = output_size
        if cnn_features is None:
            cnn_features = (conv_channels, 128, 128)
        self.input_proj = nn.Linear(input_dim, 32)
        self.activation = nn.SiLU()
        self.blocks = nn.ModuleList()
        current_dim = 32
        for feature_dim in cnn_features:
            self.blocks.append(
                ResidualTemporalConvBlock(
                    current_dim,
                    feature_dim,
                    kernel_size=kernel_size,
                )
            )
            current_dim = feature_dim
        self.dropout = nn.Dropout(p=dropout)
        self.output_proj = nn.Linear(current_dim, output_size)

    def forward(self, data):
        if data.shape[-2:] != (self.history_len, self.input_dim):
            raise ValueError(
                f"Expected sequence shape (..., {self.history_len}, {self.input_dim}), "
                f"got {tuple(data.shape)}"
            )
        leading_shape = data.shape[:-2]
        x = data.reshape(-1, self.history_len, self.input_dim)
        x = self.activation(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.output_proj(self.dropout(x))
        return x.reshape(*leading_shape, self.out_channels)


class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=None,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()
        self.out_channels = output_size
        self.encoder = MLPProj(
            input_size,
            output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, data):
        if data.shape[-1] == self.encoder.projection[0].in_features:
            return self.encoder(data)
        leading_shape = data.shape[:-2]
        data = data.reshape(*leading_shape, -1)
        return self.encoder(data)


class RealRobotLowdimEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=None,
        num_layers=1,
        dropout=0.0,
        input_name=None,
        history_len=10,
        ft_dim=6,
        conv_channels=64,
        cnn_features=None,
        kernel_size=3,
        encoder_type="auto",
        encoder_type_by_modality=None,
        input_dim_by_modality=None,
    ):
        super().__init__()
        self.out_channels = output_size
        self.token_name = input_name or "lowdim"
        self.token_out_channels = {self.token_name: output_size}

        encoder_types = dict(encoder_type_by_modality or {})
        input_dims = dict(input_dim_by_modality or {})
        selected_type = encoder_types.get(self.token_name, encoder_type)
        if selected_type == "auto":
            selected_type = "conv" if self.token_name in SEQUENCE_KEYS else "mlp"

        if selected_type == "conv":
            input_dim = input_dims.get(self.token_name)
            if input_dim is None and self.token_name in {"left_force_history", "right_force_history"}:
                input_dim = ft_dim
            self.encoder = SequenceConvEncoder(
                input_size,
                output_size,
                history_len=history_len,
                input_dim=input_dim,
                conv_channels=conv_channels,
                cnn_features=cnn_features,
                kernel_size=kernel_size,
                dropout=dropout,
            )
        elif selected_type == "mlp":
            self.encoder = MLPEncoder(
                input_size,
                output_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"Unsupported encoder type '{selected_type}' for lowdim modality '{self.token_name}'. "
                "Expected one of: auto, mlp, conv."
            )

    def forward(self, data):
        return self.encoder(data)

    def encode_tokens(self, data):
        return {self.token_name: self.forward(data)}


ForceHistoryConvEncoder = SequenceConvEncoder
