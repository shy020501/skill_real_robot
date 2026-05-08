import torch
import torch.nn as nn

from quest.algos.utils.mlp_proj import MLPProj


FORCE_HISTORY_KEYS = {"right_force_history", "force_history"}
STATE_FORCE_HISTORY_KEYS = {"right_state_force_history"}


class ForceHistoryConvEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        history_len=10,
        ft_dim=6,
        conv_channels=64,
        kernel_size=3,
        dropout=0.0,
    ):
        super().__init__()
        expected_size = history_len * ft_dim
        if input_size != expected_size:
            raise ValueError(
                f"Force history input_size={input_size} does not match "
                f"history_len * ft_dim = {history_len} * {ft_dim} = {expected_size}"
            )

        padding = kernel_size // 2
        self.history_len = history_len
        self.ft_dim = ft_dim
        self.out_channels = output_size
        self.encoder = nn.Sequential(
            nn.Conv1d(ft_dim, conv_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=-2),
            nn.Dropout(p=dropout),
            nn.Linear(conv_channels, output_size),
        )

    def forward(self, data):
        leading_shape = data.shape[:-1]
        x = data.reshape(-1, self.history_len, self.ft_dim)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        return x.reshape(*leading_shape, self.out_channels)


class StateForceHistoryEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        history_len=10,
        ft_dim=6,
        conv_channels=64,
        state_hidden_size=None,
        state_num_layers=1,
        fusion_hidden_size=None,
        dropout=0.0,
    ):
        super().__init__()
        history_size = history_len * ft_dim
        state_size = input_size - history_size
        if state_size <= 0:
            raise ValueError(
                f"state_force_history input_size={input_size} must be larger than "
                f"force history size {history_size}"
            )

        self.state_size = state_size
        self.history_size = history_size
        self.out_channels = output_size
        branch_size = output_size

        self.state_encoder = MLPProj(
            state_size,
            branch_size,
            hidden_size=state_hidden_size,
            num_layers=state_num_layers,
            dropout=dropout,
        )
        self.force_history_encoder = ForceHistoryConvEncoder(
            history_size,
            branch_size,
            history_len=history_len,
            ft_dim=ft_dim,
            conv_channels=conv_channels,
            dropout=dropout,
        )

        if fusion_hidden_size is None:
            fusion_hidden_size = output_size
        self.fusion = nn.Sequential(
            nn.Linear(branch_size * 2, fusion_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_hidden_size, output_size),
        )

    def forward(self, data):
        state = data[..., :self.state_size]
        force_history = data[..., self.state_size:self.state_size + self.history_size]
        state_emb = self.state_encoder(state)
        force_emb = self.force_history_encoder(force_history)
        return self.fusion(torch.cat([state_emb, force_emb], dim=-1))


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
        kernel_size=3,
        fusion_hidden_size=None,
    ):
        super().__init__()
        self.out_channels = output_size

        if input_name in FORCE_HISTORY_KEYS:
            self.encoder = ForceHistoryConvEncoder(
                input_size,
                output_size,
                history_len=history_len,
                ft_dim=ft_dim,
                conv_channels=conv_channels,
                kernel_size=kernel_size,
                dropout=dropout,
            )
        elif input_name in STATE_FORCE_HISTORY_KEYS:
            self.encoder = StateForceHistoryEncoder(
                input_size,
                output_size,
                history_len=history_len,
                ft_dim=ft_dim,
                conv_channels=conv_channels,
                state_hidden_size=hidden_size,
                state_num_layers=num_layers,
                fusion_hidden_size=fusion_hidden_size,
                dropout=dropout,
            )
        else:
            self.encoder = MLPProj(
                input_size,
                output_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )

    def forward(self, data):
        return self.encoder(data)
