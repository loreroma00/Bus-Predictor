"""Model definitions: OccupancyLSTM (crowd classification) and BusLSTM (Neural-ODE delay regressor)."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchdiffeq import odeint_adjoint as odeint

class OccupancyLSTM(nn.Module):
    """Predicts occupancy/crowd level (7-class classification) using LSTM.

    Supports variable-length sequences via pack_padded_sequence.
    """

    def __init__(self,
                 n_x1_dense_features: int,
                 n_x2_dense_features: int,
                 x1_cat_cardinalities: list,
                 x2_cat_cardinalities: list,
                 encoder_hidden_size: int = 128,
                 lstm_hidden_size: int = 128,
                 num_lstm_layers: int = 2,
                 num_occupancy_classes: int = 7
        ):
        """Build embeddings, FCNN encoder, LSTM decoder, and crowd-classification head."""
        super(OccupancyLSTM, self).__init__()

        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size

        # ================================
        # 1. ENCODER
        # ================================

        self.x1_embeddings = nn.ModuleList()
        x1_total_emb_dim = 0

        for num_categories in x1_cat_cardinalities:
            emb_dim = min(50, (num_categories+1)//2)
            self.x1_embeddings.append(nn.Embedding(num_categories, emb_dim))
            x1_total_emb_dim += emb_dim

        encoder_input_size = n_x1_dense_features + x1_total_emb_dim
        self.encoder_fcnn = nn.Sequential(
            nn.Linear(encoder_input_size, encoder_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(encoder_hidden_size, encoder_hidden_size),
            nn.ReLU()
        )

        # ================================
        # 2. DECODER
        # ================================

        self.x2_embeddings = nn.ModuleList()
        x2_total_emb_dim = 0
        for num_categories in x2_cat_cardinalities:
            emb_dim = min(50, (num_categories + 1) // 2)
            self.x2_embeddings.append(nn.Embedding(num_categories, emb_dim))
            x2_total_emb_dim += emb_dim

        decoder_input_size = n_x2_dense_features + x2_total_emb_dim + encoder_hidden_size
        self.decoder_lstm = nn.LSTM(
            input_size = decoder_input_size,
            hidden_size = lstm_hidden_size,
            num_layers = num_lstm_layers,
            batch_first = True,
            dropout = 0.2 if num_lstm_layers > 1 else 0.0
        )

        # ================================
        # 3. OUTPUTS
        # ================================

        self.head_crowd = nn.Sequential(
            nn.Linear(lstm_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_occupancy_classes)
        )

        print(f"OccupancyLSTM initialized.")
        print(f"Input Encoder: {encoder_input_size} | Input Decoder: {decoder_input_size}")

    def forward(self, x1_cat, x1_dense, x2_cat, x2_dense, lengths=None):
        """Encode trip context, run packed LSTM over stops, and return crowd logits."""
        batch_size = x1_cat.size(0)
        seq_length = x2_dense.size(1)

        # ================================
        # 1. ENCODER
        # ================================

        x1_emb_list = []
        for i, emb_layer in enumerate(self.x1_embeddings):
            x1_emb_list.append(emb_layer(x1_cat[:, i]))
        x1_emb_concat = torch.cat(x1_emb_list, dim=1) if x1_emb_list else torch.empty(batch_size, 0, device=x1_cat.device)

        encoder_input = torch.cat([x1_dense, x1_emb_concat], dim=1)
        context = self.encoder_fcnn(encoder_input)

        # ================================
        # 2. DECODER
        # ================================

        x2_emb_list = []
        for i, emb_layer in enumerate(self.x2_embeddings):
            x2_emb_list.append(emb_layer(x2_cat[:, :, i]))
        x2_emb_concat = torch.cat(x2_emb_list, dim=2) if x2_emb_list else torch.empty(batch_size, seq_length, 0, device=x2_cat.device)

        context_repeated = context.unsqueeze(1).repeat(1, seq_length, 1)
        decoder_input = torch.cat([x2_dense, x2_emb_concat, context_repeated], dim=2)

        # Use pack_padded_sequence for variable-length efficiency
        if lengths is not None:
            lengths_cpu = lengths.cpu().clamp(min=1)
            packed = pack_padded_sequence(decoder_input, lengths_cpu, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.decoder_lstm(packed)
            lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=seq_length)
        else:
            lstm_out, _ = self.decoder_lstm(decoder_input)

        # ================================
        # 3. OUTPUTS
        # ================================

        pred_crowd = self.head_crowd(lstm_out)

        return pred_crowd


class ODEFunc(nn.Module):
    """Small MLP that parameterises the ODE vector field ``dh/dt`` used by BusLSTM."""

    def __init__(self, hidden_dim):
        """Build the 2-layer MLP with Tanh activation."""
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_dim)
        )

    def forward(self, t, h):
        """Return dh/dt at time ``t`` given hidden state ``h``."""
        return self.net(h)


class BusLSTM(nn.Module):
    """Predicts delay using Neural ODE + LSTMCell.

    Supports variable-length sequences via lengths and t_grid parameters.
    The ODE evolves the hidden state between stops proportionally to
    physical distance, and the LSTMCell injects per-stop features.
    """

    def __init__(self, n_x1_dense_features: int, n_x2_dense_features: int, x1_cat_cardinalities: list, x2_cat_cardinalities: list, encoder_hidden_size: int = 128, lstm_hidden_size: int = 128):
        """Build embeddings, FCNN encoder, Neural-ODE block, LSTMCell decoder, and delay head."""
        super(BusLSTM, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size

        # ================================
        # 1. ENCODER
        # ================================

        self.x1_embeddings = nn.ModuleList()
        x1_total_emb_dim = 0

        for num_categories in x1_cat_cardinalities:
            emb_dim = min(50, (num_categories+1)//2)
            self.x1_embeddings.append(nn.Embedding(num_categories, emb_dim))
            x1_total_emb_dim += emb_dim

        encoder_input_size = n_x1_dense_features + x1_total_emb_dim
        self.encoder_fcnn = nn.Sequential(
            nn.Linear(encoder_input_size, encoder_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(encoder_hidden_size, encoder_hidden_size),
            nn.ReLU()
        )

        # =========================
        # 2. DECODER ODE - LSTM
        # =========================

        self.x2_embeddings = nn.ModuleList()
        x2_total_emb_dim = 0
        for num_categories in x2_cat_cardinalities:
            emb_dim = min(50, (num_categories + 1) // 2)
            self.x2_embeddings.append(nn.Embedding(num_categories, emb_dim))
            x2_total_emb_dim += emb_dim

        decoder_input_size = n_x2_dense_features + x2_total_emb_dim + encoder_hidden_size
        self.ode_func = ODEFunc(lstm_hidden_size)
        self.lstm_cell = nn.LSTMCell(decoder_input_size, lstm_hidden_size)

        # =========================
        # 3. OUTPUTS
        # =========================

        self.head_time = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        print(f"BusLSTM (ODE) initialized.")
        print(f"Input Encoder: {encoder_input_size} | Input Decoder: {decoder_input_size}")

    def forward(self, x1_cat, x1_dense, x2_cat, x2_dense, lengths=None, t_grid=None):
        """Encode trip context then ODE-evolve + LSTMCell over stops to emit per-stop delay predictions."""
        batch_size = x1_cat.size(0)
        seq_length = x2_dense.size(1)

        # Determine effective sequence length (skip padding)
        if lengths is not None:
            max_len = int(lengths.max().item())
        else:
            max_len = seq_length

        # ================================
        # 1. ENCODER
        # ================================

        x1_emb_list = []
        for i, emb_layer in enumerate(self.x1_embeddings):
            x1_emb_list.append(emb_layer(x1_cat[:, i]))
        x1_emb_concat = torch.cat(x1_emb_list, dim=1) if x1_emb_list else torch.empty(batch_size, 0, device=x1_cat.device)

        encoder_input = torch.cat([x1_dense, x1_emb_concat], dim=1)
        context = self.encoder_fcnn(encoder_input)

        # ================================
        # 2. DECODER (ODE + LSTMCell)
        # ================================

        x2_emb_list = []
        for i, emb_layer in enumerate(self.x2_embeddings):
            x2_emb_list.append(emb_layer(x2_cat[:, :, i]))
        x2_emb_concat = torch.cat(x2_emb_list, dim=2) if x2_emb_list else torch.empty(batch_size, seq_length, 0, device=x2_cat.device)

        # Build ODE time grid from per-sample stop distances or uniform fallback
        if t_grid is None:
            t_grid = torch.linspace(0, 1, seq_length, device=x1_dense.device).unsqueeze(0).expand(batch_size, -1)

        h_t = torch.zeros(batch_size, self.lstm_hidden_size, device=x1_dense.device)
        c_t = torch.zeros(batch_size, self.lstm_hidden_size, device=x1_dense.device)

        outputs = []

        for t_step in range(max_len):
            # 1. ODE: Continuous evolution between stops
            if t_step > 0:
                t_prev = t_grid[:, t_step - 1]
                t_curr = t_grid[:, t_step]

                # Per-sample ODE integration (use mean t_span for batch efficiency)
                t_span = torch.stack([t_prev.mean(), t_curr.mean()]).to(x1_dense.device)

                # Only integrate if there's actual distance between stops
                if t_span[1] > t_span[0]:
                    h_t = odeint(self.ode_func, h_t, t_span, method='euler')[-1]

            # 2. LSTMCell: Inject per-stop features
            current_x2_dense = x2_dense[:, t_step, :]
            current_x2_emb = x2_emb_concat[:, t_step, :] if x2_emb_concat.size(2) > 0 else torch.empty(batch_size, 0, device=x1_dense.device)

            current_input = torch.cat([current_x2_dense, current_x2_emb, context], dim=1)
            h_t, c_t = self.lstm_cell(current_input, (h_t, c_t))

            outputs.append(h_t)

        outputs = torch.stack(outputs, dim=1)  # [batch, max_len, hidden]

        # Pad back to full seq_length if max_len < seq_length
        if max_len < seq_length:
            pad = torch.zeros(batch_size, seq_length - max_len, self.lstm_hidden_size, device=outputs.device)
            outputs = torch.cat([outputs, pad], dim=1)

        pred_time = self.head_time(outputs)

        return pred_time
