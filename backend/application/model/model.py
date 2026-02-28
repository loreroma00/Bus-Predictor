import torch
import torch.nn as nn

class BusLSTM(nn.Module):
    
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
        super(BusLSTM, self).__init__()

        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size

        # ================================
        # 1. ENCODER
        # ================================

        # A - Embeddings 
        
        self.x1_embeddings = nn.ModuleList()
        x1_total_emb_dim = 0

        for num_categories in x1_cat_cardinalities:
            emb_dim = min(50, (num_categories+1)//2)
            self.x1_embeddings.append(nn.Embedding(num_categories, emb_dim))
            x1_total_emb_dim += emb_dim

        # B. FCNN Network

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

        # A. Embeddings for Decoder

        self.x2_embeddings = nn.ModuleList()
        x2_total_emb_dim = 0
        for num_categories in x2_cat_cardinalities:
            emb_dim = min(50, (num_categories + 1) // 2)
            self.x2_embeddings.append(nn.Embedding(num_categories, emb_dim))
            x2_total_emb_dim += emb_dim

        # B. LSTM

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

        # Head 1 - Delay
        self.head_time = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        # Head 2 - Occupancy
        self.head_crowd = nn.Sequential(
            nn.Linear(lstm_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_occupancy_classes)
        )

        print(f"Model initialized.")
        print(f"Input Encoder: {encoder_input_size} | Input Decoder: {decoder_input_size}")
        
    def forward(self, x1_cat, x1_dense, x2_cat, x2_dense):

        # ================================
        # 1. ENCODER
        # ===============================

        # A. Categorical X1


        x1_emb_list = []
        for i, emb_layer in enumerate(self.x1_embeddings):
            x1_emb_list.append(emb_layer(x1_cat[:, i]))
        x1_emb_concat = torch.cat(x1_emb_list, dim=1) if x1_emb_list else torch.empty(x1_cat.size(0), 0).to(x1_cat.device)

        # B. Combining Dense & Embedding
        
        encoder_input = torch.cat([x1_dense, x1_emb_concat], dim=1)
        context = self.encoder_fcnn(encoder_input)

        # ===============================
        # 2. DECODER
        # ===============================

        # A. Categorical X2

        x2_emb_list = []
        for i, emb_layer in enumerate(self.x2_embeddings):
            x2_emb_list.append(emb_layer(x2_cat[:, :, i]))

        x2_emb_concat = torch.cat(x2_emb_list, dim=2) if x2_emb_list else torch.empty(batch_size, x2_cat.size(1), 0).to(x2_cat.device)

        # B. Decoder Input

        seq_length = x2_dense.size(1)
        context_repeated = context.unsqueeze(1).repeat(1, seq_length, 1)
        decoder_input = torch.cat([x2_dense, x2_emb_concat, context_repeated], dim=2)

        # C. Injecting into LSTM
        lstm_out, _ = self.decoder_lstm(decoder_input)

        # ===============================
        # 3. OUTPUTS
        # ===============================

        pred_time = self.head_time(lstm_out)
        pred_crowd = self.head_crowd(lstm_out)

        return pred_time, pred_crowd
        
