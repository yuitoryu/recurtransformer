import torch.nn as nn
import torch

class RecurrentTransformer(nn.Module):
    def __init__(self, 
                 d_model=768, 
                 nhead=16,
                 dim_feedforward=2048,  
                 activation='relu',
                 max_seq_len=1024,
                 vocab_size=50257,
                 device='cpu',
                 recur_pattern=[(16,1)]):

        super().__init__()
        self.recur_pattern = recur_pattern
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        # Create decoder layers by group
        self.layers = nn.ModuleList()
        for num_layers, _ in recur_pattern:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation,
                layer_norm_eps=1e-5,
                batch_first=True,
            )
            group = nn.TransformerEncoder(layer, num_layers=num_layers)

            self.layers.append(group)
            
        # Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embedding
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # To logits
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            # QKV is stored as a parameter, rather than a child Linear module.
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)

        x = self.token_embedding(x) + self.position_embedding(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len,
            device=x.device,
            dtype=x.dtype,
        )

        for group, (_, num_repeats) in zip(self.layers, self.recur_pattern):
            for _ in range(num_repeats):
                for layer in group.layers:
                    x = layer(x, src_mask=causal_mask)

        x = self.final_norm(x)
        return self.lm_head(x)