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
                norm_first=True,
                device=device
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

class WeightedBlock(nn.Module):
    """Reuse m causal layers n times and weight their per-loop outputs.

    Concatenate loop outputs at each position and compute n causal weights
    separately for each token.
    Inputs and outputs have shape (batch, sequence, d_model).
    """

    def __init__(self,
                 pattern,
                 d_model=768, 
                 nhead=16,
                 dim_feedforward=2048,  
                 activation='relu',
                 device='cpu',
                 dropout=0.1):
        
        super().__init__()
        self.repeat = pattern[1]
        if pattern[0] < 1 or self.repeat < 1:
            raise ValueError("pattern must contain positive layer and repeat counts")
        router_dim = d_model // 12
        if router_dim < 1:
            raise ValueError("WeightedBlock requires d_model >= 12")
        
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=True,
            device=device)
        self.encoder = nn.TransformerEncoder(layer, num_layers=pattern[0])
        self.w_down = nn.Linear(self.repeat * d_model, router_dim, bias=False, device=device)
        self.causal_attn = nn.MultiheadAttention(
            router_dim, num_heads=1, dropout=0.0, batch_first=True, device=device,
        )
        # W_out in the routing equation is W_up in the dimension definition.
        self.w_up = nn.Linear(router_dim, self.repeat, bias=False, device=device)
        
    def forward(self, x):
        seq_len = x.size(1)
        causal_mask = torch.ones(
            seq_len, seq_len, dtype=torch.bool, device=x.device,
        ).triu(1)

        loop_outputs = []
        for _ in range(self.repeat):
            for layer in self.encoder.layers:
                x = layer(x, src_mask=causal_mask)
            loop_outputs.append(x)

        # C has shape (batch, sequence, n * d_model).
        compressed = self.w_down(torch.cat(loop_outputs, dim=-1))
        routed_hidden, _ = self.causal_attn(
            compressed, compressed, compressed,
            attn_mask=causal_mask, need_weights=False,
        )
        weights = self.w_up(routed_hidden).softmax(dim=-1)  # (batch, sequence, n)
        hidden = torch.stack(loop_outputs, dim=2)  # (batch, sequence, n, d_model)
        return (hidden * weights.unsqueeze(-1)).sum(dim=2)
        
    
class RecurrentTransformerW(nn.Module):
    """Build ordinary or weighted blocks from [layer_count, repeat_count]."""

    def __init__(self, 
                 d_model=768, 
                 nhead=16,
                 dim_feedforward=2048,  
                 activation='relu',
                 max_seq_len=1024,
                 vocab_size=50257,
                 device='cpu',
                 recur_pattern=[[1,1], [6,1], [1,1]]):

        super().__init__()
        self.recur_pattern = recur_pattern
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        self.layers = nn.ModuleList()
        for num_layers, num_repeats in recur_pattern:
            if num_layers < 1 or num_repeats < 1:
                raise ValueError("pattern must contain positive layer and repeat counts")
            if num_repeats == 1:
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    activation=activation,
                    layer_norm_eps=1e-5,
                    batch_first=True,
                    norm_first=True,
                    device=device,
                )
                group = nn.TransformerEncoder(layer, num_layers=num_layers)
            else:
                group = WeightedBlock(
                    pattern=[num_layers, num_repeats],
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    activation=activation,
                    device=device,
                )
            self.layers.append(group)

        self.token_embedding = nn.Embedding(vocab_size, d_model, device=device)
        self.position_embedding = nn.Embedding(max_seq_len, d_model, device=device)
        self.final_norm = nn.LayerNorm(d_model, device=device)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device)

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

        for group in self.layers:
            if isinstance(group, WeightedBlock):
                x = group(x)
            else:
                x = group(x, mask=causal_mask)

        x = self.final_norm(x)
        return self.lm_head(x)
