"""Block Attention Residuals; block_size counts attention and FFN sublayers."""

import torch
from torch import nn


class BlockAttentionResidual(nn.Module):
    """Attend to completed block sums and the current block's partial sum."""

    def __init__(self, d_model, eps=1e-5, device="cpu"):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(d_model, device=device))
        self.norm = nn.RMSNorm(d_model, eps=eps, device=device)

    def forward(self, sources):
        values = torch.stack(sources)
        score_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
        keys = self.norm(values.to(score_dtype))
        scores = (keys * self.query.to(score_dtype)).sum(dim=-1)
        weights = scores.softmax(dim=0).to(values.dtype)
        return (weights.unsqueeze(-1) * values).sum(dim=0)


class BlockHistory:
    """Per-forward history shared across groups and recurrent executions."""

    def __init__(self, embedding, block_size):
        self.completed = [embedding]
        self.partial = None
        self.count = 0
        self.block_size = block_size

    def sources(self):
        if self.partial is None:
            return self.completed
        return self.completed + [self.partial]

    def append(self, branch_output):
        self.partial = branch_output if self.partial is None else self.partial + branch_output
        self.count += 1
        if self.count == self.block_size:
            self.completed.append(self.partial)
            self.partial = None
            self.count = 0


class BlockAttnResLayer(nn.Module):
    """Share branch weights across repeats, with independent depth queries."""

    def __init__(self, d_model, nhead, dim_feedforward, num_repeats,
                 dropout=0.1, activation="relu", device="cpu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True, device=device,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, device=device)
        self.linear2 = nn.Linear(dim_feedforward, d_model, device=device)
        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.norm2 = nn.LayerNorm(d_model, device=device)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = {"relu": nn.ReLU, "gelu": nn.GELU}[activation]()
        self.attn_res = nn.ModuleList(
            BlockAttentionResidual(d_model, device=device) for _ in range(num_repeats)
        )
        self.ffn_res = nn.ModuleList(
            BlockAttentionResidual(d_model, device=device) for _ in range(num_repeats)
        )

    def forward(self, history, causal_mask, repeat_index):
        x = self.norm1(self.attn_res[repeat_index](history.sources()))
        attention, _ = self.self_attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        history.append(self.dropout1(attention))
        x = self.norm2(self.ffn_res[repeat_index](history.sources()))
        feedforward = self.linear2(self.dropout(self.activation(self.linear1(x))))
        history.append(self.dropout2(feedforward))


class RecurrentTransformerBlockAttnRes(nn.Module):
    """Standalone recurrent Transformer with block-compressed depth history.

    Each pattern entry is (independent Transformer layers, execution repeats).
    Block boundaries count executed attention/FFN branches globally, including
    repeated executions. Embeddings remain a separate source; an unfinished last
    block is included in the output readout. Default block_size=4 therefore groups
    two complete Transformer layers. No weights are tied to the LM head.
    """

    def __init__(self, d_model=768, nhead=16, dim_feedforward=2048,
                 activation="relu", max_seq_len=1024, vocab_size=50257,
                 device="cpu", recur_pattern=((16, 1),), block_size=4,
                 dropout=0.1):
        super().__init__()
        if block_size < 1:
            raise ValueError("block_size must be positive (counting attention/FFN sublayers)")
        self.recur_pattern = recur_pattern
        self.block_size = block_size
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.layers = nn.ModuleList(
            nn.ModuleList(
                BlockAttnResLayer(d_model, nhead, dim_feedforward, repeats,
                                 dropout, activation, device)
                for _ in range(count)
            )
            for count, repeats in recur_pattern
        )
        self.token_embedding = nn.Embedding(vocab_size, d_model, device=device)
        self.position_embedding = nn.Embedding(max_seq_len, d_model, device=device)
        self.output_attn_res = BlockAttentionResidual(d_model, device=device)
        self.final_norm = nn.LayerNorm(d_model, device=device)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, inputs):
        length = inputs.size(1)
        positions = torch.arange(length, device=inputs.device)
        embedding = self.token_embedding(inputs) + self.position_embedding(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            length, device=inputs.device, dtype=embedding.dtype,
        )
        history = BlockHistory(embedding, self.block_size)
        for group, (_, repeats) in zip(self.layers, self.recur_pattern):
            for repeat_index in range(repeats):
                for layer in group:
                    layer(history, causal_mask, repeat_index)
        output = self.output_attn_res(history.sources())
        return self.lm_head(self.final_norm(output))
