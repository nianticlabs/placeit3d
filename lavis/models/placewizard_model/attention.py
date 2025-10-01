import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, batch_mask, attn_mask=None, pe=None):
        """
        source (B, N_p, d_model)
        batch_offsets Tensor (b, n_p)
        query Tensor (b, n_q, d_model)
        attn_masks Tensor (b, n_q, n_p)
        """
        B = query.shape[0]
        query = self.with_pos_embed(query, pe)
        k = v = source
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(
                B*self.nhead, query.shape[1], k.shape[1])
            output, output_weight = self.attn(
                query, k, v, key_padding_mask=batch_mask, attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, output_weight = self.attn(
                query, k, v, key_padding_mask=batch_mask)
        self.dropout(output)
        output = output + query
        self.norm(output)

        return output, output_weight  # (b, n_q, d_model), (b, n_q, n_v)


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.nhead = nhead
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, x_mask=None, attn_mask=None, pe=None):
        """
        x Tensor (b, n_w, c)
        x_mask Tensor (b, n_w)
        """
        B = x.shape[0]
        q = k = self.with_pos_embed(x, pe)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(
                1, self.nhead, 1, 1).view(B*self.nhead, q.shape[1], k.shape[1])
            # (1, 100, d_model)
            output, _ = self.attn(
                q, k, x, key_padding_mask=x_mask, attn_mask=attn_mask)
        else:
            output, _ = self.attn(q, k, x, key_padding_mask=x_mask)
        output = self.dropout(output) + x
        output = self.norm(output)
        return output


class FFN(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        elif activation_fn == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.net(x)
        output = output + x
        output = self.norm(output)
        return output
