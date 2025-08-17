# models/hrm/hrm_lm_adapter.py
import torch
import torch.nn as nn

class HRMAsTextLM(nn.Module):
    """
    Thin wrapper that exposes your HRM (inner) as a causal LM that accepts embeddings.
      - .tok_emb -> use HRM's token embedding module
      - .d_model -> hidden size
      - forward_with_embeds(inputs_embeds, attention_mask=None) -> logits (B,L,V)
    """
    def __init__(self, hrm_inner):
        super().__init__()
        self.inner = hrm_inner
        self.tok_emb = hrm_inner.embed_tokens               # CastedEmbedding
        self.d_model = hrm_inner.config.hidden_size
        self.lm_head = hrm_inner.lm_head                    # reuse HRM head

    @torch.no_grad()
    def embed(self, input_ids):
        return self.tok_emb(input_ids)

    def forward_with_embeds(self, inputs_embeds, attention_mask=None):
        h = self.inner.core_forward(inputs_embeds, attention_mask=attention_mask)  # (B,L,D)
        return self.lm_head(h)                                                     # (B,L,V)
    