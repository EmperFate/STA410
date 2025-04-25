import math
import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility – sinusoidal timestep embedding (unchanged)
# ---------------------------------------------------------------------------

def timestep_embedding(timesteps: torch.Tensor, dim: int, *, max_period: int = 10_000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings (like Transformer positional encodings)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # zero‑pad for odd dimensions
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

# ---------------------------------------------------------------------------
# Optional Top-K sampling helper function (fixed to accept temperature)
# ---------------------------------------------------------------------------

def topk_sample(logits: torch.Tensor, k: Optional[int] = None, temperature: float = 1.0) -> torch.Tensor:
    """Sample each position from its top-k tokens or full distribution.
    Temperature scales the logits before sampling (higher = more random).
    If k=1, use greedy decoding.
    If k=None, sample from full distribution."""
    # Scale logits by temperature
    if temperature != 1.0:
        logits = logits / temperature

    if k == 1:
        # Greedy decode
        return logits.argmax(dim=-1)
    # Full distribution sampling
    if k is None:
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs.view(-1, logits.size(-1)), 1).view(*logits.shape[:-1])
    # Top-k sampling
    topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
    probs = torch.softmax(topk_vals, -1)
    choice = torch.multinomial(probs.view(-1, k), 1).view(*logits.shape[:-1])
    return torch.gather(topk_idx, -1, choice.unsqueeze(-1)).squeeze(-1)

# ---------------------------------------------------------------------------
# Conditional ε‑network – a Transformer that predicts noise with conditioning
# ---------------------------------------------------------------------------

class ConditionalTransformerNoisePredictor(nn.Module):
    """Conditional ε‑network for the diffusion model (takes both x_t and conditional embedding)."""

    def __init__(self, *, embedding_dim: int = 768, num_layers: int = 3, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.time_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.SiLU(), nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Enhanced conditioning mechanism
        self.cond_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Cross-attention for better conditioning
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with conditioning
        
        Args:
            x: [B,L,D] - Noisy embeddings at timestep t
            t: [B] - Timesteps
            cond: [B,L',D] - Conditional embeddings (corrupted text)
        """
        t_emb = self.time_proj(timestep_embedding(t, x.size(-1))).unsqueeze(1)  # [B,1,D]
        
        if cond is not None:
            h = x + t_emb
            cond_processed = self.cond_proj(cond)
            attn_output, _ = self.cross_attn(h, cond_processed, cond_processed)
            h = h + attn_output
            h = self.encoder(h)
        else:
            h = self.encoder(x + t_emb)
        
        return self.out_proj(h)

# ---------------------------------------------------------------------------
# Main class – ConditionalDiffusionTextDenoiser
# ---------------------------------------------------------------------------

class ConditionalDiffusionTextDenoiser:
    """Conditional diffusion model for text denoising with optional HMC refinement."""

    def __init__(
        self,
        *,
        model_dim: int = 768,
        num_timesteps: int = 20,
        tokenizer_name: str = "bert-base-uncased",
        schedule_type: str = "linear",
        noise_scale: float = 0.05,
        device: str | torch.device | None = None,
        gradient_checkpointing: bool = True,
        k_sampling: Optional[int] = None,
        freeze_lm_head: bool = True,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_dim = model_dim
        self.num_timesteps = num_timesteps
        self.noise_scale = noise_scale
        self.gradient_checkpointing = gradient_checkpointing
        self.k_sampling = k_sampling
        self.freeze_lm_head = freeze_lm_head
        self._last_input_texts = []

        # Tokenizer and encoder
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.embedder = AutoModel.from_pretrained(tokenizer_name).to(self.device)
        if self.gradient_checkpointing:
            self.embedder.gradient_checkpointing_enable()
        self.embedder.eval()
        for p in self.embedder.parameters():
            p.requires_grad_(False)

        # Projection
        hidden_sz = self.embedder.config.hidden_size
        if hidden_sz == self.model_dim:
            self.proj = nn.Identity().to(self.device)
        else:
            self.proj = nn.Linear(hidden_sz, self.model_dim, bias=False).to(self.device)
            nn.init.orthogonal_(self.proj.weight)

        # LM head weight tying
        if hidden_sz == self.model_dim:
            self.lm_head = nn.Linear(self.model_dim, self.tokenizer.vocab_size, bias=False).to(self.device)
            with torch.no_grad():
                self.lm_head.weight.copy_(self.embedder.embeddings.word_embeddings.weight)
                if self.freeze_lm_head:
                    self.lm_head.weight.requires_grad_(False)
        else:
            self.lm_head = nn.Linear(self.model_dim, self.tokenizer.vocab_size, bias=False).to(self.device)
            bert_embs = self.embedder.embeddings.word_embeddings.weight
            U, S, V = torch.svd(bert_embs)
            proj_mat = V[:, :self.model_dim] @ torch.diag(S[:self.model_dim]) / torch.sqrt(torch.tensor(self.model_dim))
            with torch.no_grad():
                self.lm_head.weight.copy_(bert_embs @ proj_mat)
                if self.freeze_lm_head:
                    self.lm_head.weight.requires_grad_(False)

        # Noise predictor
        self.noise_predictor = ConditionalTransformerNoisePredictor(
            embedding_dim=model_dim,
            num_heads=min(8, model_dim // 64)
        ).to(self.device)

        # Diffusion schedule
        self.betas = self._make_beta_schedule(num_timesteps, schedule_type).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    @staticmethod
    def _make_beta_schedule(T: int, schedule_type: str = "linear") -> torch.Tensor:
        if schedule_type == "linear":
            return torch.linspace(1e-4, 2e-2, T)
        if schedule_type == "cosine":
            s = 0.008
            x = torch.linspace(0, T, T + 1)
            alphas = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            return torch.clip(betas, 1e-4, 0.999)
        raise ValueError(schedule_type)

    def _get_embeddings(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        self._last_input_texts = texts
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
        with torch.no_grad():
            last_hidden = self.embedder(**enc, output_hidden_states=False).last_hidden_state
        emb = self.proj(last_hidden)
        return emb, enc.attention_mask

    def _embeddings_to_text(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        k: Optional[int] = None,
        temperature: float = 1.0,
    ) -> List[str]:
        with torch.no_grad():
            logits = self.lm_head(embeddings)
            k_to_use = k if k is not None else self.k_sampling
            token_ids = topk_sample(logits, k=k_to_use, temperature=temperature)
            if attention_mask is not None:
                pad_id = self.tokenizer.pad_token_id or 0
                token_ids = token_ids * attention_mask.long() + (1 - attention_mask.long()) * pad_id
            texts = self.tokenizer.batch_decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        return texts

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0) * self.noise_scale
        sc = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        ss = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sc * x0 + ss * noise

    def _p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        eps_pred = self.noise_predictor(x_t, t, cond)
        if mask is not None:
            eps_pred = eps_pred * mask.unsqueeze(-1)
        sqrt_recip_alpha_t = torch.sqrt(1.0 / self.alphas[t]).view(-1, 1, 1)
        sqrt_recipm1_alpha_t = torch.sqrt(1.0 / self.alphas[t] - 1).view(-1, 1, 1)
        mean = sqrt_recip_alpha_t * (x_t - sqrt_recipm1_alpha_t * eps_pred)
        var = self.posterior_variance[t].view(-1, 1, 1).expand_as(mean)
        return {"mean": mean, "var": var, "log_var": var.log(), "eps_pred": eps_pred}

    def _p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        stats = self._p_mean_variance(x_t, t, mask, cond)
        noise = torch.randn_like(x_t)
        nonzero = (t != 0).float().view(-1, 1, 1)
        return stats["mean"] + nonzero * (stats["var"].sqrt() * noise)

    def _hmc_refine(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        *,
        L: int = 3,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        def energy(z: torch.Tensor) -> torch.Tensor:
            stats = self._p_mean_variance(z, t, mask, cond)
            return 0.5 * (((z - stats["mean"]) ** 2) / stats["var"]).sum(dim=(-1, -2))

        def grad_energy_fn(x: torch.Tensor) -> torch.Tensor:
            x_leaf = x.detach().requires_grad_(True)
            e = energy(x_leaf).sum()
            (grad,) = torch.autograd.grad(e, x_leaf)
            return grad

        m = torch.randn_like(x_t)
        H0 = energy(x_t) + 0.5 * (m**2).sum(dim=(-1, -2))
        z, p = x_t.clone(), m.clone()
        p = p - 0.5 * eps * grad_energy_fn(z)
        for _ in range(L):
            z = z + eps * p
            if mask is not None:
                z = z * mask.unsqueeze(-1)
            if _ < L - 1:
                p = p - eps * grad_energy_fn(z)
        p = p - 0.5 * eps * grad_energy_fn(z)
        p = -p
        H1 = energy(z) + 0.5 * (p**2).sum(dim=(-1, -2))
        log_acc = torch.clamp(H0 - H1, max=20.0, min=-20.0)
        accept_prob = torch.exp(log_acc)
        accept_mask = (torch.rand_like(accept_prob) < accept_prob).view(-1, 1, 1).float()
        logger.info(f"HMC accept-rate = {accept_mask.mean().item():.3f}")
        if torch.isnan(accept_prob).any():
            return x_t
        return accept_mask * z + (1 - accept_mask) * x_t

    def _p_sample_loop(
        self,
        x_T: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        *,
        use_hmc: bool = True,
        L: int = 3,
        eps: float = 1e-3,
        show_progress: bool = False,
    ) -> torch.Tensor:
        x = x_T
        iterator = tqdm(reversed(range(self.num_timesteps))) if show_progress else reversed(range(self.num_timesteps))
        for timestep in iterator:
            t = torch.full((x.size(0),), timestep, dtype=torch.long, device=self.device)
            x = self._p_sample(x, t, mask, cond)
            if use_hmc and timestep > 0:
                x = self._hmc_refine(x, t - 1, mask, cond, L=L, eps=eps)
        return x

    def fit(
        self,
        clean_texts: List[str],
        corrupted_texts: List[str],
        *,
        batch_size: int = 8,
        num_epochs: int = 20,
        lr: float = 1e-4,
        token_loss_weight: float = 1.0,
    ) -> "ConditionalDiffusionTextDenoiser":
        assert len(clean_texts) == len(corrupted_texts), "Clean and corrupted text lists must have the same length"
        trainable_params = list(self.noise_predictor.parameters())
        if not isinstance(self.proj, nn.Identity):
            trainable_params.extend(list(self.proj.parameters()))
        if not self.freeze_lm_head:
            trainable_params.extend(list(self.lm_head.parameters()))
        opt = torch.optim.AdamW(trainable_params, lr=lr)
        idx = np.arange(len(clean_texts))
        for epoch in range(1, num_epochs + 1):
            np.random.shuffle(idx)
            epoch_diffusion_loss = 0.0
            epoch_token_loss = 0.0
            batch_count = 0
            for i in range(0, len(clean_texts), batch_size):
                batch_indices = idx[i : i + batch_size]
                batch_clean = [clean_texts[j] for j in batch_indices]
                batch_corrupted = [corrupted_texts[j] for j in batch_indices]
                clean_emb, clean_mask = self._get_embeddings(batch_clean)
                corr_emb, corr_mask = self._get_embeddings(batch_corrupted)
                cond_emb = corr_emb
                t = torch.randint(0, self.num_timesteps, (clean_emb.size(0),), device=self.device)
                noise = torch.randn_like(clean_emb) * self.noise_scale
                x_t = self.q_sample(clean_emb, t, noise)
                eps_pred = self.noise_predictor(x_t, t, cond=cond_emb)
                if clean_mask is not None:
                    mask = clean_mask.unsqueeze(-1)
                    eps_pred = eps_pred * mask
                    noise = noise * mask
                diffusion_loss = F.mse_loss(eps_pred, noise)
                enc = self.tokenizer(batch_clean, padding=True, truncation=True, 
                                    max_length=64, return_tensors="pt").to(self.device)
                labels = enc.input_ids
                logits = self.lm_head(clean_emb)
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                token_loss = F.cross_entropy(
                    logits_flat, 
                    labels_flat, 
                    ignore_index=self.tokenizer.pad_token_id or 0
                )
                loss = diffusion_loss + token_loss_weight * token_loss
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()
                epoch_diffusion_loss += diffusion_loss.item()
                epoch_token_loss += token_loss.item()
                batch_count += 1
            logger.info(f"Epoch {epoch}/{num_epochs}  |  diffusion_loss = {epoch_diffusion_loss / batch_count:.6f}, token_loss = {epoch_token_loss / batch_count:.6f}")
        return self

    def clean(
        self,
        noisy_texts: List[str],
        *,
        use_hmc: bool = True,
        num_leapfrog_steps: int = 5,
        step_size: float = 1e-3,
        show_progress: bool = False,
        k: Optional[int] = 1,
    ) -> List[str]:
        batch_size = 8
        num_samples = len(noisy_texts)
        all_denoised = []
        for i in range(0, num_samples, batch_size):
            batch_noisy = noisy_texts[i:i+batch_size]
            corr_emb, attn_mask = self._get_embeddings(batch_noisy)
            x_T = torch.randn_like(corr_emb) * self.noise_scale
            x_0 = self._p_sample_loop(
                x_T,
                attn_mask,
                cond=corr_emb,
                use_hmc=use_hmc,
                L=num_leapfrog_steps,
                eps=step_size,
                show_progress=show_progress
            )
            batch_denoised = self._embeddings_to_text(x_0, attn_mask, k=k)
            all_denoised.extend(batch_denoised)
        return all_denoised

    def restore(
        self,
        noisy_texts: List[str],
        *,
        start_step: int | None = None,
        use_hmc: bool = True,
        num_leapfrog_steps: int = 5,
        step_size: float = 1e-3,
        show_progress: bool = False,
        k: Optional[int] = 1,
    ) -> List[str]:
        batch_size = 8
        num_samples = len(noisy_texts)
        all_restored = []
        for i in range(0, num_samples, batch_size):
            batch_noisy = noisy_texts[i:i+batch_size]
            corr_emb, mask = self._get_embeddings(batch_noisy)
            bsz = corr_emb.size(0)
            T0 = self.num_timesteps - 1 if start_step is None else int(
                max(0, min(start_step, self.num_timesteps - 1))
            )
            t_T0 = torch.full((bsz,), T0, dtype=torch.long, device=self.device)
            x = self.q_sample(corr_emb, t_T0)
            iterator = tqdm(reversed(range(T0 + 1))) if show_progress else reversed(range(T0 + 1))
            for t in iterator:
                t_t = torch.full((bsz,), t, dtype=torch.long, device=self.device)
                stats = self._p_mean_variance(x, t_t, mask, cond=corr_emb)
                noise = torch.randn_like(x) if t > 0 else 0
                x = stats["mean"] + (t > 0) * torch.sqrt(stats["var"]) * noise
                if use_hmc and t > 0:
                    x = self._hmc_refine(x, t_t - 1, mask, cond=corr_emb, L=num_leapfrog_steps, eps=step_size)
            batch_restored = self._embeddings_to_text(x, mask, k=k)
            all_restored.extend(batch_restored)
        return all_restored
