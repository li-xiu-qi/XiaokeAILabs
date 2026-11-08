"""llm_common.py — 全链路共享模块：模型、优化器、数据与生成工具。

架构参考 karpathy/nanochat 的 gpt.py 与 optim.py，按本实验的 CPU 玩具规模裁剪：
  - RMSNorm 不带学习参数（norm() 直接 F.rms_norm，对齐 nanochat「no learnable params in rmsnorm」）
  - RoPE 旋转位置编码（base=100000），先施加旋转、后做 QK norm（对齐 nanochat 顺序）
  - MLP 默认 relu^2（nanochat 选择），保留 SwiGLU 作为可选对照
  - 词嵌入与 lm_head 默认不共享（untied），保留 tie_weights 开关
  - 嵌入后接一次 norm；logits 做 softcap=15 软截断
  - GQA（n_kv_heads <= n_heads）代码路径保留，玩具规模默认 MHA
  - KV cache 推理引擎：训练/评估走整段前向，生成为每步单 token 前向
  - Muon（Newton-Schulz 正交化）负责隐藏矩阵，AdamW 负责嵌入/lm_head/标量

所有编号脚本（01~08）通过 `import llm_common` 复用本模块，不再需要 importlib 路径加载。
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 小模型在多核 Windows 上默认 8 线程反而因争用变慢，实测 4 线程最快；可用环境变量覆盖
torch.set_num_threads(int(os.environ.get("LLM_THREADS", "4")))

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"
TOK_PATH = RESULT_DIR / "tokenizer" / "tokenizer.json"
CKPT_DIR = RESULT_DIR / "checkpoints"
EOS = "<|endoftext|>"
PAD = "<|pad|>"


# ------------------------------------------------------------------- 配置 ----


@dataclass
class ModelConfig:
    vocab_size: int = 2048
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 6
    n_kv_heads: int = 6  # GQA：key/value 头数，<= n_heads 且整除；等于时为普通 MHA
    d_ff: int = 768  # MLP 隐层（relu^2 取 4*d_model；SwiGLU 取约 8/3*d_model）
    max_len: int = 256  # RoPE 预计算长度
    rope_base: float = 100000.0
    dropout: float = 0.0
    tie_weights: bool = False  # nanochat 默认 untied
    mlp_type: str = "relu2"  # "relu2"（nanochat）或 "swiglu"（对照）
    qk_norm: bool = True  # RoPE 后对 q/k 做 RMSNorm（nanochat QK norm）
    embed_norm: bool = True  # 词嵌入后接一次 norm（nanochat）
    softcap: float = 15.0  # logits 软截断，<=0 关闭

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# -------------------------------------------------------------- 基础组件 ----


def norm(x: torch.Tensor) -> torch.Tensor:
    """无参数 RMSNorm（nanochat 风格：不做均值中心化、无可学习权重）。"""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对 (B, H, T, D) 的 q/k 施加 RoPE。cos/sin 形状 (1, T, 1, D/2)，成对旋转维度。"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos.transpose(1, 2).to(x.dtype)  # (1, 1, T, D/2)
    sin = sin.transpose(1, 2).to(x.dtype)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


class RotaryEmbedding(nn.Module):
    """预计算 cos/sin 缓存，长度不足时自动扩展。"""

    def __init__(self, head_dim: int, max_len: int, base: float):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (T, D/2)
        cos = freqs.cos()[None, :, None, :]  # (1, T, 1, D/2)
        sin = freqs.sin()[None, :, None, :]
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def get_cos_sin(self, seq_len: int, offset: int = 0):
        if offset + seq_len > self.cos_cached.size(1):
            self._build_cache(offset + seq_len)
        return (
            self.cos_cached[:, offset : offset + seq_len],
            self.sin_cached[:, offset : offset + seq_len],
        )


class MLP(nn.Module):
    """relu^2（nanochat）或 SwiGLU 前馈，均无 bias。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.mlp_type == "swiglu":
            # SwiGLU 保持参数量与 relu^2 版接近：3 个 d_model->d_ff 投影
            self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        else:
            self.c_fc = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.c_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "gate_proj"):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return self.c_proj(F.relu(self.c_fc(x)).square())


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qk_norm = cfg.qk_norm
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_len, cfg.rope_base)

    def forward(
        self, x: torch.Tensor, kv_cache: "KVCache | None" = None, offset: int = 0
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope.get_cos_sin(t, offset)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        if self.qk_norm:  # nanochat：RoPE 之后归一化 q/k
            q, k = norm(q), norm(k)

        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)
        if self.n_kv_heads != self.n_heads:  # GQA：把 kv 头复制到 query 头数
            rep = self.n_heads // self.n_kv_heads
            if rep > 1:
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=(kv_cache is None or t > 1))
        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(y)


class Block(nn.Module):
    """预归一化残差块：x + attn(norm(x)); x + mlp(norm(x))。"""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.attn.layer_idx = layer_idx
        self.mlp = MLP(cfg)

    def forward(
        self, x: torch.Tensor, kv_cache: "KVCache | None" = None, offset: int = 0
    ) -> torch.Tensor:
        x = x + self.attn(norm(x), kv_cache=kv_cache, offset=offset)
        x = x + self.mlp(norm(x))
        return x


class KVCache:
    """逐层缓存 key/value。形状 (B, H_kv, max_len, D)，支持 batch>1 的批量解码。"""

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        max_len: int,
        batch: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        self.max_len = max_len
        self.k = [
            torch.empty(batch, n_kv_heads, max_len, head_dim, dtype=dtype) for _ in range(n_layers)
        ]
        self.v = [
            torch.empty(batch, n_kv_heads, max_len, head_dim, dtype=dtype) for _ in range(n_layers)
        ]
        self.pos = 0

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        t = k_new.size(2)
        assert self.pos + t <= self.max_len, "KV cache 超出预分配长度"
        self.k[layer_idx][:, :, self.pos : self.pos + t] = k_new
        self.v[layer_idx][:, :, self.pos : self.pos + t] = v_new
        end = self.pos + t  # 含本次写入的全部有效长度
        if layer_idx == len(self.k) - 1:
            self.pos = end
        return self.k[layer_idx][:, :, :end], self.v[layer_idx][:, :, :end]


# ------------------------------------------------------------------- 模型 ----


class CausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        offset: int = 0,
    ):
        # idx: (B, T)；训练时 targets 与 idx 同形（内部错位一格）
        b, t = idx.shape
        x = self.tok_emb(idx)
        if self.cfg.embed_norm:
            x = norm(x)
        for blk in self.blocks:
            x = blk(x, kv_cache=kv_cache, offset=offset)
        x = norm(x)
        logits = self.lm_head(x)  # (B, T, V)
        if self.cfg.softcap and self.cfg.softcap > 0:  # 软截断到 ±softcap
            logits = self.cfg.softcap * torch.tanh(logits / self.cfg.softcap)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_param_groups(self) -> dict:
        """按 nanochat num_scaling_params 的口径分组统计参数。"""
        wte = sum(p.numel() for p in self.tok_emb.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        matrices = sum(p.numel() for p in self.blocks.parameters())
        total = wte + lm_head + matrices
        return {
            "tok_emb": wte,
            "lm_head": lm_head,
            "transformer_matrices": matrices,
            "total": total,
        }


def build_model(cfg: ModelConfig, seed: int = 0) -> CausalLM:
    torch.manual_seed(seed)
    return CausalLM(cfg)


def save_checkpoint(model: CausalLM, cfg: ModelConfig, path: Path, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg.to_dict(), **meta}, path)


def load_checkpoint(path: Path) -> tuple[CausalLM, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CausalLM(ModelConfig.from_dict(ckpt["config"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


# ------------------------------------------------------------- Muon 优化器 ----


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz 迭代求矩阵的近似正交化（零次幂），Keller Jordan/modded-nanogpt 版。

    CPU 上用 float32 计算：小矩阵上 ns 迭代开销可忽略，且避免 bf16 在 CPU 上的性能坑。
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + 1e-7)
    transposed = False
    if X.size(0) > X.size(1):  # 保持行<=列，迭代更稳
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """对二维隐藏矩阵做动量 + Newton-Schulz 正交化的优化器（nanochat 的 matrix 优化器）。"""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            list(params),
            dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay),
        )

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp(buf, group["momentum"]) if group["momentum"] != 0 else g
                update = zeropower_via_newtonschulz5(g, group["ns_steps"])
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5  # 按形状缩放更新幅度
                if group["weight_decay"] > 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.to(p.dtype), alpha=-group["lr"] * scale)


class AdamWGroup(torch.optim.AdamW):
    """AdamW，仅用于显式声明 param_groups 的标签场景（保持接口一致）。"""


def build_optimizer(
    model: CausalLM,
    cfg: ModelConfig,
    lr: float,
    weight_decay: float = 0.0,
    optimizer: str = "muon",
    matrix_lr: float = 0.02,
    embedding_lr: float = 0.3,
    unembedding_lr: float = 0.004,
    scalar_lr: float = 0.04,
):
    """按 nanochat setup_optimizer 的分组思想构建混合优化器。

    - hidden matrices（blocks 内全部参数）→ Muon（optimizer='muon' 时）
    - tok_emb / lm_head / 其它标量 → AdamW，lr 按组分别设置
    - optimizer='adamw' 时全部走 AdamW（对照组）
    """
    matrix_params = [p for p in model.blocks.parameters()]
    embedding_params = [p for p in model.tok_emb.parameters()]
    lm_head_params = [p for p in model.lm_head.parameters()]
    others = [
        p
        for n, p in model.named_parameters()
        if not n.startswith(("blocks.", "tok_emb.", "lm_head."))
    ]
    all_ids = {id(p) for p in list(model.parameters())}
    grouped_ids = {id(p) for p in matrix_params + embedding_params + lm_head_params + others}
    assert grouped_ids == all_ids, "参数分组没有覆盖全部参数"

    if optimizer == "muon":
        groups = [
            dict(
                kind="muon",
                params=matrix_params,
                lr=matrix_lr,
                momentum=0.95,
                ns_steps=5,
                weight_decay=weight_decay,
            )
        ]
    else:
        groups = [
            dict(
                kind="adamw",
                params=matrix_params,
                lr=lr,
                betas=(0.9, 0.95),
                eps=1e-10,
                weight_decay=weight_decay,
            )
        ]
    groups += [
        dict(
            kind="adamw",
            params=embedding_params,
            lr=embedding_lr,
            betas=(0.8, 0.995),
            eps=1e-10,
            weight_decay=0.001,
        ),
        dict(
            kind="adamw",
            params=lm_head_params,
            lr=unembedding_lr,
            betas=(0.8, 0.96),
            eps=1e-10,
            weight_decay=0.01,
        ),
    ]
    if others:
        groups.append(
            dict(
                kind="adamw",
                params=others,
                lr=scalar_lr,
                betas=(0.8, 0.95),
                eps=1e-10,
                weight_decay=0.0,
            )
        )

    muon_groups = [g for g in groups if g["kind"] == "muon"]
    adamw_groups = [g for g in groups if g["kind"] == "adamw"]
    optimizers = []
    if muon_groups:
        optimizers.append(Muon([{k: v for k, v in g.items() if k != "kind"} for g in muon_groups]))
    if adamw_groups:
        optimizers.append(
            torch.optim.AdamW([{k: v for k, v in g.items() if k != "kind"} for g in adamw_groups])
        )
    groups_info = [
        {"kind": g["kind"], "lr": g["lr"], "n_params": sum(p.numel() for p in g["params"])}
        for g in groups
    ]
    return optimizers, groups_info


def lr_at(
    step: int, total: int, base_lr: float, warmup_frac: float = 0.1, floor_frac: float = 0.1
) -> float:
    """warmup + cosine 衰减到 floor_frac*base_lr（对每个 param_group 同比例缩放）。"""
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def set_lrs(optimizers, total: int, step: int, base_scale: float, warmup_frac: float = 0.1) -> None:
    """按 base_scale 统一缩放所有 param_group 的初始 lr。"""
    for opt in optimizers:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, g["initial_lr"] * base_scale, warmup_frac)


def tag_initial_lrs(optimizers) -> None:
    for opt in optimizers:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]


# ------------------------------------------------------------------ 数据 ----


def load_tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(TOK_PATH))


def eos_id(tok) -> int:
    return tok.token_to_id(EOS)


def records_to_ids(records: list[str], tok) -> torch.Tensor:
    """把 EOS 分隔的记录列表编码成 token id 流（每条尾部补 eos）。"""
    eos = eos_id(tok)
    ids: list[int] = []
    for r in records:
        r = r.strip("\n")
        if not r.strip():
            continue
        ids.extend(tok.encode(r, add_special_tokens=False).ids)
        ids.append(eos)
    return torch.tensor(ids, dtype=torch.long)


def ensure_stream(ids: torch.Tensor, block_size: int) -> torch.Tensor:
    """token 流太短时循环拼接，保证能取到 block_size 窗口（玩具语料的兜底）。"""
    if len(ids) >= block_size + 2:
        return ids
    reps = (block_size + 2) // max(len(ids), 1) + 1
    return ids.repeat(reps)


def split_records(text: str, val_frac: float, seed: int = 1234) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    records = [r for r in text.split(EOS) if r.strip()]
    rng.shuffle(records)
    n_val = max(1, int(len(records) * val_frac))
    return records[n_val:], records[:n_val]  # train, val


def get_batch(
    ids: torch.Tensor, batch_size: int, block_size: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(ids) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([ids[s : s + block_size] for s in starts])
    y = torch.stack([ids[s + 1 : s + block_size + 1] for s in starts])
    return x, y


@torch.no_grad()
def evaluate(
    model: CausalLM,
    ids: torch.Tensor,
    batch_size: int,
    block_size: int,
    n_batches: int,
    generator: torch.Generator,
) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(ids, batch_size, block_size, generator)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ------------------------------------------------------------------ 生成 ----


@torch.no_grad()
def generate(
    model: CausalLM,
    idx: torch.Tensor,
    max_new: int,
    eos: int,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    use_cache: bool = True,
    seed: int | None = None,
) -> tuple[torch.Tensor, int]:
    """自回归生成。use_cache=True 走 KV cache 单步前向；结果与 naive 路径一致。"""
    model.eval()
    gen = None
    if temperature > 0 and seed is not None:
        gen = torch.Generator().manual_seed(seed)
    cache = None
    if use_cache:
        # cache 长度按本次生成实际需要分配（prompt + max_new），不受训练时 block_size 限制
        cache = KVCache(
            model.cfg.n_layers,
            model.cfg.n_kv_heads,
            model.cfg.d_model // model.cfg.n_heads,
            idx.size(1) + max_new,
            batch=idx.size(0),
        )
        _, _ = model(idx[:, :-1], kv_cache=cache)  # prefill
    n_new = 0
    for _ in range(max_new):
        if use_cache:
            # decode：新 token 的 RoPE 位置 = cache 当前位置（update 前）
            logits, _ = model(idx[:, -1:], kv_cache=cache, offset=cache.pos)
        else:
            logits, _ = model(idx)
        logits = logits[:, -1].float()
        if temperature <= 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs_sorted = torch.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs_sorted, dim=-1)
                remove = cum - probs_sorted > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.empty_like(sorted_logits).scatter(1, sorted_idx, sorted_logits)
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=gen)
        idx = torch.cat([idx, nxt], dim=1)
        n_new += 1
        if (nxt == eos).all():
            break
    model.train()
    return idx, n_new


# ------------------------------------------------------------------ 画图 ----


def plot_loss(
    history: list[dict],
    val_history: list[dict],
    out_path: Path,
    title: str,
    ylabel: str = "cross entropy",
) -> bool:
    """画 train/val loss 曲线；matplotlib 缺失时静默跳过。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    steps = [h["step"] for h in history]
    losses = [h["loss"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.plot(steps, losses, lw=0.8, alpha=0.35, color="tab:blue", label="train loss (step)")
    if len(losses) >= 10:
        k = max(1, len(losses) // 50)
        smooth = [
            sum(losses[max(0, i - k) : i + 1]) / len(losses[max(0, i - k) : i + 1])
            for i in range(len(losses))
        ]
        ax.plot(steps, smooth, lw=1.6, color="tab:blue", label="train loss (smooth)")
    if val_history:
        ax.plot(
            [v["step"] for v in val_history],
            [v["val_loss"] for v in val_history],
            "o-",
            lw=1.4,
            ms=3,
            color="tab:orange",
            label="val loss",
        )
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def avg(xs) -> float:
    return sum(xs) / len(xs) if xs else float("nan")
