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

运行契约（源码与运行产物分离）：
  - 输出写 `XL_OUT_DIR`（`out_dir()`），数据读 `XL_DATA_DIR`（`data_dir()`），
    运行标识取 `XL_RUN_ID`（`run_id()`）。
  - 默认可在本机开箱运行：未显式指定时回退到实验目录下被 `.gitignore` 忽略的
    `.work/out/<run_id>`、`.work/data`（默认 `run_id="local"`）。
  - 若需强制外部注入路径（如 CI 或受控环境），设置 `XL_REQUIRE_EXTERNAL_ENV=1`，
    未提供上述环境变量时将直接报错退出，拒绝本地回退。

设备与分布式（本次升级新增）：
  - `init_distributed(args.device, args.backend)` 统一解析设备与进程组，torchrun 启动时自动生效
  - DDP 包装/剥离（wrap_ddp / unwrap_model）、rank0 守卫、all_reduce 指标汇总
  - checkpoint 保存时剥离 DDP `module.` 前缀，加载时 `map_location` 兼容任意设备
  - KVCache / RotaryEmbedding / get_batch / evaluate / generate 全部设备对齐，Muon 优化器
    在 CUDA 与 DDP 下无需修改即可工作（Newton-Schulz 在 fp32 累加格上计算）
"""

from __future__ import annotations

import contextlib
import argparse
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# 小模型在多核 Windows 上默认 8 线程反而因争用变慢，实测 4 线程最快；可用环境变量覆盖
torch.set_num_threads(int(os.environ.get("LLM_THREADS", "4")))

ROOT = Path(__file__).resolve().parents[1]
EOS = "<|endoftext|>"
PAD = "<|pad|>"


# ------------------------------------------------------------ 运行契约 ----
#
# 默认将输出写至 XL_OUT_DIR（缺省 .work/out/<run_id>）、数据读自 XL_DATA_DIR
# （缺省 .work/data）、运行标识取 XL_RUN_ID（缺省 local）。
# 设置 XL_REQUIRE_EXTERNAL_ENV=1 时强制要求外部显式注入，禁止本地回退。


def check_available_memory() -> None:
    """仅在显式设置 XL_MIN_AVAILABLE_GIB 时检查 Linux 可用内存。"""
    raw = os.environ.get("XL_MIN_AVAILABLE_GIB")
    if raw is None:
        return
    try:
        minimum = float(raw)
    except ValueError as exc:
        raise RuntimeError("XL_MIN_AVAILABLE_GIB 必须是大于零的有限数字") from exc
    if not math.isfinite(minimum) or minimum <= 0:
        raise RuntimeError("XL_MIN_AVAILABLE_GIB 必须是大于零的有限数字")
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise RuntimeError("启用内存门槛时 WORLD_SIZE 必须为 1") from exc
    if world_size != 1:
        raise RuntimeError("启用内存门槛时仅支持单进程（WORLD_SIZE=1）")
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            line = next((line for line in f if line.startswith("MemAvailable:")), None)
        if line is None:
            raise ValueError("缺少 MemAvailable")
        fields = line.split()
        if len(fields) != 3 or fields[2] != "kB":
            raise ValueError("MemAvailable 格式无效")
        available_kib = int(fields[1])
        if available_kib < 0:
            raise ValueError("MemAvailable 为负值")
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(f"无法读取可用内存，已停止训练: {exc}") from exc
    available_gib = available_kib / (1024 * 1024)
    if available_gib < minimum:
        raise RuntimeError(
            f"可用内存 {available_gib:.2f} GiB 低于门槛 {minimum:.2f} GiB，已停止训练"
        )


def _require_external_env() -> bool:
    return os.environ.get("XL_REQUIRE_EXTERNAL_ENV", "").strip().lower() in {"1", "true", "yes"}


def _resolve_env(name: str, default_factory, *, msg_hint: str) -> str:
    v = os.environ.get(name, "").strip()
    if v:
        return v
    if _require_external_env():
        sys.exit(f"{name} 未设置（XL_REQUIRE_EXTERNAL_ENV=1 禁止本地回退）：{msg_hint}")
    return default_factory()


def run_id() -> str:
    """本次运行的标识（XL_RUN_ID），写进每个阶段的指标与 manifest。"""
    return _resolve_env("XL_RUN_ID", lambda: "local", msg_hint="请显式设置 XL_RUN_ID")


def out_dir() -> Path:
    """本次运行的输出目录（XL_OUT_DIR），不存在则创建。"""
    v = _resolve_env(
        "XL_OUT_DIR",
        lambda: str(ROOT / ".work" / "out" / run_id()),
        msg_hint="请显式设置 XL_OUT_DIR",
    )
    p = Path(v)
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir() -> Path:
    """训练数据目录（XL_DATA_DIR），不存在则创建。"""
    v = _resolve_env(
        "XL_DATA_DIR",
        lambda: str(ROOT / ".work" / "data"),
        msg_hint="请显式设置 XL_DATA_DIR",
    )
    p = Path(v)
    p.mkdir(parents=True, exist_ok=True)
    return p


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
        # 与 inv_freq 同设备同 dtype：模型 .to(cuda) 后缓存仍随模块走，运行期扩容也不落 CPU
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
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
        device: torch.device | str | None = None,
    ):
        self.max_len = max_len
        self.k = [
            torch.empty(batch, n_kv_heads, max_len, head_dim, dtype=dtype, device=device)
            for _ in range(n_layers)
        ]
        self.v = [
            torch.empty(batch, n_kv_heads, max_len, head_dim, dtype=dtype, device=device)
            for _ in range(n_layers)
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
    # unwrap：DDP 包装保存时剥离 module. 前缀，单卡加载无需特殊处理
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {"model_state": unwrap_model(model).state_dict(), "config": cfg.to_dict(), **meta}, tmp
    )
    tmp.replace(path)


def save_resume_state(
    path: Path,
    model: CausalLM,
    cfg: ModelConfig,
    optimizers,
    step: int,
    scaler=None,
    history: list | None = None,
    val_history: list | None = None,
    elapsed_seconds: float = 0.0,
    extra: dict | None = None,
    **meta,
) -> None:
    """原子保存完整可续训状态（权重 + 优化器 + GradScaler + step + 曲线 + RNG）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = {
        "model_state": unwrap_model(model).state_dict(),
        "config": cfg.to_dict(),
        "optimizer_states": [opt.state_dict() for opt in optimizers],
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "step": int(step),
        "history": list(history or []),
        "val_history": list(val_history or []),
        "elapsed_seconds": float(elapsed_seconds),
        "rng_py": random.getstate(),
        "rng_torch": torch.get_rng_state(),
        "extra": dict(extra or {}),
        **meta,
    }
    torch.save(payload, tmp)
    tmp.replace(path)


def load_resume_state(
    path: Path,
    model: CausalLM,
    optimizers,
    device: torch.device | str | None = None,
    scaler=None,
) -> tuple[int, list, list, float, dict]:
    """从 resume checkpoint 恢复模型、优化器、scaler 与训练进度。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    target = unwrap_model(model)
    state = ckpt["model_state"]
    try:
        target.load_state_dict(state)
    except Exception:
        target.load_state_dict(_strip_module_prefix(state))
    if device is not None:
        target.to(device)
    opt_states = ckpt.get("optimizer_states") or []
    for opt, ost in zip(optimizers, opt_states):
        opt.load_state_dict(ost)
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return (
        int(ckpt.get("step", 0)),
        list(ckpt.get("history", [])),
        list(ckpt.get("val_history", [])),
        float(ckpt.get("elapsed_seconds", 0.0)),
        ckpt,
    )


def resolve_resume_path(args, ckpt_dir: Path, filename: str) -> Path | None:
    """按 --resume > --resume-auto (checkpoints/<filename>) 解析续训路径。"""
    explicit = getattr(args, "resume", None)
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"指定的 --resume checkpoint 不存在: {p}")
        return p
    if getattr(args, "resume_auto", False):
        cand = ckpt_dir / filename
        if cand.exists():
            return cand
    return None


def resume_training_if_needed(
    args,
    ckpt_dir: Path,
    filename: str,
    model: CausalLM,
    optimizers,
    device: torch.device,
    ctx: DistContext | None,
    scaler=None,
    train_gen: torch.Generator | None = None,
    val_gen: torch.Generator | None = None,
    sample_rng: random.Random | None = None,
    tag: str = "",
) -> tuple[int, list, list, float]:
    """若需续训，加载完整状态并按 rank 决定是否还原局部 RNG。"""
    rpath = resolve_resume_path(args, ckpt_dir, filename)
    if rpath is None:
        return 0, [], [], 0.0
    start_step, history, val_history, elapsed_prev, ckpt = load_resume_state(
        rpath, model, optimizers, device=device, scaler=scaler
    )
    saved_world = ckpt.get("world_size")
    cur_world = ctx.world_size if ctx is not None else 1
    rank = ctx.rank if ctx is not None else 0
    extra = ckpt.get("extra") or {}
    # rank0（或同 world_size 对应 rank）还原 RNG，保持采样流无缝衔接；跨 world_size 新增 rank 保留各自错开种子
    if rank == 0 or saved_world == cur_world:
        if train_gen is not None and isinstance(extra.get("train_gen"), torch.Tensor):
            train_gen.set_state(extra["train_gen"].cpu())
        if val_gen is not None and isinstance(extra.get("val_gen"), torch.Tensor):
            val_gen.set_state(extra["val_gen"].cpu())
        if sample_rng is not None and extra.get("sample_rng") is not None:
            sample_rng.setstate(extra["sample_rng"])
    if is_rank0(ctx):
        print(
            f"[{tag or 'resume'}] 从 {rpath} 恢复: completed_step={start_step}/{args.steps}, "
            f"elapsed={elapsed_prev:.1f}s (saved_world={saved_world} -> cur_world={cur_world})"
        )
    return start_step, history, val_history, elapsed_prev


def checkpoint_step_hook(
    args,
    ckpt_dir: Path,
    filename: str,
    model: CausalLM,
    cfg: ModelConfig,
    optimizers,
    completed_step: int,
    ctx: DistContext | None,
    scaler=None,
    history: list | None = None,
    val_history: list | None = None,
    t0: float = 0.0,
    elapsed_offset: float = 0.0,
    train_gen: torch.Generator | None = None,
    val_gen: torch.Generator | None = None,
    sample_rng: random.Random | None = None,
    stage: str = "",
    tag: str = "",
) -> None:
    """周期性保存可续训快照；配合 --inject-fail-step 支持受控故障演练。"""
    save_every = int(getattr(args, "save_every", 0) or 0)
    inject_step = int(getattr(args, "inject_fail_step", 0) or 0)
    should_save = (save_every > 0 and completed_step % save_every == 0) or (
        inject_step > 0 and completed_step == inject_step
    )
    if should_save:
        if is_rank0(ctx):
            extra = {}
            if train_gen is not None:
                extra["train_gen"] = train_gen.get_state()
            if val_gen is not None:
                extra["val_gen"] = val_gen.get_state()
            if sample_rng is not None:
                extra["sample_rng"] = sample_rng.getstate()
            elapsed = elapsed_offset + (time.time() - t0)
            save_resume_state(
                ckpt_dir / filename,
                model,
                cfg,
                optimizers,
                step=completed_step,
                scaler=scaler,
                history=history,
                val_history=val_history,
                elapsed_seconds=elapsed,
                extra=extra,
                stage=stage,
                world_size=ctx.world_size if ctx is not None else 1,
            )
            print(f"[{tag or 'ckpt'}] 已保存断点 {ckpt_dir / filename} (step={completed_step})")
        if ctx is not None and ctx.enabled:
            import torch.distributed as dist

            dist.barrier()
    if inject_step > 0 and completed_step == inject_step:
        raise RuntimeError(f"[{tag or 'fault'}] 触发故障注入退出 (step={completed_step})")


def reset_peak_memory(device: torch.device | str | None) -> None:
    """重置 CUDA 峰值显存统计（CPU 时无操作）。"""
    dev = torch.device(device) if device is not None else resolve_device("auto")
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(dev)


def memory_stats(device: torch.device | str | None) -> dict:
    """读取当前设备的显存统计（MB）；CPU 返回空指标占位。"""
    dev = torch.device(device) if device is not None else resolve_device("auto")
    if dev.type != "cuda" or not torch.cuda.is_available():
        return {
            "device_type": dev.type,
            "peak_vram_allocated_mb": 0.0,
            "peak_vram_reserved_mb": 0.0,
            "gpu_total_vram_mb": 0.0,
            "vram_util_pct": 0.0,
        }
    alloc = torch.cuda.max_memory_allocated(dev) / (1024**2)
    resv = torch.cuda.max_memory_reserved(dev) / (1024**2)
    total = torch.cuda.get_device_properties(dev).total_memory / (1024**2)
    return {
        "device_type": "cuda",
        "peak_vram_allocated_mb": round(alloc, 2),
        "peak_vram_reserved_mb": round(resv, 2),
        "gpu_total_vram_mb": round(total, 1),
        "vram_util_pct": round(100.0 * resv / max(total, 1e-6), 2),
    }


def tensor_size_mb(t: torch.Tensor | None) -> float:
    if t is None:
        return 0.0
    return round(t.numel() * t.element_size() / (1024**2), 3)


def _strip_module_prefix(state: dict) -> dict:
    return {
        (k[len("module.") :] if k.startswith("module.") else k): v for k, v in state.items()
    }


def load_checkpoint(
    path: Path, map_location=None, device: torch.device | str | None = None
) -> tuple[CausalLM, dict]:
    """加载 checkpoint。map_location 缺省 cpu；device 给定则再搬到目标设备。

    兼容 DDP 产出的权重：优先原样加载，键名带 module. 前缀时自动剥离后重试。
    """
    ckpt = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    model = CausalLM(ModelConfig.from_dict(ckpt["config"]))
    state = ckpt["model_state"]
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(_strip_module_prefix(state))
    if device is not None:
        model.to(device)
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


def load_tokenizer(path: str | Path | None = None):
    from tokenizers import Tokenizer

    tok_path = Path(path) if path else (out_dir() / "tokenizer" / "tokenizer.json")
    return Tokenizer.from_file(str(tok_path))


def token_bytes_from_tokenizer(tok) -> torch.Tensor:
    """ByteLevel 词表中每个字符对应一个原始字节，特殊标记不计字节。"""
    from tokenizers import pre_tokenizers

    alphabet = set(pre_tokenizers.ByteLevel.alphabet())
    special_ids = {tok.token_to_id(EOS), tok.token_to_id(PAD)}
    lengths = []
    for token_id in range(tok.get_vocab_size()):
        if token_id in special_ids:
            lengths.append(0)
            continue
        token = tok.id_to_token(token_id)
        if not token or any(char not in alphabet for char in token):
            raise ValueError(f"token {token_id} 不是受支持的 ByteLevel 词表项")
        lengths.append(len(token))
    return torch.tensor(lengths, dtype=torch.int32)


def load_token_bytes(tok_dir: str | Path | None = None, tok=None) -> torch.Tensor | None:
    """加载字节长度；旧版错误缓存只在内存中校正，不改写原始文件。"""
    base = Path(tok_dir) if tok_dir else (out_dir() / "tokenizer")
    p = base / "token_bytes.pt"
    if not p.exists():
        return None
    saved = torch.load(p, map_location="cpu", weights_only=True)
    expected = token_bytes_from_tokenizer(tok or load_tokenizer(base / "tokenizer.json"))
    if not torch.equal(saved, expected):
        print("[BPB] token_bytes.pt 与 ByteLevel 词表不一致；本次在内存中校正，旧评估结果需重新计算。", file=sys.stderr)
    return expected

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
    ids: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: torch.Generator,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(ids) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([ids[s : s + block_size] for s in starts])
    y = torch.stack([ids[s + 1 : s + block_size + 1] for s in starts])
    if device is not None:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def evaluate(
    model: CausalLM,
    ids: torch.Tensor,
    batch_size: int,
    block_size: int,
    n_batches: int,
    generator: torch.Generator,
    device: torch.device | str | None = None,
) -> float:
    model = unwrap_model(model)
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(ids, batch_size, block_size, generator, device=device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate_bpb_ppl(
    model: CausalLM,
    ids: torch.Tensor,
    token_bytes: torch.Tensor | None,
    batch_size: int = 16,
    block_size: int = 256,
    device: torch.device | str | None = None,
    max_windows: int = 256,
) -> dict:
    """统一测试集评估：在固定 token 流上按连续不重叠窗口计算 CE / BPB / PPL。

    - mean_loss_nats: 所有预测位置的平均交叉熵（nat/token）
    - bpb: 文本预测位置的 total_nll_bits / total_bytes；跨词表比较须对齐原文和评测范围
    - ppl: exp(mean_loss_nats)
    窗口从头顺序切分；不同词表的固定 token 数窗口不一定对应相同原文。
    """
    model = unwrap_model(model)
    model.eval()
    dev = device if device is not None else next(model.parameters()).device
    n_tokens = len(ids)
    if n_tokens < 2 or block_size < 1 or max_windows < 1:
        raise ValueError("BPB 评估至少需要两个 token，block_size 和 max_windows 必须大于零")
    n_win = min(max_windows, max(1, (n_tokens - 1) // block_size))
    total_nll_nats = 0.0
    total_positions = 0
    total_bytes = 0
    total_bits = 0.0
    for w in range(n_win):
        s = w * block_size
        x = ids[s : s + block_size].unsqueeze(0).to(dev)
        y = ids[s + 1 : s + block_size + 1].unsqueeze(0).to(dev)
        logits, _ = model(x)
        logp = torch.log_softmax(logits[:, :y.size(1)].float(), dim=-1)
        tgt = y
        nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (1, T-1) nats
        total_nll_nats += float(nll.sum())
        total_positions += nll.numel()
        if token_bytes is not None:
            tb = token_bytes.to(dev)[tgt.reshape(-1)]
            total_bytes += int(tb.sum())
            # 与分母一致，只累计对应真实文本字节的预测位置。
            total_bits += float(nll.reshape(-1)[tb > 0].sum()) / math.log(2.0)
    model.train()
    mean_nats = total_nll_nats / max(total_positions, 1)
    out = {
        "eval_tokens": total_positions,
        "eval_windows": n_win,
        "mean_loss_nats": round(mean_nats, 6),
        "ppl": round(math.exp(mean_nats), 4),
    }
    if token_bytes is not None and total_bytes > 0:
        out["eval_bytes"] = total_bytes
        out["bpb"] = round(total_bits / total_bytes, 6)
    return out


def build_ids_from_source(
    source: str | Path | None,
    tok,
    max_chars: int | None = None,
    doc_cap: int = 10000,
    eos: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """把外部语料（parquet 文件/目录或 txt）编码为 token id 流，带字符上限。

    与 01 的 iter_texts 同一套截断逻辑（doc_cap + max_chars），因此分词器训练
    与预训练看到的数据口径一致；各 DDP rank 独立编码同一分片，结果逐位一致。
    返回 (ids, stats)。
    """
    import dataset

    if eos is None:
        eos = eos_id(tok)
    ids: list[int] = []
    n_docs = 0
    total_chars = 0
    for doc in dataset.iter_texts(data_source=source, split="train", max_chars=max_chars, doc_cap=doc_cap):
        d = doc.strip("\n")
        if not d.strip():
            continue
        ids.extend(tok.encode(d, add_special_tokens=False).ids)
        ids.append(eos)
        n_docs += 1
        total_chars += len(d)
    stats = {
        "source": str(source) if source else None,
        "docs": n_docs,
        "chars": total_chars,
        "tokens": len(ids),
        "max_chars": max_chars,
        "doc_cap": doc_cap,
    }
    return torch.tensor(ids, dtype=torch.long), stats


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
    model = unwrap_model(model)
    model.eval()
    gen = None
    if temperature > 0 and seed is not None:
        # 采样 Generator 必须与 logits 同设备：CUDA 张量用 CPU generator 会直接报错
        gen = torch.Generator(device=idx.device).manual_seed(seed)
    cache = None
    if use_cache:
        # cache 长度按本次生成实际需要分配（prompt + max_new），不受训练时 block_size 限制
        param = next(model.parameters())
        cache = KVCache(
            model.cfg.n_layers,
            model.cfg.n_kv_heads,
            model.cfg.d_model // model.cfg.n_heads,
            idx.size(1) + max_new,
            batch=idx.size(0),
            dtype=param.dtype,
            device=idx.device,
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


# ------------------------------------------------- 设备与分布式上下文 ----
#
# 统一入口：init_distributed(args.device, args.backend)。torchrun 启动时读环境变量
# WORLD_SIZE / RANK / LOCAL_RANK 自动建组；直接 python 启动则退化为单进程。


def resolve_device(spec: str = "auto") -> torch.device:
    """解析 --device：auto（有卡用卡，没卡 CPU）/ cpu / cuda / cuda:N。"""
    s = (spec or "auto").strip().lower()
    if s in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if s == "cpu":
        return torch.device("cpu")
    if s == "cuda" or s.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"--device {spec} 但当前环境未检测到可用 CUDA 设备")
        if s == "cuda":  # 规范成 cuda:<current>，避免 torch.device('cuda') != cuda:0 的比较坑
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device(s)
    raise ValueError(f"无法解析的 --device 取值: {spec}（可选 auto / cpu / cuda / cuda:N）")


def env_world_size() -> int:
    """torchrun 会注入 WORLD_SIZE；手动拉起时也可用同名环境变量。"""
    for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        v = os.environ.get(key)
        if v:
            try:
                return max(1, int(v))
            except ValueError:
                pass
    return 1


def pick_backend(spec: str, device: torch.device) -> str:
    """解析分布式后端。auto：CUDA 且编译了 NCCL 用 nccl，否则 gloo（Windows 官方
    wheel 未编译 NCCL，CUDA 张量走 gloo 同样正确，只是带宽低一档）。"""
    s = (spec or "auto").strip().lower()
    if s not in ("auto", "nccl", "gloo"):
        raise ValueError(f"无法解析的 --backend 取值: {spec}（可选 auto / nccl / gloo）")
    if s != "auto":
        if s == "nccl" and not torch.distributed.is_nccl_available():
            raise RuntimeError(
                "backend=nccl 不可用（当前 torch 未编译 NCCL，Windows 官方 wheel 常态），请改用 gloo"
            )
        return s
    if device.type == "cuda" and torch.distributed.is_nccl_available():
        return "nccl"
    return "gloo"


@dataclass
class DistContext:
    """单进程 = enabled False；多进程 = DDP 上下文（rank / world / 设备）。"""

    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str = "none"
    device: torch.device = torch.device("cpu")

    def is_rank0(self) -> bool:
        return self.rank == 0


def init_distributed(device: str = "auto", backend: str = "auto") -> DistContext:
    """按环境变量决定是否建组，返回统一上下文。单进程时零开销直通。"""
    dev = resolve_device(device)
    world_env = env_world_size()
    if world_env <= 1:
        return DistContext(False, 0, 1, 0, "none", dev)
    if not torch.distributed.is_available():
        raise RuntimeError("检测到 WORLD_SIZE>1，但当前 torch 不支持 distributed")
    be = pick_backend(backend, dev)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    # LLM_INIT_METHOD：默认 env://（torchrun 标准路径）。个别 Windows wheel 的 TCPStore
    # 被编译成强制 libuv 且未编入 libuv（torchrun agent 直接起不来），可用 file:// 的
    # FileStore 绕过，例如 LLM_INIT_METHOD=file:///C:/temp/xk_store
    init_method = os.environ.get("LLM_INIT_METHOD") or None
    if not torch.distributed.is_initialized():
        if init_method:
            torch.distributed.init_process_group(
                backend=be,
                init_method=init_method,
                rank=int(os.environ.get("RANK", 0)),
                world_size=world_env,
            )
        else:
            torch.distributed.init_process_group(backend=be)
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    if dev.type == "cuda":
        n_dev = torch.cuda.device_count()
        if local >= n_dev:
            # 单卡多进程演示：进程数超过可见 GPU 数时回落共用 cuda:0
            # （标准多卡场景不会触发；真实双卡双进程各自绑定 local_rank）
            if rank == 0:
                print(
                    f"[dist] 警告: world_size={world} 大于可见 GPU 数 {n_dev}，"
                    f"多进程将共用 cuda:0（单卡演示模式）"
                )
            local = 0
        dev = torch.device("cuda", local)
        torch.cuda.set_device(dev)
    ctx = DistContext(True, rank, world, local, be, dev)
    if rank == 0:
        print(f"[dist] 初始化进程组: backend={be}, world_size={world}, device={dev}")
    return ctx


def cleanup_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_rank0(ctx: DistContext | None) -> bool:
    return ctx is None or ctx.rank == 0


def wrap_ddp(model: nn.Module, ctx: DistContext | None, **kw) -> nn.Module:
    """多进程时包 DDP；单进程原样返回（调用方无需分支）。"""
    if ctx is None or not ctx.enabled:
        return model
    kwargs = dict(kw)
    if ctx.device.type == "cuda":
        kwargs["device_ids"] = [ctx.local_rank]
    return nn.parallel.DistributedDataParallel(model, **kwargs)


def unwrap_model(model: nn.Module) -> nn.Module:
    """剥离 DDP/DataParallel 包装，拿回原始 CausalLM（保存与推理都用它）。"""
    return model.module if hasattr(model, "module") else model


def all_reduce_sum(value, ctx: DistContext | None):
    """跨 rank 求和；单进程原样返回。"""
    if ctx is None or not ctx.enabled:
        return value
    t = value.detach().clone().to(ctx.device) if isinstance(value, torch.Tensor) else torch.tensor(
        float(value), device=ctx.device
    )
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_reduce_mean(value, ctx: DistContext | None) -> float:
    """跨 rank 求平均（指标汇总口径），返回 python float。"""
    if ctx is None or not ctx.enabled:
        return float(value)
    t = all_reduce_sum(value, ctx)
    return float(t) / ctx.world_size


def shard_list(items: list, ctx: DistContext | None) -> list:
    """按 rank 交错分片：多卡并行采样/评估时每个 rank 拿不同的子集。"""
    if ctx is None or not ctx.enabled:
        return list(items)
    return list(items)[ctx.rank :: ctx.world_size]


def rank_seed(seed: int, ctx: DistContext | None) -> int:
    """按 rank 偏移种子：保证不同 rank 抽到不同训练 batch（同 rank 可复现）。"""
    return int(seed) + 100_003 * (ctx.rank if ctx is not None else 0)


def batch_generator(seed: int, ctx: DistContext | None) -> torch.Generator:
    return torch.Generator().manual_seed(rank_seed(seed, ctx))


def effective_batch(base: int, scale: float = 1.0, world_size: int = 1) -> int:
    """--scale 按比例放大每卡 batch；多卡时 world_size 由调用方显式传入（不再隐式乘）。"""
    return max(1, int(round(base * (scale if scale else 1.0))))


def dist_info(ctx: DistContext | None, args=None) -> dict:
    """落进 metrics.json 的运行环境信息。"""
    info = {
        "distributed": bool(ctx is not None and ctx.enabled),
        "world_size": ctx.world_size if ctx else 1,
        "rank": ctx.rank if ctx else 0,
        "backend": ctx.backend if ctx else "none",
        "device": str(ctx.device if ctx else resolve_device("auto")),
    }
    if args is not None:
        info["amp"] = getattr(args, "amp", "none")
        info["grad_accum"] = getattr(args, "grad_accum", 1)
        info["scale"] = getattr(args, "scale", 1.0)
    return info


class AmpCtx:
    """混合精度上下文：autocast + （仅 fp16）GradScaler。

    bf16 动态范围与 fp32 一致，不需要 scaler；fp16 仅 CUDA（CPU 上慢且数值不稳，直接报错）。
    """

    def __init__(self, amp: str = "none", device: torch.device | str | None = None):
        a = (amp or "none").strip().lower()
        if a not in ("none", "bf16", "fp16"):
            raise ValueError(f"无法解析的 --amp 取值: {amp}（可选 none / bf16 / fp16）")
        dev = torch.device(device) if device is not None else resolve_device("auto")
        self.amp = a
        self.device = dev
        self.scaler = None
        if a == "fp16":
            if dev.type != "cuda":
                raise RuntimeError("amp=fp16 仅支持 CUDA；CPU 上请用 none 或 bf16")
            self.scaler = torch.amp.GradScaler("cuda")

    @property
    def dtype(self):
        return {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[self.amp]

    def autocast(self):
        if self.amp == "none":
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.dtype)

    def backward(self, loss: torch.Tensor) -> None:
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizers, model: nn.Module | None = None, grad_clip: float | None = None) -> None:
        """clip + optimizer step；fp16 时先 unscale 再 clip。"""
        params = None
        if model is not None:
            params = [p for p in unwrap_model(model).parameters() if p.requires_grad]
        if self.scaler is not None:
            for opt in optimizers:
                self.scaler.unscale_(opt)
            if grad_clip and params:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            for opt in optimizers:
                self.scaler.step(opt)
            self.scaler.update()
        else:
            if grad_clip and params:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            for opt in optimizers:
                opt.step()


def add_device_args(parser: argparse.ArgumentParser, scale_help: str | None = None) -> argparse.ArgumentParser:
    """给 03~08 的 argparse 统一挂上设备 / 分布式 / AMP / 累积 / scale 参数。"""
    parser.add_argument(
        "--device", default="auto", help="运行设备：auto（有卡用卡）/ cpu / cuda / cuda:N"
    )
    parser.add_argument(
        "--amp", default="none", choices=["none", "bf16", "fp16"],
        help="混合精度：none / bf16 / fp16（fp16 仅 CUDA）",
    )
    parser.add_argument(
        "--backend", default="auto", choices=["auto", "nccl", "gloo"],
        help="分布式后端：auto / nccl / gloo（Windows 无 NCCL 时 auto 落 gloo）",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=1, dest="grad_accum",
        help="梯度累积微批数（等效放大 batch，显存不足时用）",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help=scale_help or "规模系数：按比例放大每卡 batch（默认 1.0 保持 CPU 冒烟口径）",
    )
    return parser


def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """挂上模型尺寸超参（默认值即 ModelConfig 缺省，不传时行为不变）。"""
    parser.add_argument("--d-model", type=int, default=192, dest="d_model", help="模型维度 d_model")
    parser.add_argument("--n-layers", type=int, default=4, dest="n_layers", help="Transformer 层数")
    parser.add_argument("--n-heads", type=int, default=6, dest="n_heads", help="注意力头数")
    parser.add_argument("--n-kv-heads", type=int, default=0, dest="n_kv_heads",
                        help="GQA key/value 头数（0 = 等于 n_heads，即普通 MHA）")
    parser.add_argument("--d-ff", type=int, default=768, dest="d_ff", help="MLP 隐层维度")
    parser.add_argument("--block-size", type=int, default=128, dest="block_size",
                        help="训练/评估窗口长度（同时作为 RoPE 预计算长度）")
    return parser


def model_config_from_args(args, vocab_size: int) -> ModelConfig:
    """按命令行覆盖构造 ModelConfig（vocab_size 由分词器决定）。"""
    n_kv = getattr(args, "n_kv_heads", 0) or 0
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=n_kv if n_kv > 0 else args.n_heads,
        d_ff=args.d_ff,
        max_len=args.block_size,
    )


def add_outdir_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """挂上独立实验输出目录与自定义分词器路径（缺省走运行契约的 XL_OUT_DIR）。"""
    parser.add_argument(
        "--out-dir", type=str, default=None, dest="out_dir",
        help="独立实验输出目录（缺省用 XL_OUT_DIR；给定则指标/checkpoint/曲线全部落此目录）",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None, dest="tokenizer",
        help="自定义 tokenizer.json 路径（缺省用 out-dir/tokenizer 下的）",
    )
    return parser


def resolve_out_dir(args) -> Path:
    """解析实验输出目录：--out-dir 优先，否则回落 out_dir()（$XL_OUT_DIR 或 .work/out/<run_id>）。"""
    return Path(args.out_dir) if getattr(args, "out_dir", None) else out_dir()


def resolve_tokenizer_path(args) -> Path:
    """解析 tokenizer.json 路径：--tokenizer > out-dir/tokenizer。"""
    if getattr(args, "tokenizer", None):
        return Path(args.tokenizer)
    return resolve_out_dir(args) / "tokenizer" / "tokenizer.json"


def add_resume_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """挂上断点续训与故障演练参数。"""
    parser.add_argument(
        "--save-every", type=int, default=0, dest="save_every",
        help="每隔多少 step 原子落盘可续训断点（0 = 仅阶段结束保存）",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="显式从指定 *_resume.pt 恢复训练状态",
    )
    parser.add_argument(
        "--resume-auto", action="store_true", dest="resume_auto",
        help="若 checkpoints/<stage>_resume.pt 存在则自动续训",
    )
    parser.add_argument(
        "--inject-fail-step", type=int, default=0, dest="inject_fail_step",
        help="故障演练：到指定 step 保存断点后模拟崩溃退出（0 = 关闭）",
    )
    return parser
