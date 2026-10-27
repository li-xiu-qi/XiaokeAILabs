#!/usr/bin/env python3
"""MiniCPM-OCR-Jev 打分服务（transformers 后端，替代官方 serve/app.py 的 vLLM 后端）。

对外契约与官方 app.py 的 `/v1/rm/score_best`、`/v1/rm/score_best_batch` 一致，
MiniCPM-OCR 1.0 SDK 的 `RMClient` 可以直接把 `rm_url` 指过来。

链路：候选文本 → 官方渲染器出图（playwright 1.62.0 + Chromium）
      → Jev（Qwen3_5 主干 + score.weight）打两分类 logits → softmax[1]。

启动：
    ~/xinfer-env/bin/python jev_server.py --port 18501
"""

from __future__ import annotations

import base64
import io
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import requests
from pydantic import BaseModel, Field, model_validator
from PIL import Image

JEV_REPO = Path(os.environ.get("JEV_REPO", str(Path.home() / "minicpm-ocr/repo/scripts/minicpm-ocr-jev")))
sys.path.insert(0, str(JEV_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rm_render import CATEGORIES, Candidate, render_candidates  # noqa: E402

from jev_score import JevScorer, build_prompt, _assert_prompt_unchanged  # noqa: E402

INPUT_FORMS = ("binary", "ternary")
MAX_ITEMS = 8
THRESHOLD = 0.5

#: 渲染器跑在单独的 venv 里（playwright 1.62.0 + Pillow 12.3.0 的固定版），
#: 和打分用的环境刻意分开：一次 pip install -U 不会把 Pillow 推走而静默改分。
RENDER_PYTHON = os.environ.get(
    "RM_RENDER_PYTHON", str(Path.home() / "minicpm-ocr/jev/venv-render/bin/python")
)

UI_PATH = Path(__file__).resolve().parent / "jev_ui.html"

#: OCR 由另一个模型做（MiniCPM-OCR 1.0）。Jev 自己不会识别，只判一致性。
OCR_URL = os.environ.get("OCR_URL", "http://127.0.0.1:18499/v1/chat/completions")
OCR_MODEL = os.environ.get("OCR_MODEL_DIR", str(Path.home() / "minicpm-ocr/models/MiniCPM-OCR-1.0"))
#: 官方 SDK 的 prompt map：不同内容类型用不同指令与 token 上限。
OCR_PROMPTS = {
    "text": ("Text Recognition:", 4096),
    "formula": ("Formula Recognition:", 2048),
    "table": ("Table Recognition:", 8192),
}

app = FastAPI(title="MiniCPM-OCR-Jev (transformers backend)")
_scorer: JevScorer | None = None
_MODEL_DIR = ""


# ── 请求/响应模型（与官方 app.py 字段对齐） ─────────────────────────────────

class ScoreBestRequest(BaseModel):
    input_form: str = "binary"
    case_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    image: str = Field(..., description="原块图 base64（不带 data URL 前缀）")
    candidates: list[str] = Field(..., min_length=1)
    category: str = "text"
    render_images: Optional[list[str]] = None
    threshold: Optional[float] = None

    @model_validator(mode="after")
    def check(self) -> "ScoreBestRequest":
        if self.input_form not in INPUT_FORMS:
            raise ValueError(f"input_form must be one of {INPUT_FORMS}")
        if self.category not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}")
        if self.render_images is not None and len(self.render_images) != len(self.candidates):
            raise ValueError("render_images must match candidates length")
        return self


class ScoreBestResponse(BaseModel):
    best_index: int
    best_text: str
    scores: list[float]
    verdicts: list[str]
    stage_sec: dict[str, float] = {}
    batch_size: int = 0


class ScoreBestBatchRequest(BaseModel):
    input_form: str = "binary"
    items: list[ScoreBestRequest] = Field(..., min_length=1)


class ScoreBestBatchResponse(BaseModel):
    results: list[ScoreBestResponse]
    stage_sec: dict[str, float] = {}
    batch_size: int = 0


class OCRRequest(BaseModel):
    image: str
    category: str = "text"


# ── 单块打分 ────────────────────────────────────────────────────────────────

def _score_one_block(req: ScoreBestRequest, include_renders: bool = False):
    """返回 (ScoreBestResponse, 渲染图 base64 列表或 None)。

    include_renders 只有网页界面用，官方契约的 /v1/rm/* 不带这个字段。
    """
    work = Path(tempfile.mkdtemp(prefix="jev_req_"))
    stage: dict[str, float] = {}
    try:
        t0 = time.time()
        try:
            img = Image.open(io.BytesIO(base64.b64decode(req.image, validate=False))).convert("RGB")
        except Exception as err:
            raise HTTPException(400, f"image is not decodable: {err}") from err
        original = work / "original.png"
        img.save(original)

        cands = [
            Candidate(case_id=f"{req.case_id}__{i}", image_path=str(original),
                      text=text, category=req.category)
            for i, text in enumerate(req.candidates)
        ]
        provided = req.render_images or [None] * len(cands)
        renders = []
        pending = []
        for cand, given in zip(cands, provided):
            if given:
                p = work / f"render_{cand.case_id}.png"
                p.write_bytes(base64.b64decode(given, validate=False))
                given_path = p
            else:
                given_path = None
            renders.append(given_path)
            pending.append(given_path is None)

        to_render = [c for c, need in zip(cands, pending) if need]
        if to_render:
            rendered = render_candidates(to_render, work_dir=work, precheck=True,
                                         browser_concurrency=4, python_bin=RENDER_PYTHON)
            it = iter(rendered)
            for i, need in enumerate(pending):
                if need:
                    r = next(it)
                    renders[i] = Path(r.render_image_path) if (r.ok and r.render_image_path) else None
                    if renders[i] is None:
                        stage[f"render_fail_{i}"] = 1.0
        stage["render_sec"] = round(time.time() - t0, 3)

        t1 = time.time()
        scores: list[float] = []
        verdicts: list[str] = []
        for cand, render_path in zip(cands, renders):
            if render_path is None or not render_path.is_file():
                scores.append(0.0)
                verdicts.append("render_error")
                continue
            prob, _ = _scorer._forward_one(original, render_path, cand.text)
            scores.append(prob)
            verdicts.append("scored")
        stage["forward_sec"] = round(time.time() - t1, 3)
        stage["forward_rows"] = float(sum(1 for v in verdicts if v == "scored"))

        ok = [i for i, v in enumerate(verdicts) if v == "scored"]
        best = max(ok, key=lambda i: scores[i]) if ok else 0
        renders_b64 = None
        if include_renders:
            renders_b64 = [
                base64.b64encode(p.read_bytes()).decode() if (p and p.is_file()) else None
                for p in renders
            ]
        return ScoreBestResponse(
            best_index=best, best_text=req.candidates[best],
            scores=scores, verdicts=verdicts, stage_sec=stage,
            batch_size=int(stage.get("forward_rows", 0)),
        ), renders_b64
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ── 路由 ────────────────────────────────────────────────────────────────────

@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": _scorer is not None,
        "backend": "transformers",
        "prompt_md5": _assert_prompt_unchanged(),
        "model": _MODEL_DIR or None,
    }


@app.post("/v1/rm/score_best", response_model=ScoreBestResponse)
def score_best(req: ScoreBestRequest) -> ScoreBestResponse:
    if len(req.candidates) > MAX_ITEMS:
        raise HTTPException(400, f"at most {MAX_ITEMS} candidates per request")
    return _score_one_block(req)[0]


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """网页界面。同名目录下 jev_ui.html，改版不用重启服务（每次请求现读）。"""
    if not UI_PATH.is_file():
        return HTMLResponse("<h1>jev_ui.html 未找到</h1>", status_code=500)
    return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))


@app.post("/ui/score")
def ui_score(req: ScoreBestRequest) -> JSONResponse:
    """网页专用：在官方契约的响应上多返回一份渲染图（base64），供人看渲染是否忠实。"""
    if len(req.candidates) > MAX_ITEMS:
        raise HTTPException(400, f"最多 {MAX_ITEMS} 条候选")
    resp, renders = _score_one_block(req, include_renders=True)
    payload = resp.model_dump()
    payload["renders"] = renders
    return JSONResponse(payload)


@app.post("/ui/ocr")
def ui_ocr(req: OCRRequest) -> JSONResponse:
    """网页专用：把原图交给 1.0 识别成文字，省得手动粘贴。"""
    prompt, max_tokens = OCR_PROMPTS.get(req.category, OCR_PROMPTS["text"])
    payload = {
        "model": OCR_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{req.image}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.01,
        "top_p": 0.001,
        "generation_config": '{"top_k": 1, "repetition_penalty": 1.0, "no_repeat_ngram_size": 100}',
    }
    t0 = time.time()
    try:
        resp = requests.post(OCR_URL, json=payload, timeout=300)
    except requests.RequestException as err:
        raise HTTPException(502, f"识别服务不可达：{err}") from err
    if resp.status_code != 200:
        raise HTTPException(502, f"识别服务返回 {resp.status_code}：{resp.text[:300]}")
    try:
        text = resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as err:
        raise HTTPException(502, f"识别结果无法解析：{resp.text[:300]}") from err
    return JSONResponse({"text": text, "seconds": round(time.time() - t0, 2)})


@app.post("/v1/rm/score_best_batch", response_model=ScoreBestBatchResponse)
def score_best_batch(req: ScoreBestBatchRequest) -> ScoreBestBatchResponse:
    total = sum(len(sub.candidates) for sub in req.items)
    if total > MAX_ITEMS:
        raise HTTPException(400, f"at most {MAX_ITEMS} candidates per request, got {total}")
    t0 = time.time()
    results = [_score_one_block(sub)[0] for sub in req.items]
    return ScoreBestBatchResponse(results=results,
                                  stage_sec={"total_sec": round(time.time() - t0, 3)},
                                  batch_size=total)


def main() -> int:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0",
                    help="默认对外监听；只想本机访问时传 127.0.0.1")
    ap.add_argument("--port", type=int, default=18501)
    ap.add_argument("--model", default=os.environ.get("JEV_MODEL_DIR", str(Path.home() / "minicpm-ocr/models/MiniCPM-OCR-Jev")))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    global _scorer, _MODEL_DIR
    build_prompt("预热")
    print(f"prompt md5 校验通过: {_assert_prompt_unchanged()}")
    _MODEL_DIR = args.model
    _scorer = JevScorer(args.model, device=args.device)
    print(f"Jev 加载完成 {_scorer.load_seconds:.1f}s")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
