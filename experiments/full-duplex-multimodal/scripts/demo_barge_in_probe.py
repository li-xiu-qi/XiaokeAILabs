"""Replay two recorded audio/video segments through MiniCPM-o's realtime demo.

The first clip starts a response. Once the client receives the first assistant
audio delta, it injects speech from the second clip while keeping the video
stream active. This is a repeatable protocol-level interruption probe, not a
live microphone, playback, or human-perceived barge-in test.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import io
import json
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
INTERRUPTION_STOP_THRESHOLD_MS = 2000


def now() -> float:
    return time.perf_counter()


def evaluate_acceptance(
    session_ok: bool,
    listen_ok: bool,
    last_audio_after_ms: float | None,
) -> dict[str, Any]:
    stop_latency_ok = bool(
        last_audio_after_ms is not None
        and last_audio_after_ms <= INTERRUPTION_STOP_THRESHOLD_MS
    )
    failures = []
    if not session_ok:
        failures.append("session did not provide all required input and pre-interruption audio")
    if not listen_ok:
        failures.append("no listen event was observed after interruption input began")
    if not stop_latency_ok:
        failures.append(
            "last audio delta was absent or later than the "
            f"{INTERRUPTION_STOP_THRESHOLD_MS} ms stop threshold"
        )
    return {
        "success": session_ok and listen_ok and stop_latency_ok,
        "session_ok": session_ok,
        "listen_after_interruption_ok": listen_ok,
        "stop_latency_ok": stop_latency_ok,
        "stop_threshold_ms": INTERRUPTION_STOP_THRESHOLD_MS,
        "failure_reason": "; ".join(failures) or None,
    }


def b64_float32(samples: np.ndarray) -> str:
    import numpy as np

    return base64.b64encode(samples.astype("<f4", copy=False).tobytes()).decode("ascii")


def b64_jpeg(image: PIL.Image.Image) -> str:
    import PIL.Image

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_media(path: Path, frame_fps: float, jpeg_width: int) -> tuple[np.ndarray, list[str]]:
    """Decode one input clip to mono 16 kHz audio and sampled JPEG frames."""
    import av
    import numpy as np
    import PIL.Image

    container = av.open(str(path))
    audio_chunks: list[np.ndarray] = []
    frames: list[str] = []
    try:
        if container.streams.audio:
            resampler = av.AudioResampler(format="s16", layout="mono", rate=INPUT_SAMPLE_RATE)
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame):
                    audio_chunks.append(out.to_ndarray().reshape(-1))
    finally:
        container.close()
    if audio_chunks:
        audio = np.concatenate(audio_chunks).astype(np.float32) / 32768.0
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    else:
        audio = np.zeros(0, dtype=np.float32)

    container = av.open(str(path))
    try:
        if container.streams.video:
            stream = container.streams.video[0]
            time_base = float(stream.time_base) if stream.time_base is not None else 0.0
            next_t = 0.0
            for frame in container.decode(video=0):
                t = float(frame.pts * time_base) if frame.pts is not None else next_t
                if t + 1e-6 < next_t:
                    continue
                image = frame.to_image()
                if image.width != jpeg_width:
                    height = max(1, round(image.height * jpeg_width / image.width))
                    image = image.resize((jpeg_width, height), PIL.Image.LANCZOS)
                frames.append(b64_jpeg(image))
                next_t = t + 1.0 / frame_fps
    finally:
        container.close()
    if audio.size == 0:
        raise ValueError(f"No decodable audio track in {path}")
    if not frames:
        raise ValueError(f"No decodable video track in {path}")
    return audio, frames


def audio_chunks(audio: np.ndarray, chunk_ms: int, max_seconds: float) -> list[np.ndarray]:
    import numpy as np

    chunk_len = round(INPUT_SAMPLE_RATE * chunk_ms / 1000)
    limit = min(len(audio), round(INPUT_SAMPLE_RATE * max_seconds))
    return [
        audio[i : min(i + chunk_len, limit)].astype(np.float32, copy=False)
        for i in range(0, limit, chunk_len)
    ]


def save_wav(path: Path, samples: np.ndarray) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(OUTPUT_SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def normalize_url(url: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme in ("ws", "wss") and parsed.path:
        return url
    if parsed.scheme not in ("http", "https", "ws", "wss"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    scheme = "wss" if parsed.scheme in ("https", "wss") else "ws"
    host = parsed.netloc or parsed.path
    return f"{scheme}://{host.rstrip('/')}/v1/realtime?mode=video"


@dataclass
class State:
    started: float
    session_id: str = ""
    first_audio_at: float | None = None
    interruption_at: float | None = None
    last_audio_at: float | None = None
    listen_after_interruption: list[float] = field(default_factory=list)
    audio_before_interrupt: int = 0
    audio_after_interrupt: int = 0
    audio_samples: list[np.ndarray] = field(default_factory=list)
    input_initial: int = 0
    input_interrupt: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def event(self, kind: str, **extra: Any) -> None:
        item = {"t_ms": round((now() - self.started) * 1000, 1), "type": kind}
        item.update(extra)
        self.events.append(item)


async def send_clip(
    ws: Any,
    state: State,
    audio: np.ndarray,
    frames: list[str],
    phase: str,
    chunk_ms: int,
    frame_fps: float,
    max_seconds: float,
) -> None:
    import numpy as np

    chunks = audio_chunks(audio, chunk_ms, max_seconds)
    interval = chunk_ms / 1000
    next_send = now()
    silence = np.zeros(round(INPUT_SAMPLE_RATE * chunk_ms / 1000), dtype=np.float32)
    phase_start = now()
    for idx, chunk in enumerate(chunks):
        delay = next_send - now()
        if delay > 0:
            await asyncio.sleep(delay)
        elapsed = now() - phase_start
        frame_index = min(int(elapsed * frame_fps), len(frames) - 1)
        await ws.send(
            json.dumps(
                {
                    "type": "input.append",
                    "input": {
                        "audio": b64_float32(chunk),
                        "force_listen": False,
                        "max_slice_nums": 1,
                        "video_frames": [frames[frame_index]],
                    },
                }
            )
        )
        if phase == "initial":
            state.input_initial += 1
        else:
            if state.interruption_at is None:
                state.interruption_at = now()
                state.event("client.interruption_started", chunk_ms=chunk_ms)
            state.input_interrupt += 1
        state.event(f"client.input.{phase}", chunk_idx=idx + 1)
        next_send += interval

    # Keep the stream active briefly after the interruption material ends.
    # Silence and the last visible frame let the model produce a listen/output
    # transition without forcing listen mode.
    for idx in range(round(1.0 / interval)):
        delay = next_send - now()
        if delay > 0:
            await asyncio.sleep(delay)
        await ws.send(
            json.dumps(
                {
                    "type": "input.append",
                    "input": {
                        "audio": b64_float32(silence),
                        "force_listen": False,
                        "max_slice_nums": 1,
                        "video_frames": [frames[-1]],
                    },
                }
            )
        )
        state.event(f"client.input.{phase}_tail_silence", chunk_idx=idx + 1)
        next_send += interval


async def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import websockets

    initial_audio, initial_frames = decode_media(
        Path(args.initial_video), args.frame_fps, args.jpeg_width
    )
    interrupt_audio, interrupt_frames = decode_media(
        Path(args.interrupt_video), args.frame_fps, args.jpeg_width
    )
    url = normalize_url(args.url)
    state = State(started=now())
    interrupt_task: asyncio.Task[None] | None = None
    end_reason = "observation_window_elapsed"

    async with websockets.connect(
        url, open_timeout=args.open_timeout, max_size=args.max_message_mb * 1024 * 1024
    ) as ws:
        deadline = now() + args.max_session_s
        while now() < deadline:
            try:
                raw = await asyncio.wait_for(
                    ws.recv(), timeout=min(0.5, max(0.05, deadline - now()))
                )
            except asyncio.TimeoutError:
                if (
                    state.interruption_at is not None
                    and now() - state.interruption_at >= args.observe_after_s
                ):
                    end_reason = "post_interruption_observation_complete"
                    break
                continue
            msg = json.loads(raw)
            kind = msg.get("type", "")
            state.event(f"server.{kind}")
            if kind in ("session.queue_done", "queue_done"):
                await ws.send(
                    json.dumps(
                        {"type": "session.init", "payload": {"system_prompt": args.instructions}}
                    )
                )
                state.event("client.session.init")
            elif kind == "session.created":
                state.session_id = msg.get("session_id", "")
                await send_clip(
                    ws,
                    state,
                    initial_audio,
                    initial_frames,
                    "initial",
                    args.chunk_ms,
                    args.frame_fps,
                    args.initial_seconds,
                )
                state.event("client.initial_clip_complete")
            elif kind == "response.output.delta":
                delta_kind = msg.get("kind")
                if delta_kind == "listen":
                    if state.interruption_at is not None:
                        t = round((now() - state.started) * 1000, 1)
                        state.listen_after_interruption.append(t)
                    state.event("server.listen")
                elif delta_kind == "audio" and msg.get("audio"):
                    t = now()
                    if state.first_audio_at is None:
                        state.first_audio_at = t
                        state.event("server.first_audio")
                        # The first assistant audio delta is the reproducible
                        # trigger; the interrupting clip is streamed in real time.
                        interrupt_task = asyncio.create_task(
                            send_clip(
                                ws,
                                state,
                                interrupt_audio,
                                interrupt_frames,
                                "interrupt",
                                args.chunk_ms,
                                args.frame_fps,
                                args.interrupt_seconds,
                            )
                        )
                    state.last_audio_at = t
                    if state.interruption_at is None:
                        state.audio_before_interrupt += 1
                    else:
                        state.audio_after_interrupt += 1
                    pcm = np.frombuffer(base64.b64decode(msg["audio"]), dtype="<f4").copy()
                    state.audio_samples.append(pcm)
                    state.event("server.audio_delta", audio_samples=len(pcm))
            elif kind == "session.closed":
                end_reason = "session_closed"
                break
            elif kind == "error":
                raise RuntimeError(json.dumps(msg, ensure_ascii=False))
            # Ignore response.done: keep the socket open to observe the second
            # input and subsequent listen/audio deltas in the same session.

        with contextlib.suppress(Exception):
            await ws.send(json.dumps({"type": "session.close", "reason": "barge_probe_done"}))

    if interrupt_task is not None:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(interrupt_task, timeout=2.0)
    output = (
        np.concatenate(state.audio_samples)
        if state.audio_samples
        else np.zeros(0, dtype=np.float32)
    )
    audio_path = None
    if args.audio_output and output.size:
        save_wav(Path(args.audio_output), output)
        audio_path = args.audio_output
    interruption_ms = None
    last_audio_after_ms = None
    if state.interruption_at is not None:
        interruption_ms = round((state.interruption_at - state.started) * 1000, 1)
        if state.last_audio_at is not None and state.last_audio_at >= state.interruption_at:
            last_audio_after_ms = round((state.last_audio_at - state.interruption_at) * 1000, 1)
    session_ok = bool(
        state.session_id
        and state.input_initial
        and state.input_interrupt
        and state.first_audio_at
    )
    acceptance = evaluate_acceptance(
        session_ok=session_ok,
        listen_ok=bool(state.listen_after_interruption),
        last_audio_after_ms=last_audio_after_ms,
    )
    return {
        "probe": "demo_barge_in_probe",
        **acceptance,
        "result_scope": "recorded-media protocol events; not live playback or human-perceived latency",
        "mode": "video",
        "client_chunk_ms": args.chunk_ms,
        "session_id": state.session_id,
        "end_reason": end_reason,
        "initial_input_chunks": state.input_initial,
        "interrupt_input_chunks": state.input_interrupt,
        "assistant_audio_chunks_before_interrupt": state.audio_before_interrupt,
        "assistant_audio_chunks_after_interrupt_started": state.audio_after_interrupt,
        "first_assistant_audio_ms": (
            round((state.first_audio_at - state.started) * 1000, 1)
            if state.first_audio_at
            else None
        ),
        "interruption_started_ms": interruption_ms,
        "last_assistant_audio_delta_after_interrupt_ms": last_audio_after_ms,
        "listen_events_after_interrupt_ms": (
            [round(t - interruption_ms, 1) for t in state.listen_after_interruption]
            if interruption_ms is not None
            else []
        ),
        "output_audio_ms": round(len(output) / OUTPUT_SAMPLE_RATE * 1000, 1),
        "output_audio_rms": float(np.sqrt(np.mean(output**2))) if output.size else None,
        "output_audio_path": audio_path,
        "events": state.events,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True, help="Demo gateway base URL; video mode is used")
    p.add_argument(
        "--initial-video", required=True, help="recording containing the initial spoken request"
    )
    p.add_argument(
        "--interrupt-video",
        required=True,
        help="recording whose audio is injected after first model speech",
    )
    p.add_argument(
        "--instructions", default="请根据用户语音和画面自然回答，并在用户继续说话时保持倾听。"
    )
    p.add_argument("--chunk-ms", type=int, default=160)
    p.add_argument("--frame-fps", type=float, default=1.0)
    p.add_argument("--jpeg-width", type=int, default=640)
    p.add_argument("--initial-seconds", type=float, default=4.0)
    p.add_argument("--interrupt-seconds", type=float, default=4.0)
    p.add_argument("--observe-after-s", type=float, default=8.0)
    p.add_argument("--max-session-s", type=float, default=45.0)
    p.add_argument("--open-timeout", type=float, default=30.0)
    p.add_argument("--max-message-mb", type=int, default=128)
    p.add_argument("--output", help="write metrics JSON here")
    p.add_argument("--audio-output", help="write received assistant audio WAV here")
    args = p.parse_args()
    if not 1 <= args.chunk_ms <= 1000:
        p.error("--chunk-ms must be in 1..1000")
    if args.initial_seconds <= 0 or args.interrupt_seconds <= 0:
        p.error("clip durations must be positive")
    if args.observe_after_s <= 0 or args.max_session_s <= 0:
        p.error("observation and session caps must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        result = asyncio.run(run(args))
    except Exception as exc:
        result = {
            "probe": "demo_barge_in_probe",
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
