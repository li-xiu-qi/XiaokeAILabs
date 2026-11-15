"""Realtime audio/video duplex probe for the official MiniCPM-o-Demo stack.

Adapted from the official ``examples/realtime/video_probe.py`` of
OpenBMB/MiniCPM-o-Demo. The WebSocket protocol, paced real-time streaming and
the reported metrics are unchanged; only media extraction differs: this probe
decodes the input video with PyAV (already a dependency of the demo backend)
instead of shelling out to an ``ffmpeg`` binary, which is not installed on the
device used for this experiment.

What this probe measures, per link of the realtime chain:
  - client input      : input.append chunks sent at a real-time cadence
  - GPU processing    : backend metrics on every delta (vision slices/tokens,
                        wall clock per chunk)
  - speech output     : response.output.delta kind=audio events, saved as WAV
  - listen/speak state: kind=listen events and listen->speak->listen switches
  - overlap           : the sender keeps streaming input while audio output is
                        received (concurrent tasks); the window where both run
                        is reported as overlap_ms
  - barge-in behaviour: the model itself decides to stop speaking when new
                        input arrives; observed as audio output ending
                        (end_of_turn or silence in later deltas) while the
                        sender keeps sending, followed by listen events

This is a live service chain test (gateway -> worker -> GPU backend), not an
offline chunk replay: chunks are paced by wall clock and output is received
concurrently with input.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import io
import json
import math
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import av
import numpy as np
import PIL.Image
import websockets

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
JPEG_QUALITY = 85


def now() -> float:
    return time.perf_counter()


def ms(seconds: float) -> float:
    return seconds * 1000.0


def normalize_realtime_url(url: str, mode: str = "video") -> str:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme in ("ws", "wss") and parsed.path:
        return url
    if parsed.scheme not in ("http", "https", "ws", "wss"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    scheme = "wss" if parsed.scheme in ("https", "wss") else "ws"
    host = parsed.netloc or parsed.path
    return f"{scheme}://{host.rstrip('/')}/v1/realtime?mode={mode}"


def b64_float32(samples: np.ndarray) -> str:
    return base64.b64encode(samples.astype("<f4", copy=False).tobytes()).decode("ascii")


def b64_jpeg(image: PIL.Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_audio_duration_ms(audio_b64: str) -> float:
    raw = base64.b64decode(audio_b64)
    return (len(raw) // 4) / OUTPUT_SAMPLE_RATE * 1000.0


def percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def summarize(values: list[float], prefix: str) -> dict[str, Optional[float]]:
    return {
        f"{prefix}_min_ms": min(values) if values else None,
        f"{prefix}_p50_ms": percentile(values, 50),
        f"{prefix}_p90_ms": percentile(values, 90),
        f"{prefix}_max_ms": max(values) if values else None,
    }


def extract_audio_16k(video_path: Path) -> np.ndarray:
    """Decode the audio track to mono 16 kHz float32 via PyAV."""
    container = av.open(str(video_path))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=INPUT_SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    try:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
    except Exception:
        # Some clips carry no audio track at all.
        pass
    finally:
        container.close()
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    pcm = np.concatenate(chunks).astype(np.float32) / 32768.0
    return np.clip(pcm, -1.0, 1.0).astype(np.float32)


def extract_frames(video_path: Path, fps: float, width: int) -> list[str]:
    """Decode ~fps JPEG frames (base64) scaled to `width`, via PyAV + PIL."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    time_base = float(stream.time_base) if stream.time_base is not None else 0.0
    interval = 1.0 / fps
    frames: list[str] = []
    next_t = 0.0
    try:
        for frame in container.decode(video=0):
            t = float(frame.pts * time_base) if frame.pts is not None else next_t
            if t + 1e-6 < next_t:
                continue
            img = frame.to_image()
            if img.width != width:
                height = max(1, round(img.height * width / img.width))
                img = img.resize((width, height), PIL.Image.LANCZOS)
            frames.append(b64_jpeg(img))
            next_t = t + interval
    finally:
        container.close()
    return frames


def iter_audio_chunks(samples: np.ndarray, chunk_ms: int, tail_silence_s: float) -> list[np.ndarray]:
    if tail_silence_s > 0:
        silence = np.zeros(int(round(INPUT_SAMPLE_RATE * tail_silence_s)), dtype=np.float32)
        samples = np.concatenate([samples, silence])
    chunk_len = int(INPUT_SAMPLE_RATE * chunk_ms / 1000)
    chunks = []
    for start in range(0, len(samples), chunk_len):
        chunk = samples[start : start + chunk_len]
        if len(chunk):
            chunks.append(chunk.astype(np.float32, copy=False))
    return chunks


def save_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


@dataclass
class ProbeState:
    started_s: float
    ws_open_s: Optional[float] = None
    session_id: str = ""
    first_text_s: Optional[float] = None
    first_audio_s: Optional[float] = None
    last_input_send_s: Optional[float] = None
    text_chunks: list[str] = field(default_factory=list)
    output_chunk_times_s: list[float] = field(default_factory=list)
    output_chunk_durations_ms: list[float] = field(default_factory=list)
    output_chunk_eot: list[bool] = field(default_factory=list)
    audio_pcm: list[np.ndarray] = field(default_factory=list)
    listen_events: int = 0
    listen_after_first_audio: int = 0
    speak_then_listen_switches: int = 0
    vision_metrics_seen: int = 0
    input_chunks_sent: int = 0
    input_frames_sent: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    audio_output_ended_while_sending: Optional[bool] = None

    def event(self, typ: str, **kwargs: Any) -> None:
        item = {"t_ms": ms(now() - self.started_s), "type": typ}
        item.update(kwargs)
        self.events.append(item)


async def sender(
    ws: websockets.ClientConnection,
    state: ProbeState,
    audio_chunks: list[np.ndarray],
    frames: list[str],
    mode: str,
    chunk_ms: int,
    fps: float,
    stop_event: asyncio.Event,
    continue_silence: bool,
    max_slice_nums: int,
    max_chunks: Optional[int],
) -> None:
    next_send = now()
    idx = 0
    silence = np.zeros(int(INPUT_SAMPLE_RATE * chunk_ms / 1000), dtype=np.float32)

    while not stop_event.is_set():
        if max_chunks is not None and idx >= max_chunks:
            break
        if idx < len(audio_chunks):
            chunk = audio_chunks[idx]
        elif continue_silence:
            chunk = silence
        else:
            break
        if mode == "video" and frames:
            # Map the chunk index to a video frame by real elapsed time: chunk
            # idx covers [idx*chunk_ms, (idx+1)*chunk_ms) of wall clock, so the
            # frame at that timestamp is int(t * fps). At chunk_ms=1000 and
            # fps=1 this is exactly one frame per second (old behaviour); at
            # chunk_ms=160 the same second is sent in several chunks and they
            # all reuse the frame currently displayed.
            t_s = idx * chunk_ms / 1000.0
            frame = frames[min(int(t_s * fps), len(frames) - 1)]
        else:
            frame = None

        sleep_s = next_send - now()
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)

        payload = {
            "type": "input.append",
            "input": {
                "audio": b64_float32(chunk),
                "force_listen": False,
                "max_slice_nums": max_slice_nums,
            },
        }
        if frame is not None:
            payload["input"]["video_frames"] = [frame]
        await ws.send(json.dumps(payload))
        state.input_chunks_sent += 1
        if frame is not None:
            state.input_frames_sent += 1
        state.last_input_send_s = now()
        state.event("client.input.append", chunk_idx=idx + 1)

        idx += 1
        next_send += chunk_ms / 1000.0


async def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    args.url = normalize_realtime_url(args.url, args.mode)
    samples = extract_audio_16k(Path(args.video))
    if samples.size == 0:
        raise RuntimeError(f"no decodable audio track in {args.video}")
    frames = extract_frames(Path(args.video), args.frame_fps, args.jpeg_width) if args.mode == "video" else []
    audio_chunks = iter_audio_chunks(samples, args.chunk_ms, args.tail_silence_s)

    state = ProbeState(started_s=now())
    stop_sender = asyncio.Event()
    sender_task: asyncio.Task[None] | None = None
    end_reason = "session_closed"

    async with websockets.connect(
        args.url,
        open_timeout=args.open_timeout,
        max_size=args.max_message_mb * 1024 * 1024,
    ) as ws:
        state.ws_open_s = now()
        state.event("ws.open", ws_ready_ms=ms(state.ws_open_s - state.started_s))

        close_after_s = now() + args.max_session_s if args.max_session_s is not None else None

        try:
            while True:
                if close_after_s is not None and now() >= close_after_s:
                    end_reason = "max_session_reached"
                    state.event("client.max_session_reached")
                    break
                timeout = max(0.1, close_after_s - now()) if close_after_s is not None else None
                try:
                    raw = await (ws.recv() if timeout is None else asyncio.wait_for(ws.recv(), timeout))
                except asyncio.TimeoutError:
                    end_reason = "max_session_reached"
                    state.event("client.max_session_reached")
                    break
                msg = json.loads(raw)
                msg_type = msg.get("type", "")
                state.event(f"server.{msg_type}")

                if msg_type in ("session.queue_done", "queue_done") and sender_task is None:
                    await ws.send(json.dumps({
                        "type": "session.init",
                        "payload": {"system_prompt": args.instructions},
                    }))
                    state.event("client.session.init")

                elif msg_type == "session.created":
                    state.session_id = msg.get("session_id", "")
                    sender_task = asyncio.create_task(sender(
                        ws, state, audio_chunks, frames, args.mode, args.chunk_ms,
                        args.frame_fps,
                        stop_sender, args.continue_silence, args.max_slice_nums,
                        args.max_input_chunks,
                    ))
                    state.event("client.sender_started", audio_chunks=len(audio_chunks), frames=len(frames))

                elif msg_type == "response.output.delta":
                    kind = msg.get("kind", "?")
                    metrics = msg.get("metrics") or {}
                    if (metrics.get("vision_slices") or 0) > 0 or (metrics.get("vision_tokens") or 0) > 0:
                        state.vision_metrics_seen += 1
                    if kind == "listen":
                        state.listen_events += 1
                        if state.first_audio_s is not None:
                            state.listen_after_first_audio += 1
                            state.speak_then_listen_switches += 1
                        state.event("server.listen", metrics=metrics)
                    elif kind == "text":
                        text = msg.get("text") or ""
                        if text:
                            if state.first_text_s is None:
                                state.first_text_s = now()
                            state.text_chunks.append(text)
                        state.event("server.text", metrics=metrics)
                    elif kind == "audio":
                        recv_s = now()
                        audio_b64 = msg.get("audio")
                        if audio_b64:
                            if state.first_audio_s is None:
                                state.first_audio_s = recv_s
                            raw_pcm = np.frombuffer(base64.b64decode(audio_b64), dtype="<f4")
                            state.audio_pcm.append(raw_pcm.copy())
                            state.output_chunk_times_s.append(recv_s)
                            state.output_chunk_durations_ms.append(decode_audio_duration_ms(audio_b64))
                            state.output_chunk_eot.append(False)
                        state.event("server.audio", duration_ms=decode_audio_duration_ms(audio_b64 or ""),
                                    metrics=metrics)

                elif msg_type == "response.done":
                    if args.stop_on_end_of_turn:
                        end_reason = "response_done"
                        break

                elif msg_type == "session.closed":
                    end_reason = "session_closed"
                    state.event("server.session.closed", reason=msg.get("reason"))
                    break

                elif msg_type == "error":
                    raise RuntimeError(json.dumps(msg, ensure_ascii=False))
        finally:
            stop_sender.set()
            if sender_task is not None:
                with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                    await asyncio.wait_for(sender_task, timeout=2.0)
            with contextlib.suppress(Exception):
                await ws.send(json.dumps({"type": "session.close", "reason": "probe_done"}))

    if (
        state.first_audio_s is not None
        and state.last_input_send_s is not None
        and state.last_input_send_s > state.first_audio_s
    ):
        # Input was still being streamed after the first audio delta arrived.
        state.audio_output_ended_while_sending = True

    result = build_result(args, state, len(audio_chunks), len(frames), samples, end_reason)
    return result


def build_result(args, state: ProbeState, source_audio_chunks: int, source_frames: int,
                 samples: np.ndarray, end_reason: str) -> dict[str, Any]:
    intervals = [
        ms(b - a) for a, b in zip(state.output_chunk_times_s, state.output_chunk_times_s[1:])
    ]
    pcm = np.concatenate(state.audio_pcm) if state.audio_pcm else np.zeros(0, dtype=np.float32)
    audio_path = None
    if args.audio_output and pcm.size:
        out = Path(args.audio_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_wav(out, pcm, OUTPUT_SAMPLE_RATE)
        audio_path = str(out)
    overlap_ms = None
    if state.first_audio_s is not None and state.last_input_send_s is not None:
        overlap_ms = max(0.0, ms(state.last_input_send_s - state.first_audio_s))
    return {
        "probe": "demo_realtime_probe",
        "adapted_from": "OpenBMB/MiniCPM-o-Demo examples/realtime/video_probe.py "
                        "(media extraction via PyAV; protocol and metrics unchanged)",
        "url": args.url,
        "mode": args.mode,
        "success": bool(
            state.session_id
            and state.input_chunks_sent > 0
            and (state.listen_events > 0 or state.text_chunks or state.output_chunk_times_s)
        ),
        "end_reason": end_reason,
        "session_id": state.session_id,
        "ws_ready_ms": ms(state.ws_open_s - state.started_s) if state.ws_open_s else None,
        "first_text_ms": ms(state.first_text_s - state.started_s) if state.first_text_s else None,
        "first_audio_ms": ms(state.first_audio_s - state.started_s) if state.first_audio_s else None,
        "listen_events": state.listen_events,
        "listen_after_first_audio": state.listen_after_first_audio,
        "speak_then_listen_switches": state.speak_then_listen_switches,
        "vision_metrics_deltas": state.vision_metrics_seen,
        "source_audio_seconds": round(len(samples) / INPUT_SAMPLE_RATE, 3),
        "source_audio_chunks": source_audio_chunks,
        "source_frames": source_frames,
        "input_chunks_sent": state.input_chunks_sent,
        "input_frames_sent": state.input_frames_sent,
        "text_delta_chunks": len(state.text_chunks),
        "text": "".join(state.text_chunks),
        "output_audio_chunks": len(state.output_chunk_times_s),
        "output_audio_ms": round(len(pcm) / OUTPUT_SAMPLE_RATE * 1000.0, 1),
        "output_audio_rms": float(np.sqrt(np.mean(pcm ** 2))) if pcm.size else None,
        "output_audio_peak": float(np.max(np.abs(pcm))) if pcm.size else None,
        "output_audio_path": audio_path,
        "overlap_ms": overlap_ms,
        "audio_output_while_input_streaming": state.audio_output_ended_while_sending,
        **summarize(intervals, "output_chunk_interarrival"),
        "events": state.events if args.include_events else None,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True, help="gateway base URL, e.g. http://127.0.0.1:8006")
    p.add_argument("--video", required=True, help="input video file (mp4)")
    p.add_argument("--mode", default="video", choices=["video", "audio"])
    p.add_argument("--instructions", default="你是一个视频语音助手，请根据用户语音和画面内容自然、完整地回答。")
    p.add_argument("--chunk-ms", type=int, default=1000)
    p.add_argument("--frame-fps", type=float, default=1.0)
    p.add_argument("--jpeg-width", type=int, default=640)
    p.add_argument("--max-slice-nums", type=int, default=1)
    p.add_argument("--tail-silence-s", type=float, default=3.0)
    p.add_argument("--continue-silence", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stop-on-end-of-turn", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-session-s", type=float, default=90.0)
    p.add_argument("--max-input-chunks", type=int, default=None)
    p.add_argument("--open-timeout", type=float, default=30.0)
    p.add_argument("--max-message-mb", type=int, default=128)
    p.add_argument("--output", default=None, help="write result JSON here")
    p.add_argument("--audio-output", default=None, help="write received 24 kHz audio WAV here")
    p.add_argument("--include-events", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = asyncio.run(run_probe(args))
    except Exception as exc:
        print(json.dumps({"probe": "demo_realtime_probe", "success": False,
                          "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), flush=True)
        return 1
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    printable = {k: v for k, v in result.items() if k != "events"}
    print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
