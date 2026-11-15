"""Run the model's official duplex API on a recorded audio/video file.

This measures offline chunk processing, not interactive latency or barge-in.
"""
import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from minicpmo.utils import get_video_frame_audio_segments
from transformers import AutoModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--video', required=True)
    p.add_argument('--reference-audio', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--audio-output', help='optional 24 kHz WAV containing the returned audio chunks')
    p.add_argument('--max-chunks', type=int, default=4)
    args = p.parse_args()
    if args.max_chunks < 1:
        p.error('--max-chunks must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for this experiment')
    model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True,
                                     attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    model.eval().cuda()
    model = model.as_duplex()
    ref_audio, _ = librosa.load(args.reference_audio, sr=16000, mono=True)
    frames, audio, stacked = get_video_frame_audio_segments(
        args.video, stack_frames=1, use_ffmpeg=True, adjust_audio_length=True)
    model.prepare(prefix_system_prompt='Streaming Omni Conversation.',
                  ref_audio=ref_audio, prompt_wav_path=args.reference_audio)
    records = []
    audio_chunks = []
    for i in range(min(len(audio), args.max_chunks)):
        frame_list = []
        if i < len(frames) and frames[i] is not None:
            frame_list.append(frames[i])
        if stacked is not None and i < len(stacked) and stacked[i] is not None:
            frame_list.append(stacked[i])
        started = time.perf_counter()
        model.streaming_prefill(audio_waveform=audio[i], frame_list=frame_list,
                                max_slice_nums=1, batch_vision_feed=False)
        prefill = time.perf_counter()
        result = model.streaming_generate(prompt_wav_path=args.reference_audio,
                                          max_new_speak_tokens_per_chunk=20,
                                          decode_mode='sampling')
        finished = time.perf_counter()
        wave = result.get('audio_waveform')
        if wave is not None:
            if isinstance(wave, torch.Tensor):
                wave = wave.detach().cpu().numpy()
            wave = np.asarray(wave, dtype=np.float32).reshape(-1)
            audio_chunks.append(wave)
        records.append({'chunk': i, 'prefill_seconds': prefill - started,
                        'generation_seconds': finished - prefill,
                        'is_listen': bool(result['is_listen']),
                        'text': result.get('text'),
                        'end_of_turn': bool(result['end_of_turn']),
                        'audio_samples': len(wave) if wave is not None else 0,
                        'audio_peak': float(np.max(np.abs(wave))) if wave is not None and wave.size else 0.0})
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'mode': 'recorded-chunks', 'chunks': records},
                              ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.audio_output and audio_chunks:
        audio_out = Path(args.audio_output)
        audio_out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(audio_out), np.concatenate(audio_chunks), samplerate=24000)


if __name__ == '__main__':
    main()
