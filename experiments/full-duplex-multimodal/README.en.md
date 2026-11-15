# Full-duplex realtime multimodal interaction: MiniCPM-o 4.5

[中文](README.md)

## Background and scope

In November 2026 the Datawhale WeChat article [Li Mu: I'm back with a new model!](https://mp.weixin.qq.com/s/UM2fEPDewjzz-CFuXCGicw) covered BosonAI's full-duplex realtime multimodal model Higgs Realtime Duplex: a 45B MoE front-end model that slices time into 160 ms windows, listens and speaks at the same time, and fits thinking tokens and tool calls into the same windows, with asynchronous tool use and front-end/back-end model cooperation. The article also states that the Duplex weights are not public and only a waitlist is open, and that the already available Higgs Realtime API is a different end-to-end model/service, not the Duplex model.

This experiment does not touch waitlists, private services or paid APIs, and **does not claim to reproduce Higgs**. It uses the Apache-2.0 [MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5) (9B end-to-end audio/video model; the official [MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo) ships a realtime full-duplex demo and duplex APIs) as an open-source approximation, and verifies a comparable link on this device. Both models support continuous audio/video input with concurrent speech output, but their architectures, scale, time slices and background-tool capabilities differ. A new single probe sends client audio in 160 ms chunks, testing only the input cadence and duplex chain; it does not establish that the model internally schedules work in 160 ms windows.

## Article capabilities vs. this experiment

The table below maps the key Higgs Realtime Duplex capabilities described in the [Datawhale article](https://mp.weixin.qq.com/s/UM2fEPDewjzz-CFuXCGicw) to this experiment. "Verified" means a link-level observation on fixed recorded material only; nothing in the "Not verified" rows was tested, and this experiment's data cannot be used to infer Higgs behaviour.

| Article capability | Status in this experiment |
|---|---|
| End-to-end architecture: speech/vision encoders on the input side, audio/video decoders on the output side, no intermediate text | The subject model is itself end-to-end; this experiment does not verify the architecture, only link behaviour |
| Listening while speaking: continuous input and speech output in 160 ms windows, with silence, back-channels or replies when the user pauses | Partially tested: MiniCPM-o 4.5 received recorded audio/video paced in 160 ms client chunks and returned speech during continued input; this does not test Higgs or the model's internal timing windows |
| Model adjusts and stops speaking when the user barges in | The recorded-input probe received `listen` events, but the last server audio delta came 8.42 s after interruption began, missing the pre-set 2 s stop threshold; no human barge-in was tested |
| 160 ms windows for input, output and background tasks | Client chunking only: the probe sends audio every 160 ms; internal model windows, thinking-token timing and background-task scheduling are unverified |
| End-to-end latency of about 0.7-0.9 s; humans reply within about 200 ms | Not verified: human-perceived end-to-end latency was not measured; the video run's per-chunk wall-clock median of 868 ms is not conversational latency |
| Segmented thinking: thinking tokens split across windows | Not verified |
| Asynchronous tool calls and segmented replies | Not verified: MiniCPM-o 4.5 does not cover this architectural capability |
| Front-end interaction model cooperating with a larger back-end model | Not verified: same as above |
| Avatar (real-time face video) output | Not verified: this experiment outputs speech only |
| Zero-shot voice and face swapping | Not verified |
| Token economics: about 2.6K tokens per minute of audio/video stream; only the last 5 minutes of full multimodal context retained | Not verified: token usage was not measured |
| Big Bench Audio 98.9 and other benchmark results | Not verified: no Higgs benchmark was reproduced |

## Current status

### Current verification (2026-11-15)

On an NVIDIA GB10, the official Demo's three-service stack was started separately for video and audio mode. Separate processes avoid a 403 returned when the same worker was asked to handle a second session. Both runs used model revision `fb09628e98b1f55d8d5b863e0eccfcdd11bff499` and the 35.3-second recording `assets/omni_duplex1.mp4`. The model loaded with SDPA in about 7 seconds; flash-attn was not installed. Public metrics for the video and audio runs are in [`evidence/2026-11-15-gpu-duplex-results.json`](evidence/2026-11-15-gpu-duplex-results.json).

| Metric | Video duplex | Audio-only duplex |
|---|---:|---:|
| Input | 100 one-second audio chunks and 100 JPEG frames (36 source frames extracted; the last frame was repeated to maintain cadence) | 70 one-second audio chunks and 0 frames |
| Model responses | 126 response deltas with vision metrics | 0 response deltas with vision metrics |
| Speech output | 28 chunks, 27.6 s, RMS 0.126 | 25 chunks, 23.92 s, RMS 0.103 |
| First audio | 6.74 s | 6.36 s |
| Input/output overlap | 93.7 s | 64.1 s |

Both modes returned non-silent 24 kHz speech while the client continued sending input. The video mode also produced descriptions of the elevator scene. This supports the limited conclusion that the model produced speech while processing a continuing recorded stream; it does not reproduce Higgs or measure human-perceived end-to-end latency.

Still unverified: live microphone and speaker interaction, human listening assessment, and human interruption. Audio-only was run with a fresh service stack; the public script now runs one mode per invocation.

### GPU probe in a fresh isolated environment (2026-11-15)

A separate isolated Python environment was then created on an NVIDIA GB10, using the official Demo commit `47709a9210dfd71afa76c058e017fc8c4db5c8d2`; model weights were reused from the device's existing cache. Base packages were copied from an existing compatible environment, missing realtime backend dependencies were installed, and `pip check` passed. This was not a from-empty online installation of all dependencies, and model weights were not downloaded again. Versions and metrics are recorded in [`evidence/2026-11-15-clean-env-video.json`](evidence/2026-11-15-clean-env-video.json).

| Metric | Video duplex probe (30-second cap) |
|---|---:|
| Input | 30 one-second audio chunks and 30 video frames |
| Response deltas with vision metrics | 38 |
| Output speech | 17 chunks, 17 seconds, 24 kHz, RMS 0.117 |
| First text / audio | 7.08 / 7.09 seconds |
| Input/output overlap | 23.63 seconds |

The probe reached its 30-second session cap and cleaned up the services and ports. It verifies a GPU pipeline with fixed prerecorded input and a headless client, not live interaction or a Higgs reproduction; human interruption and human-perceived latency were not tested. The earlier 60-second controlled rerun used a different isolated environment and FFmpeg 6 runtime, and remains reported separately.

### 160 ms client-chunk probe (2026-11-15)

To test the 160 ms interval mentioned in the article, the probe client was changed from 1-second to 160 ms audio chunks on an NVIDIA GB10. Video frames were selected by each audio chunk's timestamp; the input was a 20-second prerecorded clip bundled with the Demo. Model weights were reused from the device cache. The isolated Python environment and FFmpeg 6.1.1 runtime passed TorchCodec loading and video-decoding prechecks. The cached Demo source had no Git metadata, so its exact commit could not be independently verified for this run. Metrics are in [`evidence/2026-11-15-160ms-video.json`](evidence/2026-11-15-160ms-video.json).

| Metric | Video duplex probe (20-second session cap) |
|---|---:|
| Client input | 112 160 ms audio chunks and 112 frame sends; 20 source frames were selected by timestamp and reused as needed |
| Response deltas with vision metrics | 34 |
| Listen events / speak-to-listen switches | 30 / 11 |
| First text / audio | 13.14 / 13.15 s |
| Speech output | 2 chunks, 2 s, 24 kHz, RMS 0.0479 |
| Input/output overlap | 7.01 s; speech arrived while input was still being sent |

The run ended at the 20-second cap and released all service ports. It shows that the client can pace recorded input in 160 ms chunks while MiniCPM-o 4.5 processes video and emits speech during continued input; it does not show that the model internally uses the same window. The first audio arrived after about 13.15 seconds and only two audio chunks were produced, so this is not evidence of low-latency human interaction. Live interruption, microphone/speaker use and the Higgs architecture remain unverified.

### Recorded-speech interruption probe (new verification item)

The article describes more than listening and speaking at once: it also says the model can stop speaking and return to listening when the user interrupts. The existing continuous-recording probe does not test that behavior. This probe starts a MiniCPM-o-Demo session with one recorded input, waits for the first assistant audio delta, then sends a second recorded audio segment in 160 ms chunks while continuing to send video frames. `force_listen` remains disabled. It measures Demo protocol events; it does not use a microphone or speakers and cannot measure when a person hears playback stop.

Purpose: check whether MiniCPM-o 4.5 reports a listen state and how long assistant audio deltas continue after recorded speech arrives during a response. The acceptance rule is fixed in advance: the same session must send both initial and interruption inputs, produce assistant audio before the interruption, emit at least one `listen` event after interruption input begins, and emit its last audio delta no later than 2 seconds after interruption starts. Missing any signal is recorded as a failure or insufficient evidence; prompts and settings are not tuned and the run is not repeated to change the verdict. One run is allowed, capped at 45 seconds with an 8-second post-interruption observation window. Stop at the first dependency, model-startup or session error.

With the Demo stack running as described above, choose two prerecorded videos that contain both audio and video. The model repository's `assets/omni_duplex1.mp4` can be used as the initial input and the official Demo's `assets/samples/compile.mp4` as the interruption input:

```bash
XL_OUT_DIR=/path/to/out/barge-in \
PROBE_KIND=barge-in PROBE_MODE=video PROBE_MAX_SESSION_S=45 \
INITIAL_VIDEO=/path/to/MiniCPM-o-4_5/assets/omni_duplex1.mp4 \
INTERRUPT_VIDEO=/path/to/MiniCPM-o-Demo/assets/samples/compile.mp4 \
PROBE_CHUNK_MS=160 PROBE_INITIAL_SECONDS=4 PROBE_INTERRUPT_SECONDS=4 \
PROBE_OBSERVE_AFTER_S=8 \
DEMO_DIR=/path/to/MiniCPM-o-Demo MODEL_DIR=/path/to/MiniCPM-o-4_5 \
PYTHON=/path/to/demo-venv/bin/python \
VIDEO=/path/to/MiniCPM-o-4_5/assets/omni_duplex1.mp4 \
FFMPEG_LIB_DIR=/path/to/ffmpeg-libs/aarch64-linux-gnu \
  bash scripts/run_demo_verify.sh
```

The script writes `probe_barge_in_metrics.json`, including input phases, first assistant audio time, interruption start, post-interruption audio-delta count and listen-event times. An audio delta is a server-emitted audio chunk, not proof that playback has finished.

### First GPU result (2026-11-15)

One probe ran on an NVIDIA GB10 in a newly created isolated Python environment. Empty-environment dependency checks, `pip check`, TorchCodec/FFmpeg decoding and CUDA checks passed (Python 3.12.3, PyTorch 2.11.0+cu130, CUDA 13.0). The session sent 25 initial chunks and 25 interruption chunks at 160 ms client pacing. It received one assistant audio chunk before interruption and `listen` events afterward. However, the last server audio delta arrived 8.42 seconds after interruption began, exceeding the pre-set 2-second threshold. **The interruption-stop criterion did not pass.** The session and dependency workflow completed normally; that does not mean the interruption behavior passed. A later rerun followed environment repair; its result and the probe's acceptance-check defect are described in the next section.

An audio delta only means the server emitted audio data; it does not show when playback reached a listener or directly measure perceived latency. This result applies only to this fixed recorded media, MiniCPM-o 4.5 and the Demo protocol. It does not establish live human-interruption behavior or reproduce Higgs Realtime Duplex. Scrubbed metrics are in [`evidence/2026-11-15-barge-in-probe.json`](evidence/2026-11-15-barge-in-probe.json).

### Isolated-environment rerun and acceptance-check correction (2026-11-15)

A later run in a separate isolated environment again passed `pip check`, CUDA checks and video decoding, and the model services started. However, the final audio delta still arrived 8.42 seconds after interruption, so the 2-second stop threshold was missed again. The original run summary marked the workflow `ok` because the probe checked only for a session and output, while the outer summary checked only process exit codes; neither applied the predeclared 2-second behavior criterion. That summary does not mean the interruption test passed. The rerun metrics and original summary discrepancy are recorded in [`evidence/2026-11-15-barge-in-rerun.json`](evidence/2026-11-15-barge-in-rerun.json).

The probe's success condition now requires a complete session and both inputs, a listen event after interruption, and a final audio delta no later than 2 seconds after interruption. Otherwise it exits with failure and records the reason in JSON and the report. The corrected code has now completed a GPU rerun on the same device; see the [rerun with corrected criteria](evidence/2026-11-15-barge-in-corrected-rerun.json). The new environment's `pip check`, CUDA, FFmpeg decoding, and service startup all passed. A listen event followed interruption, but the final server audio delta arrived 4.664 seconds later, beyond the 2-second threshold, so the probe exited with failure under the corrected criterion. Software packages and model weights came from existing device caches; cold-cache downloads remain unverified.

### Follow-up rerun check (2026-11-15)

An earlier run of the current repository script restarted the services and loaded the model, but the probe did not establish a session or send input. The backend closed the session because it could not find an FFmpeg shared library; the script then stopped its services and verified that the ports were released. This attempt produced no new model-capability data. The shared-library setup was later diagnosed and repaired as described below.

Additional inspection found no system FFmpeg shared libraries. The initial diagnosis incorrectly treated both FFmpeg 5's `libavutil.so.57` from the OpenCV wheel and FFmpeg 6's `libavutil.so.58` as incompatible with a `.so.60` requirement, so no GPU probe was started at that point. Later verification confirmed that torchcodec 0.15 ships backends for FFmpeg 4-8 and FFmpeg 6.1.1's `.so.58` loads its core6 backend; the actual blockers were several transitive dependencies and the blas/lapack subdirectories, as described in the controlled rerun below.

### Controlled rerun: FFmpeg dependency repair in a new isolated environment (2026-11-15)

After the rerun above was blocked by torchcodec's FFmpeg shared libraries, a controlled rerun prepared a new environment on the compute device under a fixed gate: the previous isolated environment and system directories were left untouched, a new isolated environment was created for this run, and the FFmpeg 6.1.1 runtime with its transitive dependencies was unpacked into a private prefix under the experiment directory. The precheck actually passed: the active Python imported torchcodec and loaded its FFmpeg backend, AudioDecoder opened the recorded video, the GPU was available, and all service dependencies imported. Exactly one video-mode probe then ran with a 60-second session cap. Metrics are in [`evidence/2026-11-15-rerun-newenv-video.json`](evidence/2026-11-15-rerun-newenv-video.json):

| Metric | This controlled rerun (video, 60 s cap) |
|---|---:|
| Session | Established successfully; ended at the 60-second cap |
| Input | 59 one-second audio chunks and 59 JPEG frames |
| Model responses | 80 response deltas with vision metrics |
| Speech output | 29 chunks, 28.72 s, RMS 0.135 |
| First audio | 7.25 s |
| Input/output overlap | 52.76 s |
| Speak-to-listen switches | 19 |

This run used a new isolated environment and a freshly prepared FFmpeg prefix, so it is not the same environment as the two successful runs above; it shows the chain can be re-prepared from packages cached on the device and again produce non-silent speech. It neither invalidates nor replaces the earlier video and audio runs. Audio-only mode was not run again.

### Empty-venv realtime environment installation self-check (2026-11-15)

To make the from-empty installation of the realtime Python dependencies a reusable, reviewable step, this experiment adds [`scripts/validate_realtime_environment.sh`](scripts/validate_realtime_environment.sh). It creates an empty venv (default interpreter `python3.12`, default location `.work/venv-realtime`, git-ignored; override with `PYTHON_BIN` and `VENV_DIR`), installs `pip==24.0`, `torch==2.11.0` with `torchaudio==2.11.0`, `torchcodec==0.15.0` (PyTorch cu130 index) and `requirements-realtime.txt` in a fixed order, then runs `pip check`, import checks (av, torch, torchaudio, torchcodec, transformers, fastapi, websockets, stepaudio2_minicpmo; when a distribution's module name differs, the actually imported module is recorded), a CUDA availability check, and finally decodes a 1-second WAV generated with the Python standard library through TorchCodec's `AudioDecoder`, requiring more than zero samples to validate the FFmpeg backend. Any failing step stops the run immediately; no version is changed or retried. The script writes a machine-readable `summary.json` (in `XL_OUT_DIR` when set, otherwise `validation-summary.json` inside the venv) with public information only: system architecture/OS, Python and package versions, pip check, CUDA, decode result, and failing stage/error.

Two runs were executed on the compute device on 2026-11-15. The first created the empty venv and installed pip 24.0, but torch installation stopped because the device-side proxy refused connections to the PyTorch cu130 index. Its record is [`evidence/2026-11-15-proxy-install-failure.json`](evidence/2026-11-15-proxy-install-failure.json). The second used a new empty venv and cleared only task-specific proxy variables: Python package installation and `pip check` passed, but TorchCodec could not load its FFmpeg backend because `libavutil.so.56` was unavailable. CUDA and WAV decoding checks were not reached. See [`evidence/2026-11-15-clean-install-env.json`](evidence/2026-11-15-clean-install-env.json).

A later empty-venv run cleared the task-specific proxy variables and installed the Python packages successfully; `pip check` passed and the script found PyAV's bundled `av.libs` directory. TorchCodec still could not load a compatible FFmpeg backend, so CUDA availability and WAV decoding were not reached. See the [empty-environment FFmpeg ABI check](evidence/2026-11-15-empty-venv-ffmpeg-abi.json). TorchCodec loading, CUDA availability and decoding therefore remain unverified in an empty environment. The earlier GPU streaming evidence used separate, prepared environments and remains independent.

Two further remote empty-venv runs on 2026-11-15 did not change this status. The first failed again during torch installation because the device proxy refused connections to the PyTorch cu130 index, so no PyTorch wheel was installed; see [`evidence/2026-11-15-proxy-blocked-rerun.json`](evidence/2026-11-15-proxy-blocked-rerun.json). The second installed the dependencies into a newly created venv and passed `pip check`; wheel files came from the device's existing package cache, so this was not a cold-cache download test. The script located the FFmpeg libraries bundled with the PyAV wheel and added them to `LD_LIBRARY_PATH`, but TorchCodec still could not load a backend from that directory (the loader reported missing `libavutil.so.56`); CUDA and decoding checks were not reached. See [`evidence/2026-11-15-pyav-bundled-ffmpeg-load-failure.json`](evidence/2026-11-15-pyav-bundled-ffmpeg-load-failure.json).

### Bounded FFmpeg ABI repair run (2026-11-15)

After the empty-venv self-check kept stalling on TorchCodec's FFmpeg backend, this run performed one bounded repair and completed the environment checks and the probe inside the same remote run. Root cause: torchcodec 0.15 ships one backend per FFmpeg major version (`libtorchcodec_core4`..`core8`), and each backend dlopens the matching sonames `libavutil.so.56`..`libavutil.so.60`. The `av.libs` directory bundled in PyAV wheels contains only hashed file names (for example `libavutil-<hash>.so.60.8.100`), so the dynamic linker never resolves those sonames from it; pointing `LD_LIBRARY_PATH` at a PyAV library directory therefore cannot satisfy any core backend. The repair unpacks the FFmpeg 6.1.1 Ubuntu arm64 runtime and its transitive dependencies into a private prefix for this run, where `libavutil.so.58` satisfies the core6 backend.

Two scripts were added: [`scripts/prepare_ffmpeg_prefix.sh`](scripts/prepare_ffmpeg_prefix.sh) (unpack the `.deb` files, verify the required sonames and the `blas/` and `lapack/` subdirectories, and gate on a torchcodec import plus an `AudioDecoder` open of the recorded video) and [`scripts/repair_ffmpeg_and_verify.sh`](scripts/repair_ffmpeg_and_verify.sh) (run prefix preparation, empty-venv validation, and exactly one 60-second video probe in order; any failing stage stops the run and writes failure evidence, with no retry and no second probe). Device-side paths (Demo source, model snapshot, Demo environment, `.deb` directories) are injected by the caller through environment variables; nothing device-specific is hardcoded.

All three stages passed in this remote run (`.deb` files and wheels came from caches already on the device; after the pip index metadata cache expired, the single index fetch was performed with the task's proxy variables cleared, the same run-scoped mitigation used in an earlier recorded run):

| Stage | Result |
|---|---|
| FFmpeg prefix | 80 `.deb` files unpacked; required sonames and the `blas/`/`lapack/` subdirectories present; torchcodec import and recorded-video decode gate passed |
| Empty-venv validation | installation and `pip check` passed; av, torch, torchaudio, torchcodec, transformers, fastapi, websockets and stepaudio2 all imported; CUDA available (GB10, CUDA 13.0); 1-second WAV decode passed |
| Video GPU probe | session established, reached the 60-second cap, services and ports cleaned up |

Probe metrics (video, 60-second cap, model revision `fb09628e98b1f55d8d5b863e0eccfcdd11bff499`, material `assets/omni_duplex1.mp4`):

| Metric | This repair run |
|---|---:|
| Input | 59 one-second audio chunks and 59 JPEG frames |
| Model responses | 81 response deltas with vision metrics |
| Speech output | 23 chunks, 23.2 s, 24 kHz, RMS 0.0988 |
| First text / audio | 7.48 / 7.49 seconds |
| Input/output overlap | 52.75 seconds |
| Listen events / speak-to-listen switches | 35 / 31 |

The model produced speech while input continued and described the elevator scene (the displayed floor moved from 21F to 24F). The full record is in [`evidence/2026-11-15-ffmpeg-abi-repair.json`](evidence/2026-11-15-ffmpeg-abi-repair.json). This closes the "empty venv plus a TorchCodec-loadable FFmpeg ABI" gap, but it still used only caches already present on the device: cold-cache downloads, online retrieval of FFmpeg packages, other platforms and other FFmpeg major versions remain unverified, as do live microphone/speaker use, human interruption, human-perceived latency and any Higgs reproduction. This run does not replace the earlier video, audio-only, 30-second and 160 ms results.

### Earlier exploratory records (not the results above)

Earlier JSON records in this directory are exploratory: offline `as_duplex()` runs over 35 chunks (one with 26 listening and 9 speaking chunks; another with 24 listening and 11 speaking that saved 34.2 s of 24 kHz audio, partially cross-checked with an independent Whisper model), a 4-chunk smoke run (all listening responses), and an official end-to-end test with 4 recorded audio/video chunks (all listening responses, no audio output). The 03:29 realtime probe also sent video frames in "audio" mode, so it is not evidence for an audio-only run. None of these records substitutes for the successful video and audio runs above.

## Environment

The official offline example specifies Linux, an NVIDIA GPU, Python 3.10, `transformers==4.51.0`, `torch>=2.3.0,<=2.8.0`, matching torchaudio, `minicpmo-utils[all]>=1.0.5,<=1.0.6` and FFmpeg; the official Web demo recommends more than 28 GB of GPU memory. Weights require approximately 18.7 GB plus speech-decoder assets, Demo source and output space.

Dependency coverage (checked against the scripts' imports and the official Demo's pinned `requirements.txt`, 2026-11-15): offline execution uses this directory's `requirements.txt`; realtime services and probe use the new [`requirements-realtime.txt`](requirements-realtime.txt). Neither file includes hardware-specific torch, torchaudio or torchcodec; install a build matching the target CUDA version and architecture separately. The realtime file pins the Python packages used in the validated environment. It does not replace a platform-compatible PyTorch wheel or imply validation on other platforms.

The following realtime environment steps are based on the Linux ARM64 GPU run. Install torch, torchaudio and torchcodec builds matching the device's CUDA version and architecture first. These example versions were tested on the listed device; PyTorch 2.11 is newer than the model card's stated `<=2.8.0` range, so verify compatibility before using them elsewhere.

```bash
python3.12 -m venv .venv-realtime
. .venv-realtime/bin/activate
python -m pip install pip==24.0
# CUDA 13.0 builds used in this run; select a matching build for other devices via PyTorch's official selector.
python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install torchcodec==0.15.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-realtime.txt
python -m pip check
python -c "import torch, torchaudio, torchcodec, av, numpy, PIL, websockets; print(torch.__version__, torch.cuda.is_available())"
python -c "from torchcodec.decoders import AudioDecoder; print('torchcodec FFmpeg backend loaded')"
```

The steps above are captured in [`scripts/validate_realtime_environment.sh`](scripts/validate_realtime_environment.sh) (documented from this Linux ARM64 + CUDA 13.0 run; other platforms need their own matching PyTorch wheels via PyTorch's official selector), which performs the empty-venv installation and self-check in one command:

```bash
PYTHON_BIN=python3.12 bash scripts/validate_realtime_environment.sh
# Override the venv location (default <experiment>/.work/venv-realtime, git-ignored):
VENV_DIR=/path/to/venv bash scripts/validate_realtime_environment.sh
```

The script writes `summary.json` (public information only; no hostname, username, addresses or private paths). Empty-venv Python package installation and `pip check` have passed in later runs; when `FFMPEG_LIB_DIR` is not set explicitly, the script falls back to the FFmpeg libraries bundled with the PyAV wheel, but TorchCodec still could not load a compatible backend from them, so CUDA and decode checks were not reached. Other devices need FFmpeg shared libraries that TorchCodec can actually load (for example a separately prepared prefix as described above) or an explicit `FFMPEG_LIB_DIR`. The 2026-11-15 repair run closed this gap: with an explicit `FFMPEG_LIB_DIR` pointing at an unpacked FFmpeg 6.1.1 prefix, the same script completed installation, `pip check`, imports, CUDA and decoding in an empty venv (see the section above).

The earlier exploratory runs used Python 3.12.3, torch 2.11.0+cu130, torchaudio 2.11.0+cu130, torchcodec 0.15.0, transformers 4.51.0, accelerate 1.12.0, fastapi 0.162.7, uvicorn 0.54.0, websockets 17.4 and PyAV 16.0.1; that older environment had unresolved `pip check` items. The new probe used a separate environment; its versions and status are in the evidence JSON. Do not apply the old environment's issues to the new environment, or describe the new one as a from-empty installation.

To avoid a moving default branch, pin a public Demo revision. The commit below was checked through the official GitHub API on 2026-11-15, and its `requirements.txt` was read the same day for the coverage analysis above; it is a current installation example, but there is no evidence that it exactly matches the private source archive used in the historical successful runs:

```bash
git clone https://github.com/OpenBMB/MiniCPM-o-Demo.git "$DEMO_DIR"
git -C "$DEMO_DIR" checkout 47709a9210dfd71afa76c058e017fc8c4db5c8d2
python -m pip install -r "$DEMO_DIR/requirements.txt"
python -m pip check
# Probe import check: the Demo dependencies do not explicitly include PyAV or Pillow; install them in the same environment if missing
python -c "import av, numpy, PIL, websockets; print('probe imports OK')"
# e.g.: python -m pip install av pillow
```

Do not mix dependencies between the two environments: recorded history shows a constraint conflict between `minicpmo-utils==1.0.6` and the Demo's `librosa>=0.10.2`, so installing this experiment's `requirements.txt` into the Demo environment may fail dependency resolution; for the realtime path follow the Demo requirements and add only what is missing.

If `torchcodec` cannot load FFmpeg shared libraries, install runtime libraries compatible with the installed torchcodec (including `libavutil`, `libavcodec`, `libavformat`, `libswscale`, `libswresample`, and their system dependencies). The validation script accepts `FFMPEG_LIB_DIR` and adds that directory, plus optional sibling `blas/` and `lapack/` directories, to `LD_LIBRARY_PATH`. This option has not yet been tested by the empty-venv run. Before starting model services, check that torchcodec actually loads in the active environment:

```bash
ffmpeg -version
# Confirm that torchcodec in this Python environment can load its FFmpeg backend.
python -c "from torchcodec.decoders import AudioDecoder; print('torchcodec FFmpeg backend loaded')"
```

`ffmpeg -version` only confirms that the command-line program exists; it does not prove that Python extensions can load their shared libraries. The Python import above is the gate before starting this experiment. If using libraries outside system directories, add the complete dependency chain to `LD_LIBRARY_PATH` and pass the same import check. Do not rename or symlink libraries from a different FFmpeg major version. The official TorchCodec compatibility table lists 0.15 as compatible with PyTorch 2.11, but this does not replace checking FFmpeg library loading ([TorchCodec compatibility](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions)).

### FFmpeg shared libraries: steps verified by this rerun (2026-11-15)

The compute device (Ubuntu 24.04 arm64) has no system FFmpeg, and torchcodec 0.15 needs shared libraries from one of FFmpeg 4-8. The steps below were actually used and verified by the 2026-11-15 controlled rerun; all of them ran inside the experiment directory, without touching system directories or the previous environment:

1. Obtain the Ubuntu arm64 packages for the FFmpeg 6.1.1 runtime and its transitive dependencies: `libavutil58`, `libavcodec60`, `libavformat60`, `libavdevice60`, `libavfilter9`, `libswscale7`, `libswresample4`, `libpostproc57`, plus `libzvbi0t64`, `libcjson1`, `libmbedcrypto7t64`, `libblas3`, `liblapack3`, `libva2`, `libva-drm2`, `libplacebo338`, `libdav1d7`, `libass9`, `libx264-164`, `libx265-199`, `libsdl2-2.0-0` and related dependencies. This rerun unpacked `.deb` files already available in the package cache; downloading packages online was not part of the verified procedure.
2. Unpack those `.deb` files with `dpkg-deb -x` into a private prefix (for example `ffmpeg-root/`).
3. Point `LD_LIBRARY_PATH` at the prefix's `usr/lib/aarch64-linux-gnu` and append its sibling `blas/` and `lapack/` subdirectories: the real BLAS/LAPACK libraries live in those subdirectories, and the top-level symlinks are created by the package install scripts, which `dpkg-deb -x` does not run.
4. Gate: run the Python import check above (this rerun also added an AudioDecoder smoke test on the recorded video) before starting the model services.

Key point: torchcodec 0.15 ships one backend per FFmpeg major version (4-8), so FFmpeg 6.1.1 (`libavutil.so.58`) satisfies its core6 backend; the earlier note in this README claiming `.so.60` was required was incorrect. The dependencies that are easy to miss are `libzvbi`, `libcjson`, `libmbedcrypto` and the blas/lapack subdirectories. For the isolated environment, this rerun's new environment was copied from the previously working one (the uv offline install could not complete because `absl-py` was missing from the cache) and the old environment was left unchanged; installing the full Python dependency set from package caches is still unverified.

When system FFmpeg is absent, the Demo backend's torchcodec needs ABI-compatible FFmpeg shared libraries. The 2026-11-15 repair run established why the PyAV-bundled libraries cannot serve this purpose: the `av.libs` directory in PyAV 16.0.1 wheels contains only hashed file names (for example `libavutil-<hash>.so.60.8.100`), while torchcodec's core4..core8 backends dlopen the sonames `libavutil.so.56`..`libavutil.so.60`, so the dynamic linker never matches the two; this is consistent with the empty-venv runs that failed to load a backend from `av.libs` while reporting a missing `libavutil.so.56`. The working repair is the prefix prepared with the steps above, referenced through `FFMPEG_LIB_DIR` at its `usr/lib/aarch64-linux-gnu` (with the `blas/` and `lapack/` subdirectories). Accordingly, the earlier attribution of the 30-second probe environment to "FFmpeg 8 libraries bundled with PyAV 16.0.1" is inconsistent with this mechanism; that run's actual library source can no longer be verified from surviving records, and its metrics are unaffected. The earlier 60-second rerun did use separately unpacked FFmpeg 6.1.1 libraries. The probe uses PyAV for audio/frame extraction and does not need the `ffmpeg` command-line tool. The 2026-11-15 repair run closed the empty-venv loop from device caches (install, TorchCodec load, CUDA and decode checks all passed); cold-cache downloads and online retrieval of FFmpeg packages remain unverified.

The preparation and checks above are captured in scripts:

```bash
# Prepare the prefix only and pass the torchcodec gate (deb source dirs and gate interpreter required)
FFMPEG_DEB_DIRS="/path/to/deb-dir1 /path/to/deb-dir2" \
GATE_PYTHON=/path/to/venv/bin/python \
GATE_VIDEO=/path/to/recorded.mp4 \
  bash scripts/prepare_ffmpeg_prefix.sh
# One command for "prefix preparation -> empty-venv validation -> one 60-second video probe"
# Device-side paths are injected by the caller; any failing stage stops the run and writes
# failure evidence, with no retry
XL_OUT_DIR=/path/to/out XL_DATA_DIR=/path/to/data XL_RUN_ID=demo-run \
DEMO_DIR=/path/to/MiniCPM-o-Demo MODEL_DIR=/path/to/MiniCPM-o-4_5 \
DEMO_PYTHON=/path/to/demo-venv/bin/python VIDEO=/path/to/recorded.mp4 \
FFMPEG_DEB_DIRS="/path/to/deb-dir1 /path/to/deb-dir2" \
  bash scripts/repair_ffmpeg_and_verify.sh
```

Note: `repair_ffmpeg_and_verify.sh` runs video mode only, exactly one probe, and rejects session caps above 60 seconds. The empty-venv stage checks dependency installation, `pip check`, CUDA, TorchCodec import and WAV decoding. The full MiniCPM-o Demo GPU probe runs in the existing environment specified by `DEMO_PYTHON`, with the same FFmpeg prefix added to its library path. The Demo stack was therefore not run end to end in the empty venv.

## Model, data and command

Model revision `fb09628e98b1f55d8d5b863e0eccfcdd11bff499`; the model repository provides `assets/omni_duplex1.mp4` and `assets/HT_ref_audio.wav`; follow their source licensing. Include model weights and `assets/token2wav/` speech-decoder files in the download. Do not commit models or data to this repository.

Offline chunked run (official `as_duplex()` API, not realtime duplex):

```bash
huggingface-cli download openbmb/MiniCPM-o-4_5 \
  --revision fb09628e98b1f55d8d5b863e0eccfcdd11bff499 \
  --local-dir "$MODEL_DIR"
python scripts/offline_duplex.py \
  --model-dir "$MODEL_DIR" \
  --video "$MODEL_DIR/assets/omni_duplex1.mp4" \
  --reference-audio "$MODEL_DIR/assets/HT_ref_audio.wav" \
  --output "$OUTPUT_DIR/chunks.json" --audio-output "$OUTPUT_DIR/response.wav" --max-chunks 35
```

Realtime duplex verification (official Demo stack + streaming probe). Fetch the [MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo) source first, then:

Bounded follow-up for this environment repair: start the model only after torchcodec loads ABI-compatible FFmpeg libraries; run at most one video probe with a 60-second cap. Success requires a session, at least one audio input chunk, and non-silent speech output. Stop on any environment/model startup error or at the 60-second cap; do not tune or add another run.

```bash
DEMO_DIR=/path/to/MiniCPM-o-Demo   # source tree with gateway.py, worker.py, py_backend/
MODEL_DIR=/path/to/MiniCPM-o-4_5   # local model snapshot directory
git clone https://github.com/OpenBMB/MiniCPM-o-Demo.git "$DEMO_DIR"
python3 -m venv .venv-demo
. .venv-demo/bin/activate
# First install torch, torchaudio, and torchcodec for your CUDA/device architecture per PyTorch's instructions
python -m pip install -r "$DEMO_DIR/requirements.txt"
python -m pip check
# Probe import check: install missing packages in the same environment (e.g. python -m pip install av pillow)
python -c "import av, numpy, PIL, websockets; print('probe imports OK')"

VIDEO=$MODEL_DIR/assets/omni_duplex1.mp4
FFMPEG_LIB_DIR=/path/to/ffmpeg-libs/aarch64-linux-gnu   # when system FFmpeg is unavailable; may contain blas/ and lapack/
XL_OUT_DIR=/path/to/out/video PROBE_MODE=video \
DEMO_DIR=$DEMO_DIR MODEL_DIR=$MODEL_DIR PYTHON=$(command -v python) VIDEO=$VIDEO \
FFMPEG_LIB_DIR=$FFMPEG_LIB_DIR \
  bash scripts/run_demo_verify.sh
# 160 ms client-chunk probe using the Demo's bundled 20-second recording
VIDEO=$DEMO_DIR/assets/samples/compile.mp4
XL_OUT_DIR=/path/to/out/video-160ms PROBE_MODE=video \
PROBE_CHUNK_MS=160 PROBE_MAX_SESSION_S=20 \
DEMO_DIR=$DEMO_DIR MODEL_DIR=$MODEL_DIR PYTHON=$(command -v python) VIDEO=$VIDEO \
FFMPEG_LIB_DIR=$FFMPEG_LIB_DIR \
  bash scripts/run_demo_verify.sh
# Start a separate stack for audio mode and choose a new output directory.
XL_OUT_DIR=/path/to/out/audio PROBE_MODE=audio \
DEMO_DIR=$DEMO_DIR MODEL_DIR=$MODEL_DIR PYTHON=$(command -v python) VIDEO=$VIDEO \
FFMPEG_LIB_DIR=$FFMPEG_LIB_DIR \
  bash scripts/run_demo_verify.sh
```

Each invocation runs one mode and starts the backend (port 22500 by default, `--model-path` at the local model), worker (22400, `--backend-server-url` at the backend) and gateway (8006, HTTP mode). Set `PROBE_MODE=video` or `PROBE_MODE=audio`; video mode attaches one frame per audio chunk, while audio mode sends no video frames. Run the two modes separately to avoid a 403 when the same worker handles a second session. The session cap is set with `PROBE_MAX_SESSION_S` (100 s by default for video) or `PROBE_AUDIO_MAX_SESSION_S` (70 s by default for audio); the 2026-11-15 controlled rerun used `PROBE_MAX_SESSION_S=60`. The probe `scripts/demo_realtime_probe.py` is adapted from the official `examples/realtime/video_probe.py`: its WebSocket protocol, real-time pacing and metric definitions are unchanged; media extraction uses PyAV because the device has no `ffmpeg` command. Each run writes `probe_<mode>_metrics.json`, `summary.json`, `report.md`, service logs and a WAV file with received audio. On completion, the script stops the services it started and verifies that their ports are released.

Running the probe alone (with the stack already up):

```bash
python scripts/demo_realtime_probe.py --url http://127.0.0.1:8006 --mode video \
  --video "$VIDEO" --max-session-s 100 \
  --output out/probe_video_metrics.json --audio-output out/probe_video_audio.wav
```

## Known limitations

- All conclusions come from fixed recorded material and a headless client; no live microphone, speakers, human listening or real interruption.
- The standard probe defaults to one-second chunks; one 160 ms client-chunk run is also recorded, but live interaction and internal model scheduling remain unverified. The video run's per-chunk wall-clock median was 868 ms; this is not human-perceived end-to-end latency.
- The article also describes avatar video output, asynchronous tool calls and frontend/backend model collaboration. This experiment uses MiniCPM-o 4.5 and only checks frame processing and speech output during continuous prerecorded audio/video input; it does not cover those architectural capabilities or establish Higgs latency or cost.
- The interruption probe received `listen` events, but its last server audio delta came 8.42 seconds after interruption, missing the 2-second stop threshold. This is not a passing stop test. Recorded inputs and protocol events do not show that a human heard speech stop or that playback stopped.
- A later isolated-environment rerun produced the same 8.42-second result. The old script incorrectly treated session and output presence as probe success; this check is corrected, but the new exit status has not yet been validated in a GPU run.
- Dependency setup still requires a CUDA-compatible torch stack and loadable FFmpeg shared libraries. Realtime Python packages now have a separate pinned list in [`requirements-realtime.txt`](requirements-realtime.txt). The 2026-11-15 repair run closed the empty-venv gap: with an explicit `FFMPEG_LIB_DIR` pointing at an unpacked FFmpeg 6.1.1 prefix, installation, `pip check`, imports, CUDA and decoding all passed. PyAV wheels' `av.libs` directory contains only hashed library file names, so the dynamic linker cannot match torchcodec's sonames and that directory is not a viable repair path. The run still used only caches already present on the device: cold-cache downloads, online retrieval of FFmpeg packages and other platforms remain unverified. Conflicts involving `minicpmo-utils`, `librosa` and `decord` belonged to an earlier exploratory environment and should not be attributed to these runs.
- The earlier video and audio runs used a previously existing isolated environment with unresolved `pip check` items. The later 30-second probe used a separate environment; its versions and passing `pip check` are in `evidence/2026-11-15-clean-env-video.json`. PyTorch 2.11 exceeds the model card's 2.8.0 upper bound.
- The 2026-11-15 controlled rerun used a new isolated environment created for that run (copied from the previously working environment; the uv offline install could not complete because `absl-py` was missing from the cache, and the old environment was left unchanged) plus the FFmpeg 6.1.1 prefix prepared with the steps above. The shared-library dependency can therefore be re-assembled and verified on the device, but a from-scratch installation of the whole environment and online retrieval of FFmpeg packages remain unverified; this run used `.deb` files already available in the cache. The same day's repair run likewise used device caches only (the single index fetch after the pip metadata cache expired was performed with the task's proxy variables cleared) and used a new empty venv plus a new prefix.
- Successful results include an earlier video run, an audio-only run, a 60-second controlled video rerun, a separate 30-second video probe and a 60-second FFmpeg ABI repair run (new empty venv plus new prefix) on one device. Live microphone/speaker interaction, human listening assessment and human interruption were not tested. Runs used different environments and budgets; their metrics should not be combined as repeated trials.

## Evidence

| File | Contents |
|---|---|
| `evidence/2026-11-15-gpu-duplex-results.json` | public metrics for two successful GPU streaming runs and a historical rerun that stopped before the probe because FFmpeg shared libraries were missing |
| `evidence/2026-11-15-rerun-newenv-video.json` | the 2026-11-15 controlled rerun: preparation of the new isolated environment and the FFmpeg 6.1.1 prefix, the torchcodec loading precheck, and metrics of one 60-second video probe |
| `evidence/2026-11-15-ffmpeg-abi-repair.json` | the 2026-11-15 bounded repair run: the hashed-name root cause for PyAV-bundled libraries, FFmpeg 6.1.1 prefix preparation and the torchcodec gate, empty-venv validation (versions, pip check, CUDA, decode) and metrics of one 60-second video GPU probe |
| `evidence/2026-11-15-clean-env-video.json` | the separate 30-second GPU video probe in a fresh isolated environment, including package versions and setup boundaries |
| `evidence/2026-11-15-barge-in-probe.json` | one GB10 interruption probe: fresh isolated environment and CUDA/decode checks passed; listen events were recorded, but the final audio delta missed the pre-set threshold |
| `evidence/2026-11-15-barge-in-rerun.json` | later isolated-environment rerun: environment and service workflow passed, interruption behavior still exceeded the 2-second threshold; records why the original summary misreported success and the corrected failing verdict |
| `evidence/2026-11-15-barge-in-corrected-rerun.json` | GPU rerun with corrected criteria: clean environment, decode, and service passed; final audio delta was 4.664 seconds, beyond the 2-second threshold, and the probe correctly exited as failed |
| `evidence/2026-11-15-proxy-install-failure.json` | the first 2026-11-15 empty-venv setup attempt: the device proxy refused the PyTorch index and installation stopped |
| `evidence/2026-11-15-clean-install-env.json` | the second empty-venv attempt: Python package installation and `pip check` passed; TorchCodec failed to load because FFmpeg libraries were missing, so CUDA and decode checks were not reached |
| `evidence/2026-11-15-empty-venv-ffmpeg-abi.json` | a later empty-venv check: PyAV's library directory was found, but TorchCodec could not load a compatible FFmpeg backend; CUDA and decoding were not reached |
| `evidence/2026-11-15-proxy-blocked-rerun.json` | a later 2026-11-15 empty-venv rerun: the proxy refused the PyTorch cu130 index during torch installation, so no PyTorch wheel was installed and later checks were not reached |
| `evidence/2026-11-15-pyav-bundled-ffmpeg-load-failure.json` | a later 2026-11-15 empty-venv rerun: package installation and `pip check` passed and the script used the FFmpeg libraries bundled with the PyAV wheel, but TorchCodec still could not load a backend; CUDA and decoding checks were not reached |
| `evidence/2026-11-15-160ms-video.json` | video probe with 160 ms client audio chunks; reports input/output overlap and vision response metrics, not model-internal scheduling |
| `evidence/2026-11-15-auto-ffmpeg-detect-install-blocked.json` | empty-venv validation blocked during PyTorch installation, before automatic FFmpeg library discovery |
| `evidence/2026-11-15-demo-verify-032901.json` | earlier exploratory record; its audio mode also sent video frames, so it is not used for the audio-only results above |
| `evidence/2026-11-15-recorded-smoke.json`, `evidence/2026-11-15-recorded-full.json`, `evidence/2026-11-15-recorded-audio.json` | historical records: offline chunked runs (4-chunk smoke, 35-chunk full, 35-chunk with audio) |
| `evidence/2026-11-15-demo-recorded.json` | historical record: official end-to-end test with 4 recorded chunks, all listening responses |
