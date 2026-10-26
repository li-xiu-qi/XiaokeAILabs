#!/usr/bin/env bash
set -euo pipefail

SRC=${1:?usage: bash scripts/ninfer_gb10_configure.sh <ninfer-source-dir> [build-dir]}
BUILD=${2:-$SRC/build-gb10-system-pkg}
FFMPEG_PREFIX=${NINFER_FFMPEG_PREFIX:-}
LIBCURL_PREFIX=${NINFER_LIBCURL_PREFIX:-}
[[ -z "$FFMPEG_PREFIX" && -d "$HOME/ffmpeg-env" ]] && FFMPEG_PREFIX="$HOME/ffmpeg-env"
[[ -z "$LIBCURL_PREFIX" && -d "$HOME/mamba-ninfer" ]] && LIBCURL_PREFIX="$HOME/mamba-ninfer"
if [[ -z "${NINFER_NINJA:-}" ]]; then
    if command -v ninja >/dev/null 2>&1; then
        NINJA=$(command -v ninja)
    elif [[ -x "$HOME/sglang-env/bin/ninja" ]]; then
        NINJA="$HOME/sglang-env/bin/ninja"
    else
        echo "Ninja not found; install it or set NINFER_NINJA" >&2
        exit 1
    fi
else
    NINJA="$NINFER_NINJA"
fi
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export MAX_JOBS=1
PKG_CONFIG_PATH="/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/share/pkgconfig"
[[ -n "$FFMPEG_PREFIX" ]] && PKG_CONFIG_PATH="$FFMPEG_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
[[ -n "$LIBCURL_PREFIX" ]] && PKG_CONFIG_PATH="$LIBCURL_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
export PKG_CONFIG_PATH
PREFIXES=""
[[ -n "$FFMPEG_PREFIX" ]] && PREFIXES="$FFMPEG_PREFIX"
[[ -n "$LIBCURL_PREFIX" ]] && PREFIXES="${PREFIXES:+$PREFIXES;}$LIBCURL_PREFIX"

# Use the system pkg-config: the conda build can resolve the system zlib
# include directory incorrectly on some aarch64 hosts. Do not prepend the
# libcurl prefix to LD_LIBRARY_PATH while running CMake itself.
cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121a \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_MAKE_PROGRAM="$NINJA" \
  -DPKG_CONFIG_EXECUTABLE=/usr/bin/pkg-config \
  -DCMAKE_PREFIX_PATH="$PREFIXES" \
  -DBUILD_TESTING=ON \
  -DNINFER_BUILD_BENCHMARKS=ON \
  -DNINFER_PERFORMANCE_TRACE=OFF

printf 'Build with:\n  cmake --build %q -j1\n' "$BUILD"
