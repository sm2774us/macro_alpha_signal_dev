# ── Stage 1: C++26 build ─────────────────────────────────────────────────────
FROM python:3.13-slim AS cpp-builder

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    cmake ninja-build g++-14 \
    curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir nanobind scikit-build-core

WORKDIR /build
COPY CMakeLists.txt .
COPY src/cpp/ src/cpp/

RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++-14 \
    -DPython3_ROOT_DIR=/usr/local \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -Dnanobind_DIR="$(python3 -c 'import nanobind, pathlib; print(pathlib.Path(nanobind.__file__).parent / "cmake")')" \
    -DBUILD_TESTING=OFF \
    && cmake --build build --parallel "$(nproc)" \
    && cmake --install build --prefix /install

# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Copy package source and pre-built C++ extension
COPY pyproject.toml .
COPY LICENSE .
COPY README.md .
COPY src/python/ src/python/
COPY --from=cpp-builder /install/src/python/citadel_alpha/ \
     src/python/citadel_alpha/

# Install runtime deps + package in one layer; hatchling copies pure-Python
# files only (no C compilation) since .so is already present from cpp-builder.
RUN pip install --no-cache-dir hatchling \
    && pip install --no-cache-dir --no-build-isolation . \
    && rm -rf ~/.cache /root/.cache

RUN mkdir -p /app/plots /app/artifacts

RUN useradd -m -u 1000 quant && chown -R quant:quant /app
USER quant

EXPOSE 8888

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD hls-alpha --version || exit 1

CMD ["hls-alpha", "run"]
