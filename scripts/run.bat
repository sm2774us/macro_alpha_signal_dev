@echo off
uv venv --python 3.13
call .venv\Scripts\activate
uv pip install -e ".[dev]"
hls-alpha hls-run --n-assets 8 --n-periods 2000
