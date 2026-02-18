HF_ENDPOINT=https://hf-mirror.com uv run python -m r2e_docker build_from_dataset --registry pair-diag-cn-guangzhou.cr.volces.com/code/

# 或继续使用 batch 子模块入口（同样是 Typer）
HF_ENDPOINT=https://hf-mirror.com uv run python -m r2e_docker.batch build_from_dataset --registry pair-diag-cn-guangzhou.cr.volces.com/code/

# 查看帮助
uv run python -m r2e_docker --help