# R2E Docker Image Builder

为 [R2E-Gym](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset) 数据集批量构建 Docker sandbox 镜像，并通过 F2P/P2P 验证确保镜像质量。

## 核心概念

每个数据集 entry 对应一个 git commit（**NEW/修复后** commit）。构建流程：

1. **Base image** — 每个 repo 一个，包含完整 git 历史和依赖安装
2. **Commit image** — 基于 base image，checkout 到 `commit~1`（**OLD/有 bug** 的 parent commit），复制 NEW commit 的测试文件
3. **Validation** — 在 OLD commit 上运行 NEW commit 的测试，分类结果：

| 分类 | 含义 | 要求 |
|------|------|------|
| **F2P** | expected PASSED, actual FAILED | 必须 ≥ 1（bug-revealing） |
| P2P | expected PASSED, actual PASSED | 无限制（stable） |
| F2F | expected FAILED, actual FAILED | 无限制 |
| **P2F** | expected FAILED, actual PASSED | 必须 = 0 |

验证不通过的镜像会被自动删除。

## 快速开始

```bash
# 安装依赖
uv sync

# 构建全部（base + commit + validation），默认 4 workers
HF_ENDPOINT=https://hf-mirror.com uv run python -m r2e_docker build_from_dataset \
  --registry pair-diag-cn-guangzhou.cr.volces.com/code/

# 只构建前 10 个，跳过 validation
uv run python -m r2e_docker build_from_dataset --limit 10 --no-validate

# 只构建 base images
uv run python -m r2e_docker build_from_dataset --base-only

# 自定义并发和 validation 超时
uv run python -m r2e_docker build_from_dataset --max-workers 8 --validation-timeout 600
```

## 单独验证

```bash
# 简单验证（只检查退出码）
uv run python -m r2e_docker validate <image>

# F2P/P2P 验证（需要 expected_output_json）
uv run python -m r2e_docker validate <image> \
  --expected-output '{"Class.method": "PASSED", ...}'
```

## 输出

构建失败和验证失败的日志保存在 `output/failed_logs/`：
- `base_{repo}.log` — base 构建失败
- `commit_{repo}_{hash}.log` — commit 构建失败
- `validation_{repo}_{hash}.log` — 验证失败（含 F2P/P2P 详细分类）

## 本地测试

用本地docker环境，从头跑通一个 instance 的完整 build + validate 流程（用于debug当前仓库代码逻辑）：

```bash
# 构建 1 个 base image + commit image 并自动 validation
uv run python -m r2e_docker build_from_dataset --limit 1
# 若Success，则说明已经通过F2P/P2P检验
# 若Fail，则在output/failed_logs中记录失败原因
```

## 项目结构

```
r2e_docker/
├── config.py       # DockerBuildConfig, RepoName, 测试命令
├── builder.py      # Docker 构建逻辑（base + commit）
├── batch.py        # 批量构建 + validation 集成
├── validator.py    # F2P/P2P 验证逻辑
├── cli.py          # CLI 入口
├── dockerfiles/    # 各 repo 的 base Dockerfile
└── install_scripts/# 各 repo 的安装脚本
```
