# mls-infer-opt

## 开发环境

```bash
# 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装项目及开发依赖（可编辑模式）
pip install -e ".[dev]"
```

## 常用命令

```bash
pytest          # 运行测试
ruff check .    # 代码检查
ruff format .   # 代码格式化
mypy src        # 类型检查
```
