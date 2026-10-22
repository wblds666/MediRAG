# =============================================================================
# MediRAG 后端镜像：把 FastAPI 应用打包成可移植的运行环境
# 构建：docker build -t medrag-api:1.0 .
# 运行：docker run -d --name medrag-api -p 8000:8000 \
#         -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
#         -v medrag-data:/app/data -v medrag-logs:/app/logs medrag-api:1.0
# =============================================================================
FROM python:3.11-slim

# PYTHONUNBUFFERED：日志实时输出，docker logs 才能立刻看到（不加会被缓冲住）
# PYTHONDONTWRITEBYTECODE：不生成 .pyc，镜像更干净
# TZ：容器时区，保证日志时间、数据库 created_at 是北京时间
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 系统依赖：rapidocr-onnxruntime 依赖的 opencv-python 在 slim 镜像里缺 libGL / libglib，
# 不装这两包会导致容器内 `import cv2` 失败（扫描件 OCR 直接不可用）
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 先只拷贝依赖清单再安装：只要依赖没变，改业务代码时这一层能命中缓存，构建快很多
COPY requirements.txt .
# 国内服务器/内网构建时，可换成清华源加速：
# RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -r requirements.txt

# 再拷贝业务代码（这两行改动最频繁，放最后）
COPY app.py .
COPY static ./static

# 数据/日志目录：运行时用 volume 挂出来，容器删了数据也不会丢
RUN mkdir -p /app/data/uploads /app/logs

# 说明：这里用 root 是为了挂载宿主机目录时省去权限麻烦；
# 生产建议新建非 root 用户：RUN useradd -m app && USER app（注意 data/logs 目录要 chown）

EXPOSE 8000

# 健康检查：容器编排（compose/k8s）据此判断实例是否真的可用
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

# workers=1：SQLite 是单文件库，多进程写容易锁；要横向扩容就换 PostgreSQL/pgvector 再调大 workers
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
