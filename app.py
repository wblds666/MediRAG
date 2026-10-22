# -*- coding: utf-8 -*-
"""
MediRAG · 医疗知识库问答系统（后端单文件版）
=====================================================================
一句话说明：
    把医疗资料（PDF / txt）传进来 → 切成小段 → 向量化 → 存进 SQLite；
    用户提问时先把问题向量化 → 检索最相关的几段资料 → 拼进 Prompt 交给本地大模型
    （Ollama + Qwen）生成"带页码出处的回答" → 再跑一遍引用校验 → 问答记录写回数据库。

技术栈对应关系（面试可直接照着讲）：
    Web 前端      : static/index.html（原生 HTML + fetch 调接口）
    服务端 API    : FastAPI（本文件），提供 /api/upload、/api/ask 等 12 个接口
    数据库 DB     : SQLite（documents / chunks / qa_logs 三张表，含"向量列"）
    AI 能力       : LangChain 负责切分/向量化/工具调用，Ollama 跑 Qwen 与 bge-m3
    RAG           : 递归切分 + 向量化 + 余弦相似度检索 + 邻块扩展 + 多查询改写
    抗幻觉        : 引用校验（覆盖率/越界编号/弱支撑句）+ 逐句归因 + 自动补引用 + 常驻免责
    Agent/工具调用 : get_current_time 工具，让模型自己决定"要不要查时间"
    日志          : logging + RotatingFileHandler（控制台 + 轮转文件）；问答日志只留最近 10 条
    部署          : Dockerfile + nginx.conf + docker-compose.yml

本地启动：
    uvicorn app:app --host 0.0.0.0 --port 8000     然后打开 http://localhost:8000

接口清单：
    GET    /                       前端页面
    GET    /api/health             健康检查（Docker HEALTHCHECK / Nginx 探活）
    POST   /api/upload             上传资料 → 解析/OCR → 切分 → 向量化 → 写库（重复内容 409）
    POST   /api/ask                提问 → 多查询检索 → 生成 → 引用校验 → 日志写库
    GET    /api/documents          资料列表（含 serial / display_no）
    PATCH  /api/documents/{id}     修改资料编号
    POST   /api/documents/renumber 按上传顺序重排编号 1..N
    POST   /api/documents/tidy     按编号重命名磁盘文件（不断开关联）
    GET    /api/documents/{id}/file 打开原始文件（PDF 支持 #page=N 跳页）
    DELETE /api/documents/{id}     删除单份（连向量和原始文件）
    DELETE /api/documents          清空全部（问答日志保留）
    GET    /api/qa-logs            最近问答日志（只保留最近 N 条）
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import sqlite3
import time
import zlib
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

# ============================================================================
# 0. 配置项：全部可用环境变量覆盖，这样同一份镜像能跑开发/测试/生产（Docker 里改 env 即可）
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))        # 数据目录（生产要挂宿主机卷做备份）
UPLOAD_DIR = DATA_DIR / "uploads"                                # 原始文件留档，便于追溯
LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "logs"))          # 日志目录
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "medrag.db"))     # SQLite 数据库文件

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # Ollama 服务地址（容器里写 http://ollama:11434）
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")                 # 对话模型（医疗问答建议 7B 起步，可换 qwen2.5:14b）
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")                 # 向量模型（中文/医疗文本首选 bge-m3）
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))                 # 每段最大字符数
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))            # 相邻段重叠字符，防止答案正好被切开
TOP_K = int(os.getenv("TOP_K", "4"))                             # 检索返回的资料段数
QA_LOG_KEEP = int(os.getenv("QA_LOG_KEEP", "10"))                # 问答日志只保留最近多少条（超出删最早）
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "60"))            # 单文件大小上限（医疗教材 PDF 常见几十 MB）
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "3"))         # Agent 工具调用最多循环几轮（防死循环）
ENABLE_TOOLS = os.getenv("ENABLE_TOOLS", "true").lower() == "true"  # 是否启用工具调用

# ---------- 扫描件 OCR（手机拍照/扫描导出的 PDF 没有文字层，只能靠 OCR）----------
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() == "true"  # 文字层不合格时是否自动 OCR
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "300"))           # 单次最多 OCR 多少页（防接口长时间阻塞）
OCR_DPI = int(os.getenv("OCR_DPI", "200"))                       # 渲染 DPI：越高越准也越慢
MIN_TEXT_CHARS = int(os.getenv("MIN_TEXT_CHARS", "200"))         # 少于此字数 → 判定"没抽到正文"
MIN_CJK_RATIO = float(os.getenv("MIN_CJK_RATIO", "0.10"))        # 中文占比下限（垃圾文本几乎没中文）
MIN_LATIN_RATIO = float(os.getenv("MIN_LATIN_RATIO", "0.10"))    # 英文占比下限（兼容英文资料）

# ---------- 检索增强 & 幻觉体检 ----------
NEIGHBOR_WINDOW = int(os.getenv("NEIGHBOR_WINDOW", "1"))         # 命中的段落前后各带几段一起喂模型
MULTI_QUERY = os.getenv("MULTI_QUERY", "true").lower() == "true"  # 复杂问题是否先拆成多条检索式
MULTI_QUERY_N = int(os.getenv("MULTI_QUERY_N", "3"))             # 改写出几条检索式
MIN_SUPPORT_RATIO = float(os.getenv("MIN_SUPPORT_RATIO", "0.08"))  # 引用段落与答案句的重合度下限
CITATION_REPAIR = os.getenv("CITATION_REPAIR", "true").lower() == "true"  # 引用覆盖率不达标时自动重写一次
MIN_COVERAGE = float(os.getenv("MIN_COVERAGE", "0.5"))           # 低于这个覆盖率就触发补引用
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "2500"))  # 送进 Prompt 的资料总字数上限（太长模型会丢格式要求）
NEIGHBOR_MAX_CHARS = int(os.getenv("NEIGHBOR_MAX_CHARS", "800")) # 单条命中扩展后的最大字数

OCR_HINT = (
    "这个 PDF 没有可提取的文字层（扫描件 / 手机拍照导出的图片型 PDF）。"
    "系统尝试 OCR 但不可用：请 pip install pymupdf rapidocr-onnxruntime，"
    "或先用 OCRmyPDF / Adobe Acrobat 把它另存为「可搜索 PDF」再上传。"
)


# ============================================================================
# 1. 日志：招聘要求里的"日志处理"。控制台一份（docker logs 能看），文件一份（事后排查）
# ============================================================================
def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    logger = logging.getLogger("medrag")
    logger.setLevel(logging.INFO)
    if logger.handlers:                     # uvicorn --reload 会重复 import，避免日志打两遍
        return logger

    # ① 控制台：容器化部署后 `docker logs medrag-api` 直接能看到
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # ② 文件 + 按大小轮转：单文件 5MB，保留 5 份，超过自动滚动删除，防止把磁盘写满
    file_handler = RotatingFileHandler(
        LOG_DIR / "medrag.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


log = setup_logging()


def now_str() -> str:
    """统一的时间字符串，写库和打日志都用它，方便对时间线。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================================
# 2. 数据库：招聘要求里的"数据库存储"。SQLite 零依赖，最简；生产可无损换成 MySQL/PostgreSQL
# ============================================================================
SCHEMA = """
-- 资料表：一次上传一条记录
CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT    NOT NULL,          -- 资料标题（取文件名）
    filename     TEXT    NOT NULL,          -- 原始文件名
    file_type    TEXT,                      -- 扩展名
    size_bytes   INTEGER,                   -- 文件大小
    chunk_count  INTEGER DEFAULT 0,         -- 切成多少段
    page_count   INTEGER DEFAULT 0,         -- 原始 PDF 共几页
    embed_model  TEXT,                      -- 用的哪个向量模型（换模型后老向量要重建）
    source       TEXT,                      -- 文本来源：text-layer（电子版）/ ocr（扫描件识别）
    serial       INTEGER,                   -- 用户自定义编号（1..N），排序和界面显示用；空=未编号
    stored_name  TEXT,                      -- uploads 里的实际文件名（改名后仍能找到）
    hash         TEXT,                      -- 文件内容 sha256，用于重复上传判重
    created_at   TEXT    NOT NULL
);

-- 段落表：既存文本，也存向量（最简做法；生产环境建议换 Chroma / pgvector / Milvus）
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,           -- 第几段，用于回答里标注出处
    page        INTEGER,                    -- 来自第几页：医生可以直接翻原文核对
    content     TEXT    NOT NULL,           -- 段落原文
    embedding   TEXT    NOT NULL,           -- 向量（JSON 数组文本）
    dim         INTEGER NOT NULL            -- 向量维度，检索时按维度过滤，避免换模型后串味
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

-- 问答日志表：每次提问都落库，可做"热门问题统计/人工复核/效果评估"
CREATE TABLE IF NOT EXISTS qa_logs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL,
    sources    TEXT,                        -- JSON：命中的资料段
    tool_calls TEXT,                        -- JSON：工具调用轨迹
    queries    TEXT,                        -- JSON：实际用到的检索式（多查询改写结果）
    grounding  TEXT,                        -- JSON：引用校验结果（覆盖率/未标注/弱支撑）
    model      TEXT,                        -- 实际用的模型
    latency_ms REAL,                        -- 端到端耗时，做性能监控
    created_at TEXT NOT NULL
);
"""


@contextmanager
def db():
    """
    一次请求开一个连接，用完就关：简单、天然线程安全（FastAPI 会在线程池里跑同步代码）。
    真上生产可以换成连接池 + SQLAlchemy ORM，业务代码写法基本不变。
    """
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row          # 让查询结果支持 row["字段名"] 取值
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()                     # 出错回滚，保证数据一致
        raise
    finally:
        conn.close()


def ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    """轻量迁移：老库缺列就补（SQLite 支持 ADD COLUMN，不动已有数据）。"""
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
            log.info("数据库迁移：%s 表新增 %s 列", table, name)


def init_db() -> None:
    """建目录 + 建表，幂等，可以重复执行。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with db() as conn:
        conn.executescript(SCHEMA)
        ensure_columns(conn, "documents", {"source": "TEXT", "page_count": "INTEGER DEFAULT 0",
                                           "serial": "INTEGER", "stored_name": "TEXT", "hash": "TEXT"})
        ensure_columns(conn, "chunks", {"page": "INTEGER"})
        ensure_columns(conn, "qa_logs", {"queries": "TEXT", "grounding": "TEXT"})

        # 老数据补算内容哈希（重复上传判重要用）：只对还没有 hash 的记录算一次
        pending = conn.execute(
            "SELECT id, stored_name FROM documents WHERE hash IS NULL AND stored_name IS NOT NULL"
        ).fetchall()
        for row in pending:
            path = UPLOAD_DIR / row["stored_name"]
            if path.exists():
                conn.execute("UPDATE documents SET hash = ? WHERE id = ?",
                             (hashlib.sha256(path.read_bytes()).hexdigest(), row["id"]))
        if pending:
            log.info("已为 %s 份历史资料补算内容哈希", len(pending))
    log.info("数据库就绪: %s", DB_PATH)


# ============================================================================
# 3. 向量化：RAG 的"向量化"环节。优先用 Ollama 的 bge-m3，连不上自动降级
# ============================================================================
class HashEmbeddings(Embeddings):
    """
    离线兜底向量器：不依赖任何模型，把文本的 2-gram 用 CRC32 哈希到 512 维再归一化。
    作用：没装 Ollama 时项目照样能跑通"上传→检索→回答"全流程（仅用于演示，效果一般）。
    注意用 zlib.crc32 而不是内置 hash()：内置 hash 每个进程随机加盐，重启后老向量就废了。
    """

    dim = 512

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        text = (text or "").lower()
        grams = [text[i:i + 2] for i in range(max(len(text) - 1, 1))]   # 中文按 2 字滑窗
        for g in grams:
            vec[zlib.crc32(g.encode("utf-8")) % self.dim] += 1.0        # 词袋哈希（类似 HashingVectorizer）
        norm = float(np.linalg.norm(vec))
        return (vec / norm if norm else vec).tolist()                   # L2 归一化，方便点积算余弦

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embedder() -> tuple[Embeddings, str]:
    """启动时探活：能用 Ollama 就用真向量模型，不能就降级，保证接口永远不 500。"""
    if os.getenv("EMBED_BACKEND", "ollama") == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings

            emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
            dim = len(emb.embed_query("连通性自检"))     # 真打一次请求，模型没 pull 下来会在这里失败
            log.info("向量模型就绪: Ollama/%s (dim=%s)", EMBED_MODEL, dim)
            return emb, f"ollama:{EMBED_MODEL}"
        except Exception as exc:                          # noqa: BLE001 —— 启动阶段要兜住所有异常
            log.warning("Ollama 向量模型不可用(%s)，降级为本地哈希向量（仅演示用）", exc)
    return HashEmbeddings(), "hash-fallback:512"


# ============================================================================
# 4. 大模型 & 提示词：医疗场景一定要"只依据资料回答 + 标注出处 + 不给个体化诊断"
# ============================================================================
# 合规声明：放在接口返回值里，前端/小程序/第三方系统都一样能看到，不能只靠页面提示
DISCLAIMER = (
    "本回答由 AI 依据知识库资料生成，仅供查阅参考，可能存在改写失真或张冠李戴"
    "（例如把“丝状蚴钻入皮肤”写成“虫卵在皮肤下孵化”），请按来源页码核对原文；"
    "本系统不提供诊断与处方建议，不能替代医生判断。"
)

SYSTEM_PROMPT = """你是一名严谨的医疗知识库助手，为医生和医学生查阅资料服务。
必须遵守以下规则：
1. 只依据【资料】中的内容回答，不要编造资料以外的医学结论；
2. 资料不足时直接回答"资料中未提及"，并说明还需要补充哪类资料；
3. 用中文回答，条理清晰，可以用分点；专业术语后面可加一句通俗解释；
4. 不做个体化诊断和处方建议；涉及急症、危重情况时提醒"请立即就医"。

【出处标注】这是硬性要求，不标出处的回答视为不合格：
- 【资料】里每段前面都有编号 [1][2][3]…，每个结论句末尾都必须标上来源编号；
- 一句话综合了多段资料就写多个编号，例如 [1][3]；
- 只允许使用【资料】里真实存在的编号，不要出现 [5] 这种不存在的编号；
- 宁可少说，也不要写没有资料支撑的结论；
- 用纯文本分点（1. 2. 3.），**不要**使用 markdown 的 # 和 **，保证每句话都能被逐句核对。

回答格式示例：
蛔虫成虫主要寄生于小肠，雌虫每天产卵约 24 万个 [1]。
感染途径为经口误食感染期虫卵 [1][3]。"""


def build_llm():
    """启动时探活大模型；连不上就进入离线降级模式（只回检索原文），而不是让接口报错。"""
    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)  # 医疗问答要稳，温度给 0
        llm.invoke("ping")                    # 探活：服务没起 / 模型没 pull 都会在这里抛错
        log.info("大模型就绪: Ollama/%s", LLM_MODEL)
        return llm, f"ollama:{LLM_MODEL}"
    except Exception as exc:                  # noqa: BLE001
        log.warning("Ollama 大模型不可用(%s)，进入离线降级模式（只返回检索原文）", exc)
        return None, "offline-extractive"


# ============================================================================
# 5. 工具调用：招聘要求里的 Agent / Function Calling
#    @tool 装饰器会把函数签名+docstring 自动转成模型能看懂的"工具说明书"（JSON Schema）
# ============================================================================
@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """查询当前的日期和时间。当用户询问"现在几点""今天几号""当前时间"时必须调用本工具。"""
    try:
        tz = ZoneInfo(timezone)               # Windows/精简镜像没装 tzdata 时会抛异常
    except Exception:                         # noqa: BLE001
        tz = None
    return (datetime.now(tz) if tz else datetime.now()).strftime("%Y-%m-%d %H:%M:%S %A")


TOOLS = [get_current_time]
TOOL_MAP = {t.name: t for t in TOOLS}         # 名字 → 工具对象；模型只给"名字+参数"，执行由程序负责


def time_intent(question: str) -> bool:
    """
    规则判断问题是不是在问时间：给"不支持工具调用的模型"做兜底。
    关键词刻意取得比较窄 —— 像"现在推荐用什么药"这种医疗问题不该被误判成问时间。
    """
    return any(k in question for k in ("几点", "几号", "星期几", "当前时间", "什么时间", "今天是", "今天日期"))


# ============================================================================
# 6. 文本抽取 + 切分：RAG 的"切分"环节
#    注意：PDF 分两种 —— ① 电子版（有文字层）能直接抽；② 扫描件/手机拍照版（整页是图）
#    第 ② 种必须 OCR，否则抽出来的只是一堆符号，向量化后检索永远没结果（最常见的问题）
# ============================================================================
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    # 中文医疗资料优先按 空行 → 换行 → 句号 → 感叹/问号 → 分号 → 逗号 递归切，
    # 保证尽量在"语义边界"断开，不会把一句话/一个剂量说明切一半
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
)

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")


def text_stats(text: str) -> dict:
    """给抽出来的文本做"体检"：字符数、中文占比、英文字母占比。"""
    chars = len((text or "").strip())
    cjk = len(CJK_RE.findall(text or ""))
    latin = len(LATIN_RE.findall(text or ""))
    return {
        "chars": chars,
        "cjk_ratio": round(cjk / chars, 3) if chars else 0.0,
        "latin_ratio": round(latin / chars, 3) if chars else 0.0,
    }


def is_low_quality(text: str) -> bool:
    """
    判断"这段文本是垃圾"：字符太少，或既没中文也没英文（只剩圈码①②、页码、符号）。
    扫描版 PDF 用普通方式抽出来的就是这种东西；如果放它入库，后面检索永远命中不了有效内容。
    """
    st = text_stats(text)
    return st["chars"] < MIN_TEXT_CHARS or (
        st["cjk_ratio"] < MIN_CJK_RATIO and st["latin_ratio"] < MIN_LATIN_RATIO
    )


def _pymupdf():
    """
    兼容 PyMuPDF 新旧包名：≥1.24 推荐 `import pymupdf`，老版本只有 `import fitz`。
    拿不到就抛 ImportError，由上层决定是否降级到 pypdf / 提示装包。
    """
    try:
        import pymupdf

        return pymupdf
    except ImportError:
        import fitz

        return fitz


def extract_pdf_pages(raw: bytes) -> list[tuple[int, str]]:
    """电子版 PDF 逐页抽文字：优先 PyMuPDF（中文排版兼容性更好），没装则退回 pypdf。
    返回 [(页码, 该页文本)] —— 页码必须留着，否则回答时没法告诉医生"在第几页"。"""
    try:
        pymupdf = _pymupdf()
    except ImportError:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    with pymupdf.open(stream=raw, filetype="pdf") as doc:
        return [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]


def ocr_pdf_pages(raw: bytes) -> list[tuple[int, str]]:
    """
    扫描件/拍照版 PDF → 渲染成图片 → RapidOCR 逐页识别，返回 [(页码, 该页文本)]。
    RapidOCR 是 PaddleOCR 的 ONNX 推理版：pip 装完就能用、纯 CPU 可跑、自带中文模型。
    """
    pymupdf = _pymupdf()
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()                     # 首次初始化要加载模型（约 1~3 秒），之后复用
    pages: list[tuple[int, str]] = []
    with pymupdf.open(stream=raw, filetype="pdf") as doc:
        total_pages = len(doc)
        total = min(total_pages, OCR_MAX_PAGES)      # 巨厚的书别一次 OCR 到头，防止接口超时
        for i in range(total):
            pix = doc[i].get_pixmap(dpi=OCR_DPI)     # 渲染成位图，DPI 越高越准、也越慢
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            result, _ = engine(img)                  # 返回 [[框坐标, 文本, 置信度], ...]
            pages.append((i + 1, "\n".join(item[1] for item in (result or []))))
            if (i + 1) % 10 == 0 or i + 1 == total:
                log.info("OCR 进度 %s/%s 页", i + 1, total)
    if total < total_pages:
        log.warning("PDF 共 %s 页，仅 OCR 了前 %s 页（受 OCR_MAX_PAGES 限制）", total_pages, total)
    return pages


def load_document_pages(filename: str, raw: bytes) -> tuple[list[tuple[int, str]], dict]:
    """
    统一入口：返回 ([(页码, 文本)], 元信息)。只支持两类：
      .pdf                  电子版直接抽文字层；扫描件自动 OCR
      .txt/.md/.csv/.json   直接解码（兼容 GBK/GB18030）
    其它格式（docx/pptx/xlsx/图片等）统一提示不支持 —— 需要的话先另存为 PDF。
    """
    suffix = Path(filename).suffix.lower()
    meta: dict = {"source": "text-layer", "ocr": False}

    if suffix == ".pdf":
        pages = extract_pdf_pages(raw)
        full = "\n".join(t for _, t in pages)
        meta["quality"] = text_stats(full)
        if is_low_quality(full):
            if not OCR_ENABLED:
                raise HTTPException(status_code=400, detail=OCR_HINT)
            log.info("PDF 文字层不合格（有效字符 %s），自动改用 OCR 识别", meta["quality"]["chars"])
            try:
                pages = ocr_pdf_pages(raw)
            except ImportError as exc:                    # 没装 pymupdf / rapidocr
                raise HTTPException(status_code=400, detail=OCR_HINT) from exc
            full = "\n".join(t for _, t in pages)
            meta.update(source="ocr", ocr=True, quality=text_stats(full))
        if is_low_quality(full):
            raise HTTPException(
                status_code=400,
                detail=f"OCR 后仍只识别出 {text_stats(full)['chars']} 个有效字符，"
                       f"可能是图片太模糊/手写体/整页图表，请换清晰版本或先人工校对。",
            )
        return pages, meta

    if suffix in {".txt", ".md", ".csv", ".json"}:
        for enc in ("utf-8", "gb18030", "latin-1"):     # 医院老系统导出的 txt 常是 GBK/GB18030
            try:
                return [(1, raw.decode(enc))], meta     # 纯文本没有"页"的概念，统一当作第 1 页
            except UnicodeDecodeError:
                continue
        raise HTTPException(status_code=400, detail="文本文件编码无法识别（试过 utf-8 / gb18030）")

    raise HTTPException(
        status_code=400,
        detail=f"暂不支持的文件类型：{suffix}。当前只支持 PDF 和纯文本(.txt/.md/.csv/.json)；"
               f"Word/PPT/Excel/图片请先另存为 PDF 再上传。",
    )


def build_chunks(pages: list[tuple[int, str]]) -> list[dict]:
    """
    逐页切分，并给每段记上"来自第几页"；同时按内容去重（扫描件重复页会产生一模一样的段落）。
    返回 [{"index": 段序号, "page": 页码, "text": 段落正文}]
    """
    chunks: list[dict] = []
    seen: set[str] = set()
    for page_no, text in pages:
        text = (text or "").strip()
        if not text:
            continue
        for piece in SPLITTER.split_text(text):
            key = re.sub(r"\s+", "", piece)
            if not key or key in seen:      # 去重：重复段落既浪费向量存储，又会让 Top-K 被同一段占满
                continue
            seen.add(key)
            chunks.append({"index": len(chunks), "page": page_no, "text": piece})
    return chunks


# ============================================================================
# 7. 向量检索：RAG 的"检索"环节
# ============================================================================
def search_chunks(query: str, top_k: int = TOP_K, doc_ids: list[int] | None = None) -> list[dict]:
    """
    三步：
      1) 把用户问题向量化成 q_vec；
      2) 从 chunks 表取出所有段落向量，按维度过滤（防止换过向量模型后老向量干扰）；
      3) 余弦相似度排序，取 Top-K 作为"参考资料"。
    最简做法是向量就存在 SQLite 里（JSON 文本）；数据量大了换成 Chroma/pgvector/Milvus，
    只需要替换本函数，上层接口完全不用改。
    """
    q_vec = np.asarray(EMBEDDER.embed_query(query), dtype=np.float32)
    q_dim = int(q_vec.shape[0])
    q_norm = float(np.linalg.norm(q_vec)) or 1.0
    q_vec = q_vec / q_norm                                    # 归一化后：点积 == 余弦相似度

    sql = (
        "SELECT c.id AS chunk_id, c.doc_id, c.chunk_index, c.page, c.content, c.embedding, "
        "d.title, d.file_type "
        "FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.dim = ?"
    )
    params: list = [q_dim]
    if doc_ids:                                               # 支持"只在某几份资料里检索"
        sql += f" AND c.doc_id IN ({','.join('?' * len(doc_ids))})"
        params += list(doc_ids)

    with db() as conn:
        rows = conn.execute(sql, params).fetchall()
    if not rows:
        return []

    mat = np.asarray([json.loads(r["embedding"]) for r in rows], dtype=np.float32)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    scores = mat @ q_vec                                      # 一次矩阵乘法算出全部相似度
    # 多取一些候选再按内容去重：扫描件常有重复页，不去重的话 Top-K 会被同一段内容占满
    order = np.argsort(-scores)[: max(1, min(top_k * 3, len(rows)))]

    hits: list[dict] = []
    seen: set[str] = set()
    for i in order:
        key = re.sub(r"\s+", "", rows[i]["content"])[:200]
        if key in seen:
            continue
        seen.add(key)
        hits.append(
            {
                "chunk_id": rows[i]["chunk_id"],
                "doc_id": rows[i]["doc_id"],
                "title": rows[i]["title"],
                "file_type": rows[i]["file_type"],
                "chunk_index": rows[i]["chunk_index"],
                "page": rows[i]["page"] or 1,
                "content": rows[i]["content"],
                "score": round(float(scores[i]), 4),
            }
        )
        if len(hits) >= top_k:
            break
    return hits


# ---------- 8. 复杂问题怎么"找得准"：多查询改写 + 邻块扩展 ----------
QUERY_PROMPT = """你是医学文献检索助手。请把用户的问题改写成 {n} 个用于检索的短查询，每行一个，
不要编号、不要解释、不要引号。要求：覆盖问题的不同方面（定义/分类/病因/临床表现/诊断/治疗/防治），
保留专业术语，每个查询不超过 20 个字。
用户问题：{question}"""


def rewrite_queries(question: str) -> list[str]:
    """
    复杂问题（比如"介绍线虫及其造成的疾病"）的答案在资料里是分散的：
    定义在第二章概述、分类在生活史、具体病种在后面几页。一个向量只能命中其中一块。
    这里先让模型把问题拆成几条不同的检索式，分别召回再合并 —— 这是多查询（Multi-Query）检索。
    失败或离线时退回原问题，绝不阻塞主流程。
    """
    if not MULTI_QUERY or LLM is None:
        return [question]
    try:
        raw = LLM.invoke([HumanMessage(content=QUERY_PROMPT.format(n=MULTI_QUERY_N, question=question))]).content or ""
        queries = [re.sub(r"^[\s\d\.\-、\)\(（）]+", "", line).strip() for line in raw.splitlines()]
        queries = [q for q in queries if 2 <= len(q) <= 40]
        # 原问题放第一位：改写只是补充，不能把原始意图丢掉
        return list(dict.fromkeys([question] + queries))[: MULTI_QUERY_N + 1]
    except Exception as exc:                      # noqa: BLE001
        log.warning("查询改写失败(%s)，退回原问题", exc)
        return [question]


def search_chunks_multi(question: str, top_k: int, doc_ids: list[int] | None = None) -> tuple[list[dict], list[str]]:
    """
    多查询检索：每条检索式各召回一次，按 chunk_id 合并。
    排序用 "相似度 + 命中次数奖励"：被 3 条查询同时命中的段落，显然比只命中 1 条的相关。
    """
    queries = rewrite_queries(question)
    merged: dict[int, dict] = {}
    for q in queries:
        for hit in search_chunks(q, top_k, doc_ids):
            old = merged.get(hit["chunk_id"])
            if old is None:
                hit["hit_by"] = [q]
                merged[hit["chunk_id"]] = hit
            else:
                old["hit_by"].append(q)
                old["score"] = max(old["score"], hit["score"])
    hits = sorted(merged.values(), key=lambda h: h["score"] + 0.05 * (len(h["hit_by"]) - 1), reverse=True)
    if len(queries) > 1:
        log.info("多查询检索：%s 条检索式 → 合并去重 %s 段", len(queries), len(hits))
    return hits[:top_k], queries


def expand_neighbors(hits: list[dict], window: int = NEIGHBOR_WINDOW) -> list[dict]:
    """
    邻块扩展（small-to-big）：医疗问答的答案经常跨段 —— 上一段讲到一半，下一段接着讲。
    命中段的前后各带一段一起喂给模型，比单纯调大 top_k 更划算：
    上下文更连贯，又不会把八竿子打不着的段落塞进 Prompt。
    """
    if window <= 0 or not hits:
        return hits
    with db() as conn:
        for h in hits:
            rows = conn.execute(
                "SELECT content FROM chunks WHERE doc_id = ? AND chunk_index BETWEEN ? AND ? ORDER BY chunk_index",
                (h["doc_id"], h["chunk_index"] - window, h["chunk_index"] + window),
            ).fetchall()
            # 单条不要无限长：太长会挤占 Prompt，反而让模型丢掉"必须标引用"的格式要求
            h["context"] = ("\n".join(r["content"] for r in rows) or h["content"])[:NEIGHBOR_MAX_CHARS]
    return hits


def fit_hits(hits: list[dict], budget: int = MAX_CONTEXT_CHARS) -> list[dict]:
    """按字数预算裁剪命中段：全塞进去会让 7B 模型"看不过来看"，反而更容易跑偏。"""
    kept: list[dict] = []
    used = 0
    for h in hits:
        size = len(h.get("context") or h["content"])
        if kept and used + size > budget:
            continue
        kept.append(h)
        used += size
    return kept or hits[:1]


# ---------- 9. 幻觉体检：判断"有没有瞎说"，不用再请一个模型，靠规则就能看出八成 ----------
def _bigrams(text: str) -> set[str]:
    t = re.sub(r"\s+", "", text or "")
    return {t[i:i + 2] for i in range(len(t) - 1)}


def normalize_citations(text: str) -> str:
    """
    模型常把引用写在句号后面（"……损伤。[3]"），按句号切句就会把 [3] 切到下一句去，
    导致"明明标了引用却算成 0%"。这里统一把句末标点后面的编号挪回标点前面。
    """
    return re.sub(r"([。！？；])((?:\s*\[\d+\])+)", r"\2\1", text or "")


def _overlap(text: str, source: str) -> float:
    """字符 2-gram 重合度：答案句有多少"双字组合"能在引用段落里找到。"""
    a = _bigrams(text)
    return len(a & _bigrams(source)) / len(a) if a else 0.0


def check_grounding(answer: str, hits: list[dict]) -> dict:
    """
    三个可量化的检查，直接告诉用户"哪句话可能没依据"：
      1) 没标引用编号的句子 → 最可能凭空生成的地方；
      2) 引用编号越界（模型写了 [5] 但只有 4 段）→ 明显编造；
      3) 句子和它引用的段落重合度太低 → 标"疑似无依据"，人工重点看这几句。
    这样医生只需要复核被标出来的 3~5 句，而不是把整篇回答重新读一遍。
    """
    sentences = [s.strip() for s in re.split(r"(?<=[。！？；\n])", normalize_citations(answer)) if s.strip()]
    cited = 0
    uncited: list[str] = []
    weak: list[str] = []
    invalid: list[str] = []

    for sent in sentences:
        refs = [int(x) for x in re.findall(r"\[(\d+)\]", sent)]
        body = re.sub(r"\[\d+\]", "", sent)
        body = re.sub(r"[\*#>]+|^[\s\-]+", "", body).strip()     # 去掉 markdown 记号，避免标题被当成句子
        if not body or body.endswith("：") or body.endswith(":"):
            continue                            # 小标题 / 引导句，不计入统计
        if not refs:
            if len(body) >= 12:                 # 太短的过渡句（"如下："）不计较
                uncited.append(body[:80])
            continue
        cited += 1
        support = 0.0
        for r in refs:
            if not 1 <= r <= len(hits):
                invalid.append(f"[{r}]")
                continue
            support = max(support, _overlap(body, hits[r - 1]["content"]))
        if support < MIN_SUPPORT_RATIO:
            weak.append(body[:80])

    total = max(len(sentences), 1)
    coverage = round(cited / total, 3)
    return {
        "sentences": len(sentences),
        "cited": cited,
        "coverage": coverage,                   # 引用覆盖率：有出处的句子占多少
        "uncited": uncited[:5],                 # 没标出处的句子
        "weak": weak[:5],                       # 标了出处但和原文对不上的句子
        "invalid_refs": sorted(set(invalid)),   # 编出来的引用编号
        "verdict": "ok" if (coverage >= 0.6 and not invalid and len(weak) <= 1) else "建议人工核对",
    }


def split_sentences(text: str) -> list[str]:
    """按中文句读切句，并清掉 markdown 记号；太短的标题/引导句不算句子。"""
    parts = re.split(r"(?<=[。！？；\n])", normalize_citations(text))
    out: list[str] = []
    for s in parts:
        s = re.sub(r"\[\d+\]", "", s)
        s = re.sub(r"[\*#>]+|^[\s\-]+", "", s).strip()
        if len(s) >= 12 and not s.endswith(("：", ":")):
            out.append(s)
    return out


def attribute_sentences(answer: str, hits: list[dict], limit: int = 5) -> list[dict]:
    """
    确定性归因（不依赖模型自觉）：把答案逐句与每段资料做 2-gram 重合度比对，
    得出"这句话最像出自第几页、像不像"，按最可疑的排前面。
    —— 模型不标引用时，这一列就是给医生的复核清单；标了引用也能用来交叉验证。
    """
    ranked: list[dict] = []
    for sent in split_sentences(answer):
        scores = sorted(
            ((_overlap(sent, h.get("context") or h["content"]), i) for i, h in enumerate(hits, 1)),
            reverse=True,
        )
        best_score, best_idx = scores[0]
        best_page = hits[best_idx - 1].get("page")
        # 顺带给出"次优命中"的页码：答案常常落在命中段的上一页/下一页（分页切断），一起给出来更好核查
        alt_pages: list[int] = []
        for sc, i in scores[1:4]:
            page = hits[i - 1].get("page")
            if page and page != best_page and page not in alt_pages:
                alt_pages.append(page)
        ranked.append(
            {
                "sentence": sent[:80],
                "source": best_idx,
                "doc_id": hits[best_idx - 1].get("doc_id"),
                "file_type": hits[best_idx - 1].get("file_type"),
                "page": best_page,
                "alt_pages": alt_pages[:2],
                "score": round(best_score, 3),
                "supported": best_score >= MIN_SUPPORT_RATIO,
            }
        )
    ranked.sort(key=lambda x: x["score"])      # 越不吻合越靠前，人工先看这些
    return ranked[:limit]


REPAIR_PROMPT = """你是一名严谨的医学编辑。下面这份回答没有按要求标注资料来源，请加工它：
1. 保持原有的医学内容与结构，**不要新增**【资料】以外的内容；
2. 给每个结论句补上出处编号，例如 [1] 或 [1][3]；
3. 凡是【资料】里找不到依据的句子，直接删掉；
4. 输出纯文本分点，不要解释你的修改过程。

【资料】
{context}

【待加工的回答】
{draft}"""


def repair_citations(hits: list[dict], draft: str) -> str:
    """
    自检失败后的补救：让模型对照同一份资料，把答案重写一遍并补上编号。
    这比"直接告诉用户模型没标出处"友好得多 —— 大多数情况下补完就能追溯了。
    """
    try:
        prompt = REPAIR_PROMPT.format(context=build_context(hits), draft=draft)
        out = LLM.invoke(
            [SystemMessage(content="你只输出加工后的回答本身，不要任何解释。"), HumanMessage(content=prompt)]
        ).content
        return (out or draft).strip()
    except Exception as exc:                      # noqa: BLE001
        log.warning("自动补引用失败(%s)，保留原回答", exc)
        return draft


# ============================================================================
# 8. 生成答案：检索结果拼 Prompt → 大模型；期间可能触发工具调用（Agent 循环）
# ============================================================================
def build_context(hits: list[dict]) -> str:
    """把检索结果拼成 Prompt 里的【资料】区：带编号 + 出处页码，编号要能对上答案里的 [n]。"""
    return "\n\n".join(
        f"[{i}] 出处《{h['title']}》第 {h.get('page') or 1} 页：\n{h.get('context') or h['content']}"
        for i, h in enumerate(hits, 1)
    ) or "（知识库为空，没有检索到任何资料）"


def generate_answer(question: str, hits: list[dict]) -> dict:
    """返回 {answer, tool_calls, model}。没有大模型时降级为"摘录检索原文"。"""
    context = build_context(hits)

    user_prompt = f"【资料】\n{context}\n\n【问题】\n{question}\n\n【回答】"
    tool_trace: list[dict] = []

    # ---- 降级路径：Ollama 不可用时，直接把最相关原文返回，保证服务可用 ----
    if LLM is None:
        # 问时间这种"不依赖知识库"的问题，离线也能靠规则触发工具，方便演示工具调用链路
        if time_intent(question):
            now = get_current_time.invoke({"timezone": "Asia/Shanghai"})
            tool_trace.append(
                {"name": "get_current_time", "args": {"timezone": "Asia/Shanghai"}, "mode": "rule-fallback"}
            )
            return {
                "answer": f"【离线模式】未连接大模型，此处直接给出工具调用结果：当前时间 {now}",
                "tool_calls": tool_trace,
                "model": LLM_NAME,
            }
        answer = (
            "【离线模式】未连接到大模型，先给出检索到的最相关原文：\n\n" + hits[0]["content"]
            if hits
            else "【离线模式】知识库为空，请先上传医疗资料。"
        )
        return {"answer": answer, "tool_calls": [], "model": LLM_NAME}

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
    hint_sent = False

    # ---- 主路径：把工具"绑"给模型，由模型自己决定是否调用（真正的 Function Calling）----
    if ENABLE_TOOLS:
        try:
            llm_with_tools = LLM.bind_tools(TOOLS)     # 把工具说明书注入请求
            for _ in range(MAX_TOOL_ROUNDS):           # 最多循环几轮，防模型反复调工具死循环
                ai = llm_with_tools.invoke(messages)
                messages.append(ai)

                if ai.tool_calls:                      # 模型说"我要先查时间"
                    for call in ai.tool_calls:
                        name, args = call["name"], call.get("args") or {}
                        tool_trace.append({"name": name, "args": args, "mode": "model-driven"})
                        log.info("工具调用: %s(%s)", name, args)
                        tool = TOOL_MAP.get(name)
                        result = tool.invoke(args) if tool else f"未知工具：{name}"
                        # 把工具执行结果回灌给模型，让它基于真实时间继续作答（这就是 Agent 的 ReAct 循环）
                        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
                    continue

                # 模型没主动调工具，但问题明显在问时间 → 人工补一次工具结果再问一遍（兜底）
                if time_intent(question) and not tool_trace and not hint_sent:
                    hint_sent = True
                    result = get_current_time.invoke({"timezone": "Asia/Shanghai"})
                    tool_trace.append({"name": "get_current_time", "args": {"timezone": "Asia/Shanghai"}, "mode": "rule-fallback"})
                    messages.append(HumanMessage(content=f"（系统补充：当前时间由工具查询为 {result}，请据此回答。）"))
                    continue

                return {"answer": ai.content, "tool_calls": tool_trace, "model": LLM_NAME}
        except Exception as exc:                       # noqa: BLE001
            # 常见原因：模型不支持 tools（如老版本 qwen）、Ollama 版本太旧。降级成普通问答。
            log.warning("工具调用链路失败(%s)，退化为普通问答", exc)
            messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]

    # ---- 兜底路径：模型不支持工具时，用关键词规则触发工具，把结果塞进 Prompt ----
    if time_intent(question):
        result = get_current_time.invoke({"timezone": "Asia/Shanghai"})
        tool_trace.append({"name": "get_current_time", "args": {"timezone": "Asia/Shanghai"}, "mode": "rule-fallback"})
        messages.append(HumanMessage(content=f"（系统补充：当前时间由工具查询为 {result}，请据此回答。）"))

    ai = LLM.invoke(messages)
    return {"answer": ai.content, "tool_calls": tool_trace, "model": LLM_NAME}


# ============================================================================
# 9. FastAPI 应用：接口层（前端/小程序/别的系统都调它）
# ============================================================================
EMBEDDER: Embeddings = HashEmbeddings()      # 默认兜底，lifespan 启动时会被真模型替换
EMBED_NAME = "hash-fallback:512"
LLM = None
LLM_NAME = "offline-extractive"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化：建库建表 + 探活模型；关闭时留个日志（容器优雅退出）。"""
    global EMBEDDER, EMBED_NAME, LLM, LLM_NAME
    init_db()
    EMBEDDER, EMBED_NAME = build_embedder()
    LLM, LLM_NAME = build_llm()
    log.info("=" * 70)
    log.info("MediRAG 启动完成 | 向量模型=%s | 大模型=%s | 数据库=%s", EMBED_NAME, LLM_NAME, DB_PATH)
    log.info("=" * 70)
    yield
    log.info("MediRAG 已停止")


app = FastAPI(title="MediRAG 医疗知识库问答", version="1.0.0", lifespan=lifespan)

# 允许跨域：方便前端本地起在 5500 端口、或小程序/其他系统直接对接（生产按需收紧成白名单）
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def access_log(request, call_next):
    """访问日志中间件：每个请求记录 方法/路径/状态码/耗时，慢接口一眼看得出来。"""
    start = time.perf_counter()
    response = await call_next(request)
    cost_ms = (time.perf_counter() - start) * 1000
    log.info("HTTP %s %s -> %s | %.1fms", request.method, request.url.path, response.status_code, cost_ms)
    response.headers["X-Process-Time-ms"] = f"{cost_ms:.1f}"     # 同时回传前端，方便定位慢在哪
    return response


# ---------------------------- 请求/响应模型（Pydantic 自动做参数校验 + 生成接口文档） ----------------------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: int = Field(TOP_K, ge=1, le=10, description="检索几段资料")
    doc_ids: list[int] | None = Field(None, description="只在指定资料里检索，留空=全库")


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    tool_calls: list[dict]
    queries: list[str] = []          # 实际用到的检索式（多查询改写的产物）
    grounding: dict = {}             # 引用校验结果：覆盖率 / 未标注句子 / 疑似无依据句子
    disclaimer: str = DISCLAIMER     # 合规提示：随接口返回，保证各客户端口径一致
    model: str
    latency_ms: float


# ---------------------------- 接口 1：首页（直接返回前端页面） ----------------------------
@app.get("/", include_in_schema=False)
def index():
    page = STATIC_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="前端文件 static/index.html 不存在")
    # no-store：改完前端刷新就能看到，不会被浏览器缓存住（否则用户会一直看到旧界面）
    return FileResponse(page, headers={"Cache-Control": "no-store, must-revalidate"})


# ---------------------------- 接口 2：健康检查（部署探活用） ----------------------------
@app.get("/api/health", summary="健康检查")
def health():
    with db() as conn:
        docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return {
        "status": "ok",
        "time": now_str(),
        "embed_model": EMBED_NAME,
        "llm": LLM_NAME,
        "documents": docs,
        "chunks": chunks,
    }


# ---------------------------- 接口 3：上传资料 → 切分 → 向量化 → 入库 ----------------------------
@app.post("/api/upload", summary="上传医疗资料并入库")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件超过 {MAX_UPLOAD_MB}MB 限制")

    filename = Path(file.filename or "unnamed.txt").name      # 只取文件名，防目录穿越

    # 内容判重：同一份文件（哪怕改了文件名）直接拒绝，省掉后面动辄几分钟的 OCR 和一份磁盘副本
    digest = hashlib.sha256(raw).hexdigest()
    with db() as conn:
        dup = conn.execute("SELECT id, title FROM documents WHERE hash = ?", (digest,)).fetchone()
    if dup:
        log.info("重复上传被拒绝：%s（已存在 doc_id=%s）", filename, dup["id"])
        raise HTTPException(
            status_code=409,
            detail=f"该文件已入库（资料 #{dup['id']}《{dup['title']}》），无需重复上传；如需重新解析请先删除旧资料。",
        )

    # 解析 + 必要时 OCR：都是 CPU 密集型操作，必须丢线程池，否则会卡死整个服务
    pages, meta = await run_in_threadpool(load_document_pages, filename, raw)

    chunks = build_chunks(pages)                              # ① 逐页切分，记录页码，顺带去重
    if not chunks:
        raise HTTPException(status_code=400, detail="解析后的文本为空，无法切分")
    # ② 向量化：embed_documents 是阻塞的 HTTP 调用，同样丢到线程池
    vectors = await run_in_threadpool(EMBEDDER.embed_documents, [c["text"] for c in chunks])

    total_chars = sum(len(t) for _, t in pages)
    title = Path(filename).stem
    with db() as conn:                                        # ③ 入库：先写资料表，再批量写段落表
        cur = conn.execute(
            "INSERT INTO documents(title, filename, file_type, size_bytes, chunk_count, page_count, "
            "embed_model, source, hash, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (title, filename, Path(filename).suffix.lower(), len(raw), len(chunks), len(pages),
             EMBED_NAME, meta["source"], digest, now_str()),
        )
        doc_id = int(cur.lastrowid)
        conn.executemany(
            "INSERT INTO chunks(doc_id, chunk_index, page, content, embedding, dim) VALUES (?,?,?,?,?,?)",
            [
                (doc_id, c["index"], c["page"], c["text"], json.dumps(vec, ensure_ascii=False), len(vec))
                for c, vec in zip(chunks, vectors)
            ],
        )

    # 原始文件留档。文件名 = {自增id}_{原名}：id 只用来保证唯一，**不是**界面上那个编号；
    # 真正的文件路径记在 documents.stored_name 里，所以以后想改名重组也不会断开关联。
    stored_name = f"{doc_id}_{filename}"
    (UPLOAD_DIR / stored_name).write_bytes(raw)
    with db() as conn:
        conn.execute("UPDATE documents SET stored_name = ? WHERE id = ?", (stored_name, doc_id))

    log.info("资料入库 doc_id=%s 文件=%s 来源=%s 页数=%s 字符=%s 段数=%s 向量模型=%s",
             doc_id, filename, meta["source"], len(pages), total_chars, len(chunks), EMBED_NAME)
    return {
        "doc_id": doc_id,
        "title": title,
        "filename": filename,
        "pages": len(pages),
        "chars": total_chars,
        "chunks": len(chunks),
        "source": meta["source"],           # text-layer=电子版直接抽取；ocr=扫描件识别
        "ocr": meta["ocr"],
        "text_quality": meta.get("quality"),
        "embed_model": EMBED_NAME,
        "created_at": now_str(),
    }


# ---------------------------- 接口 4：提问 → 检索 → 生成 → 记录日志 ----------------------------
@app.post("/api/ask", response_model=AskResponse, summary="基于知识库提问")
async def ask(req: AskRequest):
    t0 = time.perf_counter()
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    # ① 检索：复杂问题先拆成多条检索式，各自召回再合并（Multi-Query）
    hits, queries = await run_in_threadpool(search_chunks_multi, question, req.top_k, req.doc_ids)
    # ② 邻块扩展：把命中段的前后一段也带上，应对"答案跨段"的情况
    hits = await run_in_threadpool(expand_neighbors, hits)
    hits = await run_in_threadpool(fit_hits, hits)          # 按字数预算裁剪，别把模型"撑死"
    # ③ 生成（可能含工具调用）
    result = await run_in_threadpool(generate_answer, question, hits)
    # ④ 幻觉体检：算引用覆盖率、找出没标注/对不上的句子
    grounding = check_grounding(result["answer"], hits)
    # ⑤ 自检补救：覆盖率太低说明模型没照着标注，自动重写一次（只采纳"变好"的结果）
    if CITATION_REPAIR and LLM is not None and grounding["coverage"] < MIN_COVERAGE:
        log.warning("引用覆盖率仅 %s，触发一次自动补引用", grounding["coverage"])
        fixed = await run_in_threadpool(repair_citations, hits, result["answer"])
        patched = check_grounding(fixed, hits)
        if patched["coverage"] > grounding["coverage"]:
            old_cov = grounding["coverage"]
            result["answer"] = fixed
            patched["repaired"] = True
            grounding = patched
            log.info("补引用生效：覆盖率 %s → %s", old_cov, patched["coverage"])

    # ⑥ 逐句归因：不管模型标没标引用，都给出"每句最像出自第几页"，供人工只挑可疑的几句核对
    grounding["attribution"] = await run_in_threadpool(attribute_sentences, result["answer"], hits)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    sources = [
        {
            "index": i,
            "doc_id": h["doc_id"],
            "chunk_id": h["chunk_id"],
            "title": h["title"],
            "file_type": h.get("file_type"),      # 前端据此判断"能不能跳到第 N 页"（只有 PDF 可以）
            "page": h.get("page") or 1,
            "chunk_index": h["chunk_index"] + 1,
            "score": h["score"],
            "hit_by": h.get("hit_by", []),          # 被哪几条检索式命中（命中越多越相关）
            "preview": h["content"][:120],
        }
        for i, h in enumerate(hits, 1)
    ]

    with db() as conn:      # 问答记录入库：以后可做热门问题统计、人工复核、效果评估
        conn.execute(
            "INSERT INTO qa_logs(question, answer, sources, tool_calls, queries, grounding, "
            "model, latency_ms, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                question,
                result["answer"],
                json.dumps(sources, ensure_ascii=False),
                json.dumps(result["tool_calls"], ensure_ascii=False),
                json.dumps(queries, ensure_ascii=False),
                json.dumps(grounding, ensure_ascii=False),
                result["model"],
                latency_ms,
                now_str(),
            ),
        )
        # 只保留最近 N 条：日志表不会无限增长（SQLite 删除后空间会复用，文件不会一直变大）
        conn.execute(
            "DELETE FROM qa_logs WHERE id NOT IN (SELECT id FROM qa_logs ORDER BY id DESC LIMIT ?)",
            (QA_LOG_KEEP,),
        )

    log.info("问答完成 | 耗时=%.1fms | 检索式=%s条 | 召回=%s段 | 引用覆盖率=%s | 结论=%s | 问题=%s",
             latency_ms, len(queries), len(hits), grounding["coverage"], grounding["verdict"], question[:50])

    return AskResponse(
        answer=result["answer"], sources=sources, tool_calls=result["tool_calls"],
        queries=queries, grounding=grounding, disclaimer=DISCLAIMER,
        model=result["model"], latency_ms=latency_ms,
    )


# ---------------------------- 接口 5：资料列表 / 问答日志（演示 DB 查询） ----------------------------
@app.get("/api/documents", summary="已入库资料列表")
def list_documents():
    with db() as conn:
        # 排序：有自定义编号的按编号在前，未编号的按上传顺序排在后面
        rows = conn.execute(
            "SELECT id, title, filename, file_type, size_bytes, chunk_count, page_count, "
            "embed_model, source, serial, created_at "
            "FROM documents ORDER BY COALESCE(serial, 999999) ASC, id ASC"
        ).fetchall()
    items = assign_display_nos(rows)
    return {"total": len(items), "items": items}


class DocPatch(BaseModel):
    """目前只支持改编号；留空 serial 表示恢复自动编号。"""
    serial: int | None = Field(None, ge=1, le=999, description="自定义编号 1-999")


@app.patch("/api/documents/{doc_id}", summary="修改资料编号")
def update_document(doc_id: int, patch: DocPatch):
    with db() as conn:
        if not conn.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone():
            raise HTTPException(status_code=404, detail=f"资料 {doc_id} 不存在")
        if patch.serial is not None:
            dup = conn.execute(
                "SELECT id FROM documents WHERE serial = ? AND id != ?", (patch.serial, doc_id)
            ).fetchone()
            if dup:
                raise HTTPException(status_code=409, detail=f"编号 {patch.serial} 已被资料 #{dup['id']} 占用")
        conn.execute("UPDATE documents SET serial = ? WHERE id = ?", (patch.serial, doc_id))
    log.info("资料编号修改 doc_id=%s serial=%s", doc_id, patch.serial)
    return {"id": doc_id, "serial": patch.serial}


@app.post("/api/documents/renumber", summary="按上传顺序把编号重排为 1..N")
def renumber_documents():
    """一键把编号整理成 1,2,3…（按上传先后），解决"删了又传导致编号乱跳"的问题。"""
    with db() as conn:
        rows = conn.execute("SELECT id FROM documents ORDER BY id").fetchall()
        for i, row in enumerate(rows, 1):
            conn.execute("UPDATE documents SET serial = ? WHERE id = ?", (i, row["id"]))
    log.info("资料编号重排：共 %s 份 → 1..%s", len(rows), len(rows))
    return {"renumbered": len(rows)}


# ---------------------------- 原始文件定位：数据库记路径，不靠文件名解析 ----------------------------
def document_path(stored_name: str | None, doc_id: int) -> Path | None:
    """
    找到某份资料的原始文件。
    优先用数据库里记的 stored_name（改名后也不会丢）；老数据没有该字段，
    就按历史命名规则 {id}_* 兜底找一次。
    """
    if stored_name:
        path = UPLOAD_DIR / stored_name
        if path.exists():
            return path
    matches = sorted(UPLOAD_DIR.glob(f"{doc_id}_*"))
    return matches[0] if matches else None


def safe_filename(text: str, limit: int = 40) -> str:
    """把标题变成能当文件名用的字符串：去掉路径分隔符和 Windows 非法字符。"""
    cleaned = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", text or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:limit] or "资料"


def assign_display_nos(rows: list[sqlite3.Row]) -> list[dict]:
    """
    统一算"显示编号"：有自定义 serial 用 serial，否则取最小的空闲号（跳过已被占用的）。
    列表和文件整理都用这个函数，保证两边编号一致。
    """
    used = {r["serial"] for r in rows if r["serial"] is not None}
    auto = 1
    out: list[dict] = []
    for row in rows:
        item = dict(row)
        if item["serial"] is not None:
            item["display_no"] = item["serial"]
        else:
            while auto in used:
                auto += 1
            item["display_no"] = auto
            used.add(auto)
            auto += 1
        out.append(item)
    out.sort(key=lambda x: x["display_no"])
    return out


@app.post("/api/documents/tidy", summary="按编号整理磁盘文件名（重命名，不断开关联）")
def tidy_filenames():
    """
    把 uploads 里的文件重命名成 "{显示编号}_{标题}.pdf"，让磁盘和界面编号对得上。
    因为文件路径记在 documents.stored_name 里，改名不会影响删除和"打开原文"。
    """
    with db() as conn:
        rows = conn.execute(
            "SELECT id, title, filename, serial, stored_name FROM documents "
            "ORDER BY COALESCE(serial, 999999) ASC, id ASC"
        ).fetchall()
        docs = assign_display_nos(rows)

        renamed = []
        for doc in docs:
            src = document_path(doc["stored_name"], doc["id"])
            if not src:
                continue
            ext = Path(doc["filename"]).suffix.lower() or ".pdf"
            base = f"{doc['display_no']}_{safe_filename(doc['title'])}"
            dst = UPLOAD_DIR / f"{base}{ext}"
            n = 1
            while dst.exists() and dst.resolve() != src.resolve():   # 避免重名覆盖
                dst = UPLOAD_DIR / f"{base}_{n}{ext}"
                n += 1
            if dst.resolve() == src.resolve():
                continue
            try:
                src.rename(dst)
            except OSError as exc:
                log.warning("重命名失败 %s → %s: %s", src.name, dst.name, exc)
                continue
            conn.execute("UPDATE documents SET stored_name = ? WHERE id = ?", (dst.name, doc["id"]))
            renamed.append({"id": doc["id"], "from": src.name, "to": dst.name})

    log.info("整理文件名：重命名 %s 个", len(renamed))
    return {"renamed": len(renamed), "files": renamed}


@app.get("/api/documents/{doc_id}/file", summary="打开原始资料（配合 #page=N 直接跳到对应页）")
def download_document(doc_id: int):
    """
    把上传时的原始 PDF 回传给浏览器，前端用 <a href="...#page=4"> 打开，
    Chrome/Edge 内置阅读器会自动跳到第 4 页 —— 医生核对时不用自己翻。
    """
    with db() as conn:
        row = conn.execute("SELECT stored_name, filename FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"资料 {doc_id} 不存在")
    path = document_path(row["stored_name"], doc_id)
    if not path:
        raise HTTPException(status_code=404, detail="原始文件不存在（可能已被清理）")
    # 按真实类型返回：若统一返回 application/pdf，浏览器打开 txt 会按 PDF 解析而报错
    media = {
        ".pdf": "application/pdf",
        ".csv": "text/csv",
        ".json": "application/json",
        ".txt": "text/plain",
        ".md": "text/plain",
    }.get(path.suffix.lower(), "application/octet-stream")

    # 纯文本要带上正确编码：医院导出的 txt 常是 GB18030，写死 utf-8 会一片乱码
    if media.startswith("text/") and path.suffix.lower() != ".json":
        head = path.read_bytes()[:8192]
        try:
            head.decode("utf-8")
            charset = "utf-8"
        except UnicodeDecodeError:
            charset = "gb18030"
        media = f"{media}; charset={charset}"
    elif media == "application/json":
        media = "application/json; charset=utf-8"

    # 不传 filename → 不产生 attachment 头，浏览器会内联预览而不是下载
    return FileResponse(path, media_type=media)


@app.delete("/api/documents/{doc_id}", summary="删除资料（连同向量和原始文件）")
def delete_document(doc_id: int):
    """清理用：误传、重复上传、入库了垃圾文本时删掉。**必须连磁盘上的原始文件一起删**，
    否则每传一次就多留一份几十 MB 的 PDF，data/uploads 会越堆越大。"""
    with db() as conn:
        row = conn.execute("SELECT title, stored_name FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"资料 {doc_id} 不存在")
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))   # 显式删，不依赖外键开关
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    removed = 0
    candidates = set()
    if row["stored_name"]:
        candidates.add(UPLOAD_DIR / row["stored_name"])
    candidates.update(UPLOAD_DIR.glob(f"{doc_id}_*"))     # 老数据兜底
    for f in candidates:
        if f.exists():
            try:
                f.unlink()
                removed += 1
            except OSError as exc:
                log.warning("删除原始文件失败 %s: %s", f.name, exc)
    log.info("资料已删除 doc_id=%s 标题=%s 清理文件=%s 个", doc_id, row["title"], removed)
    return {"deleted": doc_id, "title": row["title"], "deleted_files": removed}


@app.delete("/api/documents", summary="清空全部资料（重新导入用）")
def clear_documents():
    """一键清空：段落、资料、原始文件全删；问答日志（qa_logs）保留做历史分析。"""
    with db() as conn:
        docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM documents")

    removed = 0
    for f in UPLOAD_DIR.glob("*"):
        if f.is_file():
            try:
                f.unlink()
                removed += 1
            except OSError as exc:
                log.warning("清空时删除文件失败 %s: %s", f.name, exc)
    log.warning("清空知识库：资料 %s 份 / 段落 %s 段 / 文件 %s 个", docs, chunks, removed)
    return {"deleted_documents": docs, "deleted_chunks": chunks, "deleted_files": removed}


@app.get("/api/qa-logs", summary="最近问答日志（只保留最近 N 条）")
def list_qa_logs(limit: int = QA_LOG_KEEP):
    limit = max(1, min(limit, QA_LOG_KEEP))
    with db() as conn:
        rows = conn.execute(
            "SELECT id, question, answer, sources, tool_calls, queries, grounding, model, latency_ms, created_at "
            "FROM qa_logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    items = []
    with db() as conn:
        # 老日志的 sources 里没有 file_type，这里按 doc_id 回填一次，历史记录也能点开原文
        type_map = {r["id"]: r["file_type"] for r in conn.execute("SELECT id, file_type FROM documents")}
    for row in rows:
        item = dict(row)
        for key in ("sources", "tool_calls", "queries", "grounding"):
            try:
                item[key] = json.loads(item[key]) if item[key] else None
            except (TypeError, ValueError):
                item[key] = None
        for src in (item.get("sources") or []):
            if not src.get("file_type"):
                src["file_type"] = type_map.get(src.get("doc_id"))
        items.append(item)
    return {"total": len(items), "keep": QA_LOG_KEEP, "items": items}


if __name__ == "__main__":
    # 直接 python app.py 也能起（等价于 uvicorn app:app --host 0.0.0.0 --port 8000）
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
