# 运行效果与演示材料

本目录存放项目的运行证据：真实接口记录 + 界面截图 + 架构图。

- 架构图：[`architecture.svg`](architecture.svg)（README 里可直接引用）
- 截图：放在 `docs/images/` 下，按下面清单命名即可自动被本文引用

## 一、真实接口记录

以下为**实际运行**输出（Ollama + qwen2.5:7b + bge-m3，本机 CPU）。

### 1) 健康检查

```bash
curl http://localhost:8000/api/health
```

```json
{"status":"ok","time":"2026-10-23 00:16:21","embed_model":"ollama:bge-m3","llm":"ollama:qwen2.5:7b","documents":3,"chunks":308}
```

### 2) 上传扫描件 PDF（自动 OCR）

```bash
curl -F "file=@人体寄生虫学.pdf" http://localhost:8000/api/upload
```

```json
{
  "doc_id": 12,
  "title": "人体寄生虫学",
  "pages": 24,
  "chars": 20413,
  "chunks": 50,
  "source": "ocr",
  "ocr": true,
  "text_quality": {"chars": 20413, "cjk_ratio": 0.756, "latin_ratio": 0.048},
  "embed_model": "ollama:bge-m3"
}
```

> 这份 PDF 是手机拍照导出的图片型 PDF：文字层只有 66 个乱码字符，自动切 OCR 后识别出 20413 字。

### 3) 提问（多查询检索 + 页码出处 + 引用校验）

```bash
curl -X POST http://localhost:8000/api/ask -H "Content-Type: application/json" \
  -d '{"question":"介绍线虫及其造成的疾病","top_k":10}'
```

关键字段（真实返回，节选）：

```json
{
  "queries": ["介绍线虫及其造成的疾病", "线虫的定义", "线虫的分类", "线虫的病因"],
  "sources": [
    {"index": 1, "title": "人体寄生虫学", "file_type": ".pdf", "page": 5, "chunk_index": 10, "score": 0.6602},
    {"index": 2, "title": "人体寄生虫学", "file_type": ".pdf", "page": 5, "chunk_index": 9,  "score": 0.6459},
    {"index": 3, "title": "人体寄生虫学", "file_type": ".pdf", "page": 6, "chunk_index": 11, "score": 0.634}
  ],
  "grounding": {"sentences": 23, "cited": 12, "coverage": 0.522, "verdict": "建议人工核对"},
  "model": "ollama:qwen2.5:7b",
  "latency_ms": 14213.6
}
```

**留意 Top-K=10 但只返回 3 条来源**：候选取前 10 段后，还要过 `MAX_CONTEXT_CHARS=2500` 的字数预算，
每段邻块扩展后最多 800 字，2500 字只装得下 3 段 —— 日志里会打印 `召回=3段`。

### 4) 重复上传被拒

```bash
curl -F "file=@人体寄生虫学.pdf" http://localhost:8000/api/upload   # 再传一次
```

```json
{"detail": "该文件已入库（资料 #10《人体寄生虫学》），无需重复上传；如需重新解析请先删除旧资料。"}
```

HTTP 状态码 **409**；改名后重传同样会被拦（按内容 sha256 判重）。

### 5) 工具调用（问时间）

```bash
curl -X POST http://localhost:8000/api/ask -H "Content-Type: application/json" \
  -d '{"question":"现在几点了？"}'
```

```json
{"answer": "当前时间 2026-10-23 00:12:31 Friday", 
 "tool_calls": [{"name": "get_current_time", "args": {"timezone": "Asia/Shanghai"}, "mode": "model-driven"}]}
```

## 二、界面截图清单

把截图放到 `docs/images/`，用下列文件名即可直接显示在本文中。

### 1. 主问答界面（回答 + 引用条 + 校验徽章）

![主问答界面](images/01-qa.png)

> 建议截图内容：问题"介绍线虫及其造成的疾病"的回答，包含 `[1][2][3]` 引用、下方"📎 出处"引用条、
> 黄色核对三步提示、橙色引用校验条。

### 2. 引用校验明细（展开）

![引用校验明细](images/02-grounding.png)

> 点开"展开明细 ▾"，展示未标出处的句子、弱支撑句、逐句定位（含相邻页链接）。

### 3. 资料管理（编号 / 多选删除 / 清空）

![资料管理](images/03-docs.png)

> 展示编号就地编辑、勾选框、工具栏（刷新/全选/删除所选/重排编号/整理文件/清空）、列表滚动条。

### 4. 扫描件 OCR 入库

![扫描件 OCR](images/04-ocr.png)

> 上传扫描版 PDF 后提示"✅ 入库成功：人体寄生虫学，24 页 / 20413 字 → 50 段（OCR 识别）"。

### 5. Swagger 接口文档

![Swagger](images/05-swagger.png)

> `http://localhost:8000/docs`，12 个接口自动生成的交互式文档。

## 三、怎么截这些图（Windows）

1. 浏览器打开 `http://127.0.0.1:8000`，按 `Win + Shift + S` 框选 → 自动进剪贴板；
2. 粘贴到画图/微信截图工具，另存为 PNG，放到 `docs/images/`；
3. 文件名与上面一致，GitHub 上的 README 与本文就会自动显示。

建议浏览器缩放 100%、窗口宽度 ≥ 1200px，左侧"资料管理"和右侧"回答"都能完整入镜。
