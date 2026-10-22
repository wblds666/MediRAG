# -*- coding: utf-8 -*-
"""
离线把 PDF 转成 txt（扫描件会自动 OCR），再用网页上传 txt 入库。
—— 为什么需要它：接口里 OCR 是"同步"的，几百页的书会等太久；离线转一次，
   之后上传 txt 是秒级入库，也方便你人工校对识别错误。

用法：
    # 整本转换（扫描件自动识别）
    python tools/pdf_to_text.py "D:\\资料_医学\\人体寄生虫学.pdf" -o out/人体寄生虫学.txt

    # 只转第 1~50 页，并降低 DPI 提速
    python tools/pdf_to_text.py "D:\\资料_医学\\某教材.pdf" -o out/a.txt --start 1 --end 50 --dpi 150

    # 电子版（有文字层）加 --fast 跳过 OCR，直接抽文字，几秒完成
    python tools/pdf_to_text.py "电子版.pdf" -o out/b.txt --fast

依赖：pip install pymupdf rapidocr-onnxruntime
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def open_pymupdf():
    """兼容 PyMuPDF 新旧包名。"""
    try:
        import pymupdf

        return pymupdf
    except ImportError:
        import fitz

        return fitz


def main() -> int:
    ap = argparse.ArgumentParser(description="PDF → txt（扫描件自动 OCR）")
    ap.add_argument("pdf", help="输入 PDF 路径")
    ap.add_argument("-o", "--out", required=True, help="输出 txt 路径")
    ap.add_argument("--start", type=int, default=1, help="起始页（从 1 开始）")
    ap.add_argument("--end", type=int, default=0, help="结束页（0 = 到最后一页）")
    ap.add_argument("--dpi", type=int, default=200, help="OCR 渲染 DPI，越大越准越慢")
    ap.add_argument("--fast", action="store_true", help="只抽文字层，不 OCR（仅适用于电子版）")
    args = ap.parse_args()

    pymupdf = open_pymupdf()
    src = Path(args.pdf)
    if not src.exists():
        print(f"文件不存在：{src}")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    total_chars = 0
    ocr_pages = 0

    with pymupdf.open(src) as doc:
        begin = max(1, args.start) - 1
        finish = min(len(doc), args.end) if args.end else len(doc)
        print(f"共 {len(doc)} 页，处理第 {begin + 1} ~ {finish} 页")

        engine = None
        with out.open("w", encoding="utf-8") as f:
            for i in range(begin, finish):
                page = doc[i]
                text = page.get_text("text")

                # 文字层太少 → 认为是扫描页，走 OCR
                if not args.fast and len(text.strip()) < 50:
                    if engine is None:
                        from rapidocr_onnxruntime import RapidOCR

                        engine = RapidOCR()
                        print("已加载 OCR 模型")
                    pix = page.get_pixmap(dpi=args.dpi)
                    result, _ = engine(pix.tobytes("png"))
                    text = "\n".join(item[1] for item in (result or []))
                    ocr_pages += 1

                f.write(f"\n\n===== 第 {i + 1} 页 =====\n{text.strip()}\n")
                f.flush()                       # 逐页落盘：中途中断也不丢已识别的内容
                total_chars += len(text.strip())
                print(f"  第 {i + 1} 页 OK（累计 {total_chars} 字，用时 {time.time() - t0:.0f}s）")

    print(f"\n完成：{out}（{total_chars} 字，OCR {ocr_pages} 页，总耗时 {time.time() - t0:.0f}s）")
    print("下一步：把生成的 txt 拖到网页「上传医疗资料」即可（txt 走秒级文本入库，不再 OCR）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
