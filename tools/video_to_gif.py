# -*- coding: utf-8 -*-
"""
把屏幕录制（mp4 / mov / mkv）转成适合放 GitHub README 的 GIF。

用法：
    # 整段转换（默认 900px 宽、12fps）
    python tools/video_to_gif.py demo.mp4 -o docs/images/demo.gif

    # 只截中间 3~20 秒，并降帧降宽（文件更小）
    python tools/video_to_gif.py demo.mp4 -o docs/images/demo.gif --start 3 --end 20 --fps 10 --width 800

依赖 ffmpeg（二选一，脚本自动找）：
    1) 系统装了 ffmpeg：winget install Gyan.FFmpeg
    2) 或者 pip install imageio-ffmpeg（本项目 .venv 里已装，会直接用它的 ffmpeg）

原理：两遍法 —— 先生成最优调色板（palettegen），再用调色板量化（paletteuse）。
比直接用 -vf scale 出的 GIF 颜色更准、体积小一半左右。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("找不到 ffmpeg：请 winget install Gyan.FFmpeg，或 pip install imageio-ffmpeg")


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if proc.returncode != 0:
        print(proc.stderr[-1500:])
        sys.exit(f"ffmpeg 执行失败（退出码 {proc.returncode}）")


def main() -> int:
    ap = argparse.ArgumentParser(description="屏幕录制 → GitHub 演示 GIF")
    ap.add_argument("video", help="输入视频（mp4/mov/mkv…）")
    ap.add_argument("-o", "--out", required=True, help="输出 gif 路径")
    ap.add_argument("--start", type=float, default=0, help="从第几秒开始（默认 0）")
    ap.add_argument("--end", type=float, default=0, help="到第几秒结束（0=到结尾）")
    ap.add_argument("--fps", type=int, default=12, help="帧率，README 用 10~12 足够（默认 12）")
    ap.add_argument("--width", type=int, default=900, help="宽度像素（默认 900，高度按比例）")
    ap.add_argument("--max-mb", type=float, default=8.0, help="超过这个体积就自动缩宽度重试")
    args = ap.parse_args()

    src = Path(args.video)
    if not src.exists():
        sys.exit(f"文件不存在：{src}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = find_ffmpeg()
    trim: list[str] = ["-ss", str(args.start)]
    if args.end and args.end > args.start:
        trim += ["-t", str(round(args.end - args.start, 2))]   # 用 -t 指定时长，避免 -to 的绝对/相对歧义

    width = args.width
    for _ in range(3):
        # 高度用 -2：保证是偶数，GIF/H.264 都要求偶数尺寸
        scale = f"fps={args.fps},scale={width}:-2:flags=lanczos"
        # 调色板放在输出目录旁边（不用系统临时目录：某些受限环境删不掉）
        palette = out.with_name(out.stem + ".palette.png")
        try:
            # 第一遍：分析画面，生成最优调色板
            run([ffmpeg, "-y", *trim, "-i", str(src),
                 "-vf", f"{scale},palettegen=stats_mode=diff", str(palette)])
            # 第二遍：用调色板量化输出 GIF（loop 0 = 无限循环）
            run([ffmpeg, "-y", *trim, "-i", str(src), "-i", str(palette),
                 "-lavfi", f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                 "-loop", "0", str(out)])
        finally:
            palette.unlink(missing_ok=True)

        size_mb = out.stat().st_size / 1024 / 1024
        print(f"[OK] {out}  {size_mb:.2f} MB | 宽 {width}px | {args.fps}fps")
        if size_mb <= args.max_mb:
            print("把它放进 docs/images/，README 里用 ![](docs/images/demo.gif) 引用即可")
            return 0
        width = int(width * 0.8)
        print(f"[WARN] 超过 {args.max_mb} MB，缩到 {width}px 重试…")

    sys.exit("还是太大：请缩短 --start/--end 区间，或降低 --fps / --width")


if __name__ == "__main__":
    sys.exit(main())
