# 本地中英双语字幕工具

输入本地视频或音频文件，输出中英双语 `SRT` / `VTT` 字幕。

现在同时支持：

- 命令行模式
- 本地桌面界面（PySide6）

工作流：

1. 本地用 `ffmpeg` 抽取音频。
2. 上传到阿里云 `OSS`，生成临时签名 URL。
3. 调用阿里云百炼 `qwen3-asr-flash-filetrans` 做长音频识别，拿到句级/词级时间戳。
4. 调用百炼文本模型把每条字幕补成中英双语。
5. 导出 `.srt`、`.vtt` 和调试用 JSON。

## 目录结构

```text
subtitle_tool/
  __main__.py
  cli.py
pyproject.toml
README.md
```

## 环境要求

- Python 3.11+
- 本机已安装 `ffmpeg`
- 阿里云百炼 API Key
- 一个可写的阿里云 OSS Bucket

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

安装完成后可用两个命令：

```bash
bilingual-subtitles
bilingual-subtitles-gui
```

## 配置环境变量

可以直接参考项目里的 `.env.example`，CLI 和 GUI 都会自动读取当前目录下的 `.env`：

```bash
export DASHSCOPE_API_KEY="sk-xxx"

export OSS_ACCESS_KEY_ID="LTAI..."
export OSS_ACCESS_KEY_SECRET="your-secret"
export OSS_BUCKET="your-bucket-name"
export OSS_ENDPOINT="https://oss-cn-shanghai.aliyuncs.com"
```

可选项：

```bash
export DASHSCOPE_REGION="cn"                  # cn / intl
export BAILIAN_ASR_MODEL="qwen3-asr-flash-filetrans"
export BAILIAN_TRANSLATION_MODEL="qwen-plus"
export BAILIAN_ASR_LANGUAGE="zh"             # 留空则自动识别
export OSS_SUBTITLE_PREFIX="subtitle-tool"
export SUBTITLE_EMBED_MODE="none"            # none / soft / hard / both
```

## 使用方法

```bash
python3 -m subtitle_tool /path/to/video.mp4
```

或：

```bash
bilingual-subtitles /path/to/video.mp4 -o output
```

如果你希望直接产出“带字幕的视频”，可以加：

```bash
bilingual-subtitles /path/to/video.mp4 -o output --embed-mode soft
bilingual-subtitles /path/to/video.mp4 -o output --embed-mode hard
bilingual-subtitles /path/to/video.mp4 -o output --embed-mode both
```

说明：

1. `soft` 会生成可开关的内嵌字幕视频
2. `hard` 会生成已经烧录到画面里的字幕视频
3. `both` 会两种都生成
4. `hard` / `both` 依赖本机 `ffmpeg` 启用了 `subtitles/libass` 过滤器

桌面版：

```bash
bilingual-subtitles-gui
```

打开后可以直接：

1. 选择视频或音频文件
2. 填写百炼和 OSS 配置
3. 点击“开始生成字幕”
4. 在日志区查看进度和输出文件路径

常用参数：

```bash
--output-dir output
--temp-dir tmp
--region cn
--asr-language zh
--max-chars 28
--max-duration 5.5
--keep-temp-audio
--keep-remote-audio
```

## 输出文件

执行完成后，默认会在 `output/` 下生成：

- `视频名.zh-en.srt`
- `视频名.zh-en.vtt`
- `视频名.asr.json`
- `视频名.subtitles.json`
- `视频名.zh-en.softsub.mp4`（如果启用 `soft` / `both`）
- `视频名.zh-en.hardsub.mp4`（如果启用 `hard` / `both`）

## 说明

- 之所以引入 `OSS`，是因为阿里云百炼当前的 `qwen3-asr-flash-filetrans` 需要传入公网可访问的音频 URL，不支持直接上传本地文件。
- 工具默认会在任务完成后删除 OSS 上的临时音频对象；如果需要保留，可加 `--keep-remote-audio`。
- 如果视频很长，整体耗时主要取决于音频长度、百炼队列时间和翻译批次数。
