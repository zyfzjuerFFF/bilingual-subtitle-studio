from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

DEFAULT_ENV_PATH = Path(".env")
ENV_EXPORT_KEYS = (
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_REGION",
    "OSS_ACCESS_KEY_ID",
    "OSS_ACCESS_KEY_SECRET",
    "OSS_BUCKET",
    "OSS_ENDPOINT",
    "OSS_SUBTITLE_PREFIX",
    "BAILIAN_ASR_MODEL",
    "BAILIAN_TRANSLATION_MODEL",
    "BAILIAN_ASR_LANGUAGE",
    "SUBTITLE_EMBED_MODE",
)


REGION_ENDPOINTS = {
    "cn": {
        "asr_submit": "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription",
        "asr_query_base": "https://dashscope.aliyuncs.com/api/v1/tasks/",
        "chat": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    },
    "intl": {
        "asr_submit": "https://dashscope-intl.aliyuncs.com/api/v1/services/audio/asr/transcription",
        "asr_query_base": "https://dashscope-intl.aliyuncs.com/api/v1/tasks/",
        "chat": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
    },
}

STRONG_PUNCTUATION = set("。！？!?；;")
WEAK_PUNCTUATION = set("，,、：:")


class CliError(RuntimeError):
    """User-facing pipeline failure."""


@dataclass(slots=True)
class RuntimeConfig:
    dashscope_api_key: str
    region: str
    asr_model: str
    translation_model: str
    asr_language: str | None
    poll_seconds: float
    max_wait_seconds: int
    translation_batch_size: int
    max_chars: int
    max_duration_ms: int
    signed_url_expires: int
    oss_bucket: str
    oss_endpoint: str
    oss_access_key_id: str
    oss_access_key_secret: str
    oss_prefix: str
    keep_temp_audio: bool
    keep_remote_audio: bool
    embed_mode: str


@dataclass(slots=True)
class Segment:
    start_ms: int
    end_ms: int
    source_text: str
    source_language: str


@dataclass(slots=True)
class BilingualSegment:
    index: int
    start_ms: int
    end_ms: int
    zh_text: str
    en_text: str
    source_text: str
    source_language: str


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        input_path = args.input.expanduser().resolve()
        if not input_path.exists():
            raise CliError(f"输入文件不存在: {input_path}")

        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = args.temp_dir.expanduser().resolve()
        temp_dir.mkdir(parents=True, exist_ok=True)

        config = load_config(args)
        outputs = run_pipeline(input_path=input_path, output_dir=output_dir, temp_dir=temp_dir, config=config)
    except CliError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print("[done] 已生成文件:")
    for key, path in outputs.items():
        print(f"  - {key}: {path}")
    return 0


def load_dotenv(path: Path = DEFAULT_ENV_PATH, *, override: bool = False) -> dict[str, str]:
    if not path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = parse_env_value(value.strip())
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def parse_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def write_dotenv(values: dict[str, str], path: Path = DEFAULT_ENV_PATH) -> None:
    lines = [f"{key}={shell_quote_env(values[key])}" for key in ENV_EXPORT_KEYS if values.get(key)]
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def shell_quote_env(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:-]+", value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


@contextlib.contextmanager
def temporary_env(updates: dict[str, str | None]):
    original: dict[str, str | None] = {}
    try:
        for key, value in updates.items():
            original[key] = os.environ.get(key)
            if value in (None, ""):
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bilingual-subtitles",
        description="输入本地视频/音频，输出中英双语字幕。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="本地视频或音频文件路径")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="输出目录",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("tmp"),
        help="临时目录",
    )
    parser.add_argument(
        "--region",
        choices=sorted(REGION_ENDPOINTS),
        default=os.getenv("DASHSCOPE_REGION", "cn"),
        help="百炼部署地域，默认读取 DASHSCOPE_REGION",
    )
    parser.add_argument(
        "--asr-model",
        default=os.getenv("BAILIAN_ASR_MODEL", "qwen3-asr-flash-filetrans"),
        help="语音识别模型",
    )
    parser.add_argument(
        "--translation-model",
        default=os.getenv("BAILIAN_TRANSLATION_MODEL", "qwen-plus"),
        help="翻译模型",
    )
    parser.add_argument(
        "--asr-language",
        default=os.getenv("BAILIAN_ASR_LANGUAGE") or None,
        help="可选，显式指定识别语种，如 zh / en / yue；默认自动识别",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=float(os.getenv("BAILIAN_POLL_SECONDS", "3")),
        help="轮询 ASR 任务状态的间隔秒数",
    )
    parser.add_argument(
        "--max-wait-seconds",
        type=int,
        default=int(os.getenv("BAILIAN_MAX_WAIT_SECONDS", "7200")),
        help="等待 ASR 完成的最长秒数",
    )
    parser.add_argument(
        "--translation-batch-size",
        type=int,
        default=int(os.getenv("BAILIAN_TRANSLATION_BATCH_SIZE", "25")),
        help="每批翻译的字幕条数",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=int(os.getenv("SUBTITLE_MAX_CHARS", "28")),
        help="单条字幕建议最大字符数",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=float(os.getenv("SUBTITLE_MAX_DURATION", "5.5")),
        help="单条字幕建议最大时长（秒）",
    )
    parser.add_argument(
        "--signed-url-expires",
        type=int,
        default=int(os.getenv("OSS_SIGNED_URL_EXPIRES", str(24 * 60 * 60))),
        help="上传到 OSS 后生成签名 URL 的有效期（秒）",
    )
    parser.add_argument(
        "--keep-temp-audio",
        action="store_true",
        help="保留抽取出来的临时音频文件",
    )
    parser.add_argument(
        "--keep-remote-audio",
        action="store_true",
        help="保留上传到 OSS 的临时音频对象",
    )
    parser.add_argument(
        "--embed-mode",
        choices=["none", "soft", "hard", "both"],
        default=os.getenv("SUBTITLE_EMBED_MODE", "none"),
        help="是否额外输出带字幕视频：soft=可切换字幕，hard=烧录字幕，both=两种都生成",
    )
    return parser


def load_config(args: argparse.Namespace) -> RuntimeConfig:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise CliError("缺少环境变量 DASHSCOPE_API_KEY")

    if args.region == "us":
        raise CliError("美国地域当前没有 qwen3-asr-flash-filetrans，可改用 cn 或 intl")

    missing = [
        name
        for name in (
            "OSS_ACCESS_KEY_ID",
            "OSS_ACCESS_KEY_SECRET",
            "OSS_BUCKET",
            "OSS_ENDPOINT",
        )
        if not os.getenv(name)
    ]
    if missing:
        raise CliError(f"缺少 OSS 配置环境变量: {', '.join(missing)}")

    return RuntimeConfig(
        dashscope_api_key=api_key,
        region=args.region,
        asr_model=args.asr_model,
        translation_model=args.translation_model,
        asr_language=args.asr_language,
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
        translation_batch_size=args.translation_batch_size,
        max_chars=args.max_chars,
        max_duration_ms=int(args.max_duration * 1000),
        signed_url_expires=args.signed_url_expires,
        oss_bucket=os.environ["OSS_BUCKET"],
        oss_endpoint=os.environ["OSS_ENDPOINT"],
        oss_access_key_id=os.environ["OSS_ACCESS_KEY_ID"],
        oss_access_key_secret=os.environ["OSS_ACCESS_KEY_SECRET"],
        oss_prefix=os.getenv("OSS_SUBTITLE_PREFIX", "subtitle-tool"),
        keep_temp_audio=args.keep_temp_audio,
        keep_remote_audio=args.keep_remote_audio,
        embed_mode=args.embed_mode,
    )


def run_pipeline(
    *,
    input_path: Path,
    output_dir: Path,
    temp_dir: Path,
    config: RuntimeConfig,
) -> dict[str, Path]:
    base_name = input_path.stem
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    temp_audio_path = temp_dir / f"{base_name}.{run_id}.mp3"
    remote_object_key = f"{config.oss_prefix.rstrip('/')}/{run_id}/{base_name}.mp3"

    print("[1/6] 抽取音频...")
    extract_audio(input_path=input_path, output_path=temp_audio_path)

    signed_url = ""
    try:
        print("[2/6] 上传 OSS 并生成签名 URL...")
        signed_url = upload_to_oss(
            local_path=temp_audio_path,
            object_key=remote_object_key,
            config=config,
        )

        print("[3/6] 提交百炼长音频识别任务...")
        task_id = submit_transcription(file_url=signed_url, config=config)

        print("[4/6] 等待识别完成...")
        transcription_url = wait_for_transcription(task_id=task_id, config=config)
        asr_result = http_json_get(transcription_url)

        print("[5/6] 生成字幕片段并翻译...")
        source_segments = build_segments(
            asr_result=asr_result,
            max_chars=config.max_chars,
            max_duration_ms=config.max_duration_ms,
        )
        bilingual_segments = translate_segments(source_segments, config=config)

        print("[6/6] 导出字幕文件...")
        srt_path = output_dir / f"{base_name}.zh-en.srt"
        vtt_path = output_dir / f"{base_name}.zh-en.vtt"
        raw_json_path = output_dir / f"{base_name}.asr.json"
        meta_json_path = output_dir / f"{base_name}.subtitles.json"

        write_srt(srt_path, bilingual_segments)
        write_vtt(vtt_path, bilingual_segments)
        raw_json_path.write_text(
            json.dumps(asr_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        meta_json_path.write_text(
            json.dumps([asdict(item) for item in bilingual_segments], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        output_paths: dict[str, Path] = {
            "srt": srt_path,
            "vtt": vtt_path,
            "asr_json": raw_json_path,
            "subtitle_json": meta_json_path,
        }

        if config.embed_mode != "none":
            ensure_video_stream(input_path)
            if config.embed_mode in {"soft", "both"}:
                soft_path = output_dir / f"{base_name}.zh-en.softsub.mp4"
                mux_subtitles_into_video(input_path=input_path, subtitle_path=srt_path, output_path=soft_path)
                output_paths["softsub_video"] = soft_path
            if config.embed_mode in {"hard", "both"}:
                hard_path = output_dir / f"{base_name}.zh-en.hardsub.mp4"
                burn_subtitles_into_video(input_path=input_path, subtitle_path=srt_path, output_path=hard_path)
                output_paths["hardsub_video"] = hard_path

        return output_paths
    finally:
        if temp_audio_path.exists() and not config.keep_temp_audio:
            temp_audio_path.unlink()
        if signed_url and not config.keep_remote_audio:
            try:
                delete_from_oss(remote_object_key, config)
            except Exception as exc:  # pragma: no cover - cleanup best effort
                print(f"[warn] 删除 OSS 临时文件失败: {exc}", file=sys.stderr)


def extract_audio(*, input_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "64k",
        "-loglevel",
        "error",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "ffmpeg 执行失败"
        raise CliError(f"音频抽取失败: {detail}")


def ensure_video_stream(input_path: Path) -> None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(input_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or "video" not in completed.stdout:
        raise CliError("当前输入文件没有视频流，无法把字幕直接加载进视频里")


def mux_subtitles_into_video(*, input_path: Path, subtitle_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-i",
        str(subtitle_path),
        "-map",
        "0:v",
        "-map",
        "0:a?",
        "-map",
        "1:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-c:s",
        "mov_text",
        "-metadata:s:s:0",
        "language=zho",
        "-disposition:s:0",
        "default",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "ffmpeg 软字幕封装失败"
        raise CliError(f"生成软字幕视频失败: {detail}")


def burn_subtitles_into_video(*, input_path: Path, subtitle_path: Path, output_path: Path) -> None:
    ensure_hardsub_support()
    subtitle_filter = f"subtitles=filename='{escape_ffmpeg_subtitle_path(subtitle_path)}'"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:v",
        "-map",
        "0:a?",
        "-vf",
        subtitle_filter,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "ffmpeg 硬字幕烧录失败"
        raise CliError(f"生成硬字幕视频失败: {detail}")


def ensure_hardsub_support() -> None:
    command = ["ffmpeg", "-hide_banner", "-filters"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    output = f"{completed.stdout}\n{completed.stderr}"
    if " subtitles " not in output:
        raise CliError(
            "当前 ffmpeg 未启用 subtitles/libass 过滤器，无法生成硬字幕视频。"
            "可以先使用 soft 模式，或安装带 libass 的 ffmpeg 后再试。"
        )


def escape_ffmpeg_subtitle_path(path: Path) -> str:
    escaped = str(path.resolve())
    escaped = escaped.replace("\\", "\\\\")
    escaped = escaped.replace(":", r"\:")
    escaped = escaped.replace("'", r"\'")
    escaped = escaped.replace(",", r"\,")
    escaped = escaped.replace("[", r"\[")
    escaped = escaped.replace("]", r"\]")
    return escaped


def upload_to_oss(*, local_path: Path, object_key: str, config: RuntimeConfig) -> str:
    try:
        import oss2
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise CliError("缺少依赖 oss2，请先执行 `python3 -m pip install -e .`") from exc

    auth = oss2.Auth(config.oss_access_key_id, config.oss_access_key_secret)
    bucket = oss2.Bucket(auth, config.oss_endpoint, config.oss_bucket)
    bucket.put_object_from_file(object_key, str(local_path))
    signed_url = bucket.sign_url("GET", object_key, config.signed_url_expires, slash_safe=True)
    return signed_url


def delete_from_oss(object_key: str, config: RuntimeConfig) -> None:
    import oss2

    auth = oss2.Auth(config.oss_access_key_id, config.oss_access_key_secret)
    bucket = oss2.Bucket(auth, config.oss_endpoint, config.oss_bucket)
    bucket.delete_object(object_key)


def submit_transcription(*, file_url: str, config: RuntimeConfig) -> str:
    endpoint = REGION_ENDPOINTS[config.region]["asr_submit"]
    payload: dict[str, Any] = {
        "model": config.asr_model,
        "input": {
            "file_url": file_url,
        },
        "parameters": {
            "channel_id": [0],
            "enable_itn": True,
            "enable_words": True,
        },
    }
    if config.asr_language:
        payload["parameters"]["language"] = config.asr_language

    response = http_json_request(
        url=endpoint,
        method="POST",
        headers={
            "Authorization": f"Bearer {config.dashscope_api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        },
        payload=payload,
    )
    task_id = response.get("output", {}).get("task_id")
    if not task_id:
        raise CliError(f"提交 ASR 任务失败，返回中未找到 task_id: {response}")
    return str(task_id)


def wait_for_transcription(*, task_id: str, config: RuntimeConfig) -> str:
    endpoint = REGION_ENDPOINTS[config.region]["asr_query_base"] + task_id
    deadline = time.time() + config.max_wait_seconds

    while time.time() < deadline:
        response = http_json_request(
            url=endpoint,
            method="GET",
            headers={
                "Authorization": f"Bearer {config.dashscope_api_key}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable",
            },
        )
        output = response.get("output", {})
        status = str(output.get("task_status", "")).upper()
        print(f"  · 当前任务状态: {status or 'UNKNOWN'}")

        if status == "SUCCEEDED":
            transcription_url = output.get("result", {}).get("transcription_url")
            if not transcription_url:
                raise CliError(f"任务成功但未返回 transcription_url: {response}")
            return str(transcription_url)
        if status in {"FAILED", "UNKNOWN"}:
            code = output.get("code", "UNKNOWN")
            message = output.get("message", "未知错误")
            raise CliError(f"ASR 任务失败: {code} {message}")

        time.sleep(config.poll_seconds)

    raise CliError(f"等待 ASR 任务超时，超过 {config.max_wait_seconds} 秒")


def build_segments(*, asr_result: dict[str, Any], max_chars: int, max_duration_ms: int) -> list[Segment]:
    segments: list[Segment] = []
    transcripts = asr_result.get("transcripts", [])
    if not transcripts:
        raise CliError("ASR 结果中没有 transcripts 字段")

    for transcript in transcripts:
        default_language = str(transcript.get("language") or "auto")
        for sentence in transcript.get("sentences", []):
            sentence_segments = split_sentence(sentence, default_language, max_chars, max_duration_ms)
            segments.extend(sentence_segments)

    segments = [segment for segment in segments if segment.source_text]
    if not segments:
        raise CliError("未能从 ASR 结果中生成有效字幕片段")
    return merge_short_segments(segments, max_chars=max_chars, max_duration_ms=max_duration_ms)


def split_sentence(
    sentence: dict[str, Any],
    default_language: str,
    max_chars: int,
    max_duration_ms: int,
) -> list[Segment]:
    language = str(sentence.get("language") or default_language or "auto")
    words = sentence.get("words") or []
    if not words:
        text = normalize_text(str(sentence.get("text", "")))
        if not text:
            return []
        return [
            Segment(
                start_ms=int(sentence.get("begin_time", 0)),
                end_ms=int(sentence.get("end_time", sentence.get("begin_time", 0))),
                source_text=text,
                source_language=language,
            )
        ]

    chunks: list[Segment] = []
    current_text = ""
    chunk_start: int | None = None
    chunk_end: int | None = None

    for word in words:
        token = f"{word.get('text', '')}{word.get('punctuation', '')}"
        token = normalize_token(token)
        if not token:
            continue

        word_start = int(word.get("begin_time", sentence.get("begin_time", 0)))
        word_end = int(word.get("end_time", word_start))
        if chunk_start is None:
            chunk_start = word_start

        current_text = join_text(current_text, token)
        chunk_end = word_end

        text = normalize_text(current_text)
        duration = chunk_end - chunk_start
        last_char = token[-1]
        should_break = False

        if last_char in STRONG_PUNCTUATION:
            should_break = True
        elif duration >= max_duration_ms:
            should_break = True
        elif len(text) >= max_chars and last_char in WEAK_PUNCTUATION:
            should_break = True
        elif len(text) >= max_chars * 2:
            should_break = True

        if should_break and text:
            chunks.append(
                Segment(
                    start_ms=chunk_start,
                    end_ms=chunk_end,
                    source_text=text,
                    source_language=language,
                )
            )
            current_text = ""
            chunk_start = None
            chunk_end = None

    remaining = normalize_text(current_text)
    if remaining and chunk_start is not None and chunk_end is not None:
        chunks.append(
            Segment(
                start_ms=chunk_start,
                end_ms=chunk_end,
                source_text=remaining,
                source_language=language,
            )
        )
    return chunks


def merge_short_segments(
    segments: list[Segment],
    *,
    max_chars: int,
    max_duration_ms: int,
) -> list[Segment]:
    merged: list[Segment] = []
    for segment in segments:
        if not merged:
            merged.append(segment)
            continue

        previous = merged[-1]
        combined_text = normalize_text(join_text(previous.source_text, segment.source_text))
        combined_duration = segment.end_ms - previous.start_ms
        can_merge = (
            previous.source_language == segment.source_language
            and len(previous.source_text) <= max_chars // 2
            and len(segment.source_text) <= max_chars // 2
            and combined_duration <= max_duration_ms
            and len(combined_text) <= max_chars + 6
        )

        if can_merge:
            merged[-1] = Segment(
                start_ms=previous.start_ms,
                end_ms=segment.end_ms,
                source_text=combined_text,
                source_language=previous.source_language,
            )
        else:
            merged.append(segment)

    return merged


def translate_segments(segments: list[Segment], *, config: RuntimeConfig) -> list[BilingualSegment]:
    translated: list[BilingualSegment] = []
    for batch_start in range(0, len(segments), config.translation_batch_size):
        batch = segments[batch_start : batch_start + config.translation_batch_size]
        try:
            items = translate_batch(batch, config=config)
        except Exception as exc:
            print(f"[warn] 批量翻译失败，回退到单条翻译: {exc}", file=sys.stderr)
            items = translate_batch_fallback(batch, config=config)
        else:
            missing_indexes = find_incomplete_indexes(items)
            if missing_indexes:
                human_indexes = ", ".join(str(batch_start + idx + 1) for idx in missing_indexes)
                print(f"[warn] 批量翻译存在缺项，改为逐条补译: {human_indexes}", file=sys.stderr)
                repaired_items = translate_batch_fallback([batch[idx] for idx in missing_indexes], config=config)
                for missing_index, repaired_item in zip(missing_indexes, repaired_items):
                    items[missing_index] = repaired_item

        for offset, item in enumerate(items):
            item.index = batch_start + offset + 1

        translated.extend(items)

    return translated


def translate_batch(batch: list[Segment], *, config: RuntimeConfig) -> list[BilingualSegment]:
    raw_items = request_batch_translation(batch, config=config)
    if len(raw_items) != len(batch):
        raise CliError(f"翻译结果条数不匹配: 期望 {len(batch)}，实际 {len(raw_items)}")

    result: list[BilingualSegment] = []
    for source, translated in zip(batch, raw_items):
        result.append(
            BilingualSegment(
                index=0,
                start_ms=source.start_ms,
                end_ms=source.end_ms,
                zh_text=normalize_text(translated.get("zh", "")),
                en_text=normalize_text(translated.get("en", "")),
                source_text=source.source_text,
                source_language=source.source_language,
            )
        )
    return result


def request_batch_translation(batch: list[Segment], *, config: RuntimeConfig) -> list[dict[str, str]]:
    endpoint = REGION_ENDPOINTS[config.region]["chat"]
    payload_items = [
        {
            "id": idx,
            "language": item.source_language,
            "text": item.source_text,
        }
        for idx, item in enumerate(batch)
    ]
    payload = {
        "model": config.translation_model,
        "temperature": 0.1,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是字幕翻译助手。"
                    "把每条输入都转换为中英双语字幕。"
                    "输出必须是严格 JSON，对象格式为 "
                    '{"items":[{"id":0,"zh":"...","en":"..."}]}。'
                    "不要输出 Markdown，不要省略任何 id，不要新增任何 id。"
                    "如果原文是中文，zh 使用自然、简洁的原文整理版，en 为英文翻译。"
                    "如果原文是英文，en 使用自然、简洁的原文整理版，zh 为中文翻译。"
                    "如果原文是其他语言，请同时给出自然中文和自然英文。"
                    "保留专有名词，不要添加解释，不要合并条目。"
                    "每个 item 的 zh 和 en 都必须是非空字符串。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"items": payload_items}, ensure_ascii=False),
            },
        ],
    }

    response = http_json_request(
        url=endpoint,
        method="POST",
        headers={
            "Authorization": f"Bearer {config.dashscope_api_key}",
            "Content-Type": "application/json",
        },
        payload=payload,
    )
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        raise CliError(f"翻译接口返回空内容: {response}")

    parsed = parse_json_blob(content)
    items = coerce_batch_translation_items(parsed, expected_size=len(batch))
    if items is None:
        raise CliError(f"无法对齐批量翻译结果: {content}")
    return items


def translate_batch_fallback(batch: list[Segment], *, config: RuntimeConfig) -> list[BilingualSegment]:
    return [translate_single_segment(segment, config=config) for segment in batch]


def translate_single_segment(segment: Segment, *, config: RuntimeConfig) -> BilingualSegment:
    endpoint = REGION_ENDPOINTS[config.region]["chat"]
    payload = {
        "model": config.translation_model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是字幕翻译助手。"
                    '输出必须是严格 JSON，格式为 {"zh":"...","en":"..."}。'
                    "不要输出 Markdown。"
                    "zh 和 en 都必须是非空字符串。"
                    "如果原文是中文，zh 使用自然、简洁的原文整理版，en 为英文翻译。"
                    "如果原文是英文，en 使用自然、简洁的原文整理版，zh 为中文翻译。"
                    "如果原文是其他语言，请同时给出自然中文和自然英文。"
                    "保留专有名词，不要添加解释。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "language": segment.source_language,
                        "text": segment.source_text,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }

    response = http_json_request(
        url=endpoint,
        method="POST",
        headers={
            "Authorization": f"Bearer {config.dashscope_api_key}",
            "Content-Type": "application/json",
        },
        payload=payload,
    )
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        raise CliError(f"单条翻译接口返回空内容: {response}")

    parsed = parse_json_blob(content)
    zh_text = normalize_text(str(parsed.get("zh", "")))
    en_text = normalize_text(str(parsed.get("en", "")))
    if not zh_text or not en_text:
        raise CliError(f"单条翻译结果缺少 zh/en 字段: {content}")

    return BilingualSegment(
        index=0,
        start_ms=segment.start_ms,
        end_ms=segment.end_ms,
        zh_text=zh_text,
        en_text=en_text,
        source_text=segment.source_text,
        source_language=segment.source_language,
    )


def coerce_batch_translation_items(parsed: dict[str, Any], *, expected_size: int) -> list[dict[str, str]] | None:
    raw_items = parsed.get("items")
    if not isinstance(raw_items, list):
        return None

    aligned: list[dict[str, str] | None] = [None] * expected_size
    for fallback_index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue

        raw_id = item.get("id", fallback_index)
        try:
            item_index = int(raw_id)
        except (TypeError, ValueError):
            continue
        if not 0 <= item_index < expected_size:
            continue

        aligned[item_index] = {
            "zh": str(item.get("zh", "")),
            "en": str(item.get("en", "")),
        }

    return [
        item if item is not None else {"zh": "", "en": ""}
        for item in aligned
    ]


def find_incomplete_indexes(items: list[BilingualSegment]) -> list[int]:
    incomplete: list[int] = []
    for index, item in enumerate(items):
        if not item.zh_text or not item.en_text:
            incomplete.append(index)
    return incomplete


def parse_json_blob(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise CliError(f"无法解析翻译 JSON: {content}")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise CliError(f"无法解析翻译 JSON: {content}") from exc


def http_json_get(url: str) -> dict[str, Any]:
    return http_json_request(url=url, method="GET", headers={})


def http_json_request(
    *,
    url: str,
    method: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = dict(headers)
    req = request.Request(url=url, data=data, headers=request_headers, method=method)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with request.urlopen(req, timeout=180) as response:
                body = response.read().decode("utf-8")
                return json.loads(body)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(attempt * 2)
                last_error = exc
                continue
            raise CliError(f"HTTP 请求失败 ({exc.code}) {url}: {detail}") from exc
        except error.URLError as exc:
            if attempt < retries:
                time.sleep(attempt * 2)
                last_error = exc
                continue
            raise CliError(f"网络请求失败 {url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise CliError(f"接口返回的不是合法 JSON: {url}") from exc

    raise CliError(f"请求失败: {url}: {last_error}")


def normalize_token(token: str) -> str:
    token = token.replace("\u3000", " ").replace("\n", " ")
    token = re.sub(r"\s+", " ", token)
    return token.strip()


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+([，。；：！？、])", r"\1", text)
    text = re.sub(r"([（《“])\s+", r"\1", text)
    text = re.sub(r"\s+([）》”])", r"\1", text)
    return text


def join_text(left: str, right: str) -> str:
    left = left.strip()
    right = right.strip()
    if not left:
        return right
    if not right:
        return left
    if should_add_space(left[-1], right[0]):
        return f"{left} {right}"
    return f"{left}{right}"


def should_add_space(left_char: str, right_char: str) -> bool:
    if left_char.isspace() or right_char.isspace():
        return False
    if right_char in ",.;:!?，。；：！？、)]}>'\"":
        return False
    if left_char in "([{<“‘《":
        return False

    left_is_word = is_word_like(left_char)
    right_is_word = is_word_like(right_char)
    left_is_cjk = is_cjk(left_char)
    right_is_cjk = is_cjk(right_char)

    if left_is_word and right_is_word:
        return True
    if (left_is_cjk and right_is_word) or (left_is_word and right_is_cjk):
        return True
    return False


def is_word_like(char: str) -> bool:
    return char.isascii() and (char.isalnum() or char in {"'", "%", "#", "@", "&", "/", "+"})


def is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def write_srt(path: Path, segments: list[BilingualSegment]) -> None:
    lines: list[str] = []
    for item in segments:
        lines.extend(
            [
                str(item.index),
                f"{format_srt_time(item.start_ms)} --> {format_srt_time(item.end_ms)}",
                item.zh_text,
                item.en_text,
                "",
            ]
        )
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_vtt(path: Path, segments: list[BilingualSegment]) -> None:
    lines = ["WEBVTT", ""]
    for item in segments:
        lines.extend(
            [
                f"{format_vtt_time(item.start_ms)} --> {format_vtt_time(item.end_ms)}",
                item.zh_text,
                item.en_text,
                "",
            ]
        )
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def format_srt_time(value_ms: int) -> str:
    hours, remainder = divmod(max(value_ms, 0), 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def format_vtt_time(value_ms: int) -> str:
    hours, remainder = divmod(max(value_ms, 0), 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"
