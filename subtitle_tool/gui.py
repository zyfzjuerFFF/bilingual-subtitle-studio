from __future__ import annotations

import argparse
import io
import os
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from .cli import CliError, load_config, load_dotenv, run_pipeline, temporary_env, write_dotenv


class SignalStream(io.TextIOBase):
    def __init__(self, callback):
        super().__init__()
        self._callback = callback

    def write(self, text: str) -> int:
        if text:
            self._callback(text)
        return len(text)

    def flush(self) -> None:
        return None


class SubtitleWorker(QThread):
    log = Signal(str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__()
        self.payload = payload

    def run(self) -> None:
        env_map = {key: str(value).strip() for key, value in self.payload["env"].items()}
        args = argparse.Namespace(
            region=self.payload["region"],
            asr_model=self.payload["asr_model"],
            translation_model=self.payload["translation_model"],
            asr_language=self.payload["asr_language"] or None,
            poll_seconds=float(self.payload["poll_seconds"]),
            max_wait_seconds=int(self.payload["max_wait_seconds"]),
            translation_batch_size=int(self.payload["translation_batch_size"]),
            max_chars=int(self.payload["max_chars"]),
            max_duration=float(self.payload["max_duration"]),
            signed_url_expires=int(self.payload["signed_url_expires"]),
            keep_temp_audio=bool(self.payload["keep_temp_audio"]),
            keep_remote_audio=bool(self.payload["keep_remote_audio"]),
            embed_mode=str(self.payload["embed_mode"]),
        )

        stdout_stream = SignalStream(self.log.emit)
        stderr_stream = SignalStream(self.log.emit)
        try:
            with temporary_env(env_map):
                config = load_config(args)
                from contextlib import redirect_stderr, redirect_stdout

                output_dir = Path(str(self.payload["output_dir"]))
                temp_dir = Path(str(self.payload["temp_dir"]))
                output_dir.mkdir(parents=True, exist_ok=True)
                temp_dir.mkdir(parents=True, exist_ok=True)

                with redirect_stdout(stdout_stream), redirect_stderr(stderr_stream):
                    outputs = run_pipeline(
                        input_path=Path(str(self.payload["input_path"])),
                        output_dir=output_dir,
                        temp_dir=temp_dir,
                        config=config,
                    )
        except CliError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # pragma: no cover - GUI runtime errors
            traceback_text = "".join(traceback.format_exception(exc))
            self.log.emit(traceback_text)
            self.failed.emit(str(exc))
        else:
            output_payload = {key: str(value) for key, value in outputs.items()}
            self.finished_ok.emit(output_payload)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("中英双语字幕工具")
        self.resize(980, 760)
        self.worker: SubtitleWorker | None = None

        load_dotenv()
        self.root = QWidget()
        self.setCentralWidget(self.root)
        self.root_layout = QVBoxLayout(self.root)
        self.root_layout.setContentsMargins(18, 18, 18, 18)
        self.root_layout.setSpacing(14)

        self.root_layout.addWidget(self.build_paths_group())
        self.root_layout.addWidget(self.build_model_group())
        self.root_layout.addWidget(self.build_cloud_group())
        self.root_layout.addWidget(self.build_options_group())
        self.root_layout.addLayout(self.build_actions())
        self.root_layout.addWidget(self.build_log_view(), stretch=1)

        self.load_env_into_fields()

    def build_paths_group(self) -> QGroupBox:
        group = QGroupBox("文件")
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(10)

        self.input_path = QLineEdit()
        self.output_dir = QLineEdit(str((Path.cwd() / "output").resolve()))
        self.temp_dir = QLineEdit(str((Path.cwd() / "tmp").resolve()))

        layout.addWidget(QLabel("输入视频 / 音频"), 0, 0)
        layout.addWidget(self.input_path, 0, 1)
        layout.addWidget(self.make_button("选择文件", self.pick_input_file), 0, 2)

        layout.addWidget(QLabel("输出目录"), 1, 0)
        layout.addWidget(self.output_dir, 1, 1)
        layout.addWidget(self.make_button("选择目录", self.pick_output_dir), 1, 2)

        layout.addWidget(QLabel("临时目录"), 2, 0)
        layout.addWidget(self.temp_dir, 2, 1)
        layout.addWidget(self.make_button("选择目录", self.pick_temp_dir), 2, 2)
        return group

    def build_model_group(self) -> QGroupBox:
        group = QGroupBox("模型")
        layout = QFormLayout(group)

        self.region = QComboBox()
        self.region.addItems(["cn", "intl"])
        self.asr_model = QLineEdit("qwen3-asr-flash-filetrans")
        self.translation_model = QLineEdit("qwen-plus")
        self.asr_language = QLineEdit()

        layout.addRow("地域", self.region)
        layout.addRow("识别模型", self.asr_model)
        layout.addRow("翻译模型", self.translation_model)
        layout.addRow("识别语种", self.asr_language)
        return group

    def build_cloud_group(self) -> QGroupBox:
        group = QGroupBox("阿里云配置")
        layout = QFormLayout(group)

        self.dashscope_api_key = QLineEdit()
        self.dashscope_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.oss_access_key_id = QLineEdit()
        self.oss_access_key_secret = QLineEdit()
        self.oss_access_key_secret.setEchoMode(QLineEdit.EchoMode.Password)
        self.oss_bucket = QLineEdit()
        self.oss_endpoint = QLineEdit()
        self.oss_prefix = QLineEdit("subtitle-tool")

        layout.addRow("DashScope API Key", self.dashscope_api_key)
        layout.addRow("OSS AccessKey ID", self.oss_access_key_id)
        layout.addRow("OSS AccessKey Secret", self.oss_access_key_secret)
        layout.addRow("OSS Bucket", self.oss_bucket)
        layout.addRow("OSS Endpoint", self.oss_endpoint)
        layout.addRow("OSS Prefix", self.oss_prefix)
        return group

    def build_options_group(self) -> QGroupBox:
        group = QGroupBox("字幕参数")
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(10)

        self.max_chars = QSpinBox()
        self.max_chars.setRange(8, 120)
        self.max_chars.setValue(28)

        self.max_duration = QDoubleSpinBox()
        self.max_duration.setRange(1.0, 20.0)
        self.max_duration.setSingleStep(0.5)
        self.max_duration.setValue(5.5)

        self.translation_batch_size = QSpinBox()
        self.translation_batch_size.setRange(1, 100)
        self.translation_batch_size.setValue(25)

        self.poll_seconds = QDoubleSpinBox()
        self.poll_seconds.setRange(1.0, 30.0)
        self.poll_seconds.setSingleStep(1.0)
        self.poll_seconds.setValue(3.0)

        self.max_wait_seconds = QSpinBox()
        self.max_wait_seconds.setRange(60, 72000)
        self.max_wait_seconds.setValue(7200)

        self.signed_url_expires = QSpinBox()
        self.signed_url_expires.setRange(300, 604800)
        self.signed_url_expires.setValue(86400)

        self.embed_mode = QComboBox()
        self.embed_mode.addItem("不生成带字幕视频", "none")
        self.embed_mode.addItem("生成软字幕视频", "soft")
        self.embed_mode.addItem("生成硬字幕视频", "hard")
        self.embed_mode.addItem("两种都生成", "both")

        layout.addWidget(QLabel("单条最大字符数"), 0, 0)
        layout.addWidget(self.max_chars, 0, 1)
        layout.addWidget(QLabel("单条最大时长(秒)"), 0, 2)
        layout.addWidget(self.max_duration, 0, 3)
        layout.addWidget(QLabel("翻译批大小"), 1, 0)
        layout.addWidget(self.translation_batch_size, 1, 1)
        layout.addWidget(QLabel("轮询间隔(秒)"), 1, 2)
        layout.addWidget(self.poll_seconds, 1, 3)
        layout.addWidget(QLabel("最长等待(秒)"), 2, 0)
        layout.addWidget(self.max_wait_seconds, 2, 1)
        layout.addWidget(QLabel("签名 URL 有效期"), 2, 2)
        layout.addWidget(self.signed_url_expires, 2, 3)
        layout.addWidget(QLabel("视频字幕"), 3, 0)
        layout.addWidget(self.embed_mode, 3, 1, 1, 3)
        return group

    def build_actions(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(10)

        self.save_env_button = self.make_button("保存到 .env", self.save_env_file)
        self.run_button = self.make_button("开始生成字幕", self.start_job, primary=True)
        self.clear_log_button = self.make_button("清空日志", self.clear_log)

        layout.addWidget(self.save_env_button)
        layout.addWidget(self.clear_log_button)
        layout.addStretch(1)
        layout.addWidget(self.run_button)
        return layout

    def build_log_view(self) -> QGroupBox:
        group = QGroupBox("运行日志")
        layout = QVBoxLayout(group)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.log_view)
        return group

    def make_button(self, text: str, callback, *, primary: bool = False) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(callback)
        if primary:
            button.setStyleSheet(
                "QPushButton { background: #1769e0; color: white; padding: 8px 16px; "
                "border-radius: 8px; font-weight: 600; }"
                "QPushButton:disabled { background: #9db9ea; }"
            )
        return button

    def load_env_into_fields(self) -> None:
        env = dict(os.environ)
        self.dashscope_api_key.setText(env.get("DASHSCOPE_API_KEY", ""))
        self.region.setCurrentText(env.get("DASHSCOPE_REGION", "cn") or "cn")
        self.oss_access_key_id.setText(env.get("OSS_ACCESS_KEY_ID", ""))
        self.oss_access_key_secret.setText(env.get("OSS_ACCESS_KEY_SECRET", ""))
        self.oss_bucket.setText(env.get("OSS_BUCKET", ""))
        self.oss_endpoint.setText(env.get("OSS_ENDPOINT", ""))
        self.oss_prefix.setText(env.get("OSS_SUBTITLE_PREFIX", "subtitle-tool"))
        self.asr_model.setText(env.get("BAILIAN_ASR_MODEL", "qwen3-asr-flash-filetrans"))
        self.translation_model.setText(env.get("BAILIAN_TRANSLATION_MODEL", "qwen-plus"))
        self.asr_language.setText(env.get("BAILIAN_ASR_LANGUAGE", ""))
        embed_mode = env.get("SUBTITLE_EMBED_MODE", "none")
        self.embed_mode.setCurrentIndex(max(self.embed_mode.findData(embed_mode), 0))

    def collect_env_values(self) -> dict[str, str]:
        return {
            "DASHSCOPE_API_KEY": self.dashscope_api_key.text().strip(),
            "DASHSCOPE_REGION": self.region.currentText().strip(),
            "OSS_ACCESS_KEY_ID": self.oss_access_key_id.text().strip(),
            "OSS_ACCESS_KEY_SECRET": self.oss_access_key_secret.text().strip(),
            "OSS_BUCKET": self.oss_bucket.text().strip(),
            "OSS_ENDPOINT": self.oss_endpoint.text().strip(),
            "OSS_SUBTITLE_PREFIX": self.oss_prefix.text().strip(),
            "BAILIAN_ASR_MODEL": self.asr_model.text().strip(),
            "BAILIAN_TRANSLATION_MODEL": self.translation_model.text().strip(),
            "BAILIAN_ASR_LANGUAGE": self.asr_language.text().strip(),
            "SUBTITLE_EMBED_MODE": self.embed_mode.currentData(),
        }

    def collect_payload(self) -> dict[str, object]:
        input_path = self.input_path.text().strip()
        output_dir = self.output_dir.text().strip()
        temp_dir = self.temp_dir.text().strip()
        if not input_path:
            raise CliError("请选择输入视频或音频文件")
        if not output_dir:
            raise CliError("请选择输出目录")
        if not temp_dir:
            raise CliError("请选择临时目录")
        if not Path(input_path).expanduser().exists():
            raise CliError("输入文件不存在，请重新选择")

        return {
            "input_path": str(Path(input_path).expanduser().resolve()),
            "output_dir": str(Path(output_dir).expanduser().resolve()),
            "temp_dir": str(Path(temp_dir).expanduser().resolve()),
            "region": self.region.currentText(),
            "asr_model": self.asr_model.text().strip(),
            "translation_model": self.translation_model.text().strip(),
            "asr_language": self.asr_language.text().strip(),
            "poll_seconds": self.poll_seconds.value(),
            "max_wait_seconds": self.max_wait_seconds.value(),
            "translation_batch_size": self.translation_batch_size.value(),
            "max_chars": self.max_chars.value(),
            "max_duration": self.max_duration.value(),
            "signed_url_expires": self.signed_url_expires.value(),
            "keep_temp_audio": False,
            "keep_remote_audio": False,
            "embed_mode": self.embed_mode.currentData(),
            "env": self.collect_env_values(),
        }

    def start_job(self) -> None:
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "任务进行中", "当前已有任务在运行，请等待完成。")
            return

        try:
            payload = self.collect_payload()
        except CliError as exc:
            QMessageBox.warning(self, "参数不完整", str(exc))
            return

        self.append_log("准备开始处理...\n")
        self.set_running(True)
        self.worker = SubtitleWorker(payload)
        self.worker.log.connect(self.append_log)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.finished.connect(lambda: self.set_running(False))
        self.worker.start()

    def on_finished(self, outputs: dict[str, str]) -> None:
        self.append_log("\n任务完成。\n")
        for key, value in outputs.items():
            self.append_log(f"{key}: {value}\n")
        QMessageBox.information(self, "完成", "字幕已生成完成。")

    def on_failed(self, message: str) -> None:
        self.append_log(f"\n[error] {message}\n")
        QMessageBox.critical(self, "任务失败", message)

    def set_running(self, running: bool) -> None:
        self.run_button.setDisabled(running)
        self.save_env_button.setDisabled(running)

    def append_log(self, text: str) -> None:
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        self.log_view.insertPlainText(text)
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)

    def clear_log(self) -> None:
        self.log_view.clear()

    def save_env_file(self) -> None:
        values = self.collect_env_values()
        try:
            write_dotenv(values)
        except Exception as exc:  # pragma: no cover - filesystem/UI error
            QMessageBox.critical(self, "保存失败", str(exc))
            return
        QMessageBox.information(self, "已保存", f"已写入 {Path('.env').resolve()}")

    def pick_input_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择输入视频或音频",
            str(Path.cwd()),
            "Media Files (*.mp4 *.mov *.mkv *.avi *.mp3 *.wav *.m4a *.aac *.flac);;All Files (*)",
        )
        if file_path:
            self.input_path.setText(file_path)

    def pick_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_dir.text() or str(Path.cwd()))
        if directory:
            self.output_dir.setText(directory)

    def pick_temp_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择临时目录", self.temp_dir.text() or str(Path.cwd()))
        if directory:
            self.temp_dir.setText(directory)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Bilingual Subtitle Tool")
    app.setStyleSheet(
        """
        QWidget {
            font-family: "PingFang SC", "Noto Sans CJK SC", sans-serif;
            font-size: 13px;
        }
        QGroupBox {
            font-weight: 600;
            border: 1px solid #d5ddeb;
            border-radius: 10px;
            margin-top: 10px;
            padding-top: 12px;
            background: #f8fbff;
        }
        QGroupBox::title {
            left: 12px;
            padding: 0 4px;
        }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
            border: 1px solid #c7d3ea;
            border-radius: 8px;
            padding: 6px 8px;
            background: white;
        }
        QPlainTextEdit {
            background: #0f1720;
            color: #d8e4ff;
            font-family: "SFMono-Regular", "Menlo", monospace;
        }
        QPushButton {
            padding: 8px 14px;
            border-radius: 8px;
            border: 1px solid #bfd0ef;
            background: white;
        }
        QPushButton:hover {
            background: #eef4ff;
        }
        """
    )
    window = MainWindow()
    window.show()
    return app.exec()
