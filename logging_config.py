"""统一日志配置。"""

import logging
import logging.handlers
import re
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "app.log"
LOG_FORMATTER = logging.Formatter(
    "[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


class StripAnsiFilter(logging.Filter):
    """去掉日志中的终端颜色控制符，避免写入 app.log 后显示乱码。"""

    @staticmethod
    def _clean(value):
        if isinstance(value, str):
            return ANSI_ESCAPE_RE.sub("", value)
        return value

    def filter(self, record):
        record.msg = self._clean(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(self._clean(arg) for arg in record.args)
        elif isinstance(record.args, dict):
            record.args = {key: self._clean(value) for key, value in record.args.items()}
        return True


def setup_logging() -> logging.Logger:
    """配置根日志，返回项目根 logger。重复调用不会重复添加 handler。"""

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    if getattr(root_logger, "_multi_agent_logging_configured", False):
        return logging.getLogger("multi_agent")

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(LOG_FORMATTER)
    file_handler.addFilter(StripAnsiFilter())

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(LOG_FORMATTER)
    console_handler.addFilter(StripAnsiFilter())

    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger._multi_agent_logging_configured = True

    werkzeug_logger = logging.getLogger("werkzeug")
    werkzeug_logger.setLevel(logging.INFO)
    werkzeug_logger.handlers.clear()
    werkzeug_logger.propagate = True

    return logging.getLogger("multi_agent")
