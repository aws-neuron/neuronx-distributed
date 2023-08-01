# Standard Library
import logging
import os
import sys

_logger_initialized = False
_log_level = None


def get_log_level():
    global _log_level
    if _log_level is None:
        default = "info"
        log_level = os.environ.get("NXD_LOG_LEVEL", default=default)
        log_level = log_level.lower()

        # allowed_levels = ["info", "trace", "debug", "warning", "error", "fatal", "off"]
        if log_level == "off":
            level = logging.FATAL + 1
        elif log_level == "fatal":
            # fatal added so that log level can take same values for cpp and py
            # fatal in cpp exceptions kills the process
            # so use fatal for that only
            level = logging.FATAL
        elif log_level == "error":
            level = logging.ERROR
        elif log_level == "warning":
            level = logging.WARNING
        elif log_level == "info":
            level = logging.INFO
        elif log_level in ["debug", "trace"]:
            level = logging.DEBUG
        else:
            level = logging.INFO
        _log_level = level
    return _log_level


class PackagePathFilter(logging.Filter):
    def filter(self, record):
        pathname = record.pathname
        record.relativepath = None
        abs_sys_paths = map(os.path.abspath, sys.path)
        for path in sorted(abs_sys_paths, key=len, reverse=True):  # longer paths first
            if not path.endswith(os.sep):
                path += os.sep
            if pathname.startswith(path):
                record.relativepath = os.path.relpath(pathname, path)
                break
        return True


def get_logger(name="neuronx_distributed"):
    global _logger_initialized
    if not _logger_initialized:
        level = get_log_level()
        logger = logging.getLogger(name)
        hide_time = os.getenv("NXD_LOG_HIDE_TIME", "False")

        fmt = "["
        if hide_time.lower() in ["true", "1"]:
            hide_time = True
        else:
            hide_time = False
            fmt += "%(asctime)s.%(msecs)03d: "
        logger.handlers = []
        log_formatter = logging.Formatter(
            fmt=fmt + "%(levelname).1s %(relativepath)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(log_formatter)
        stdout_handler.addFilter(PackagePathFilter())
        logger.addHandler(stdout_handler)

        if level:
            logger.setLevel(level)
        else:
            logger.disabled = True
        _logger_initialized = True
        logger.propagate = False
    return logging.getLogger(name)
