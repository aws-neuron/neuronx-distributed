# Standard Library
import logging
import os
import sys
from functools import lru_cache, wraps
from typing import Any, Optional, Callable, TypeVar
from typing_extensions import ParamSpec

# Third Party
import torch

T = TypeVar("T")
P = ParamSpec("P")


@lru_cache
def get_log_level() -> int:
    """Get the log level as configured or the default"""
    log_level = os.environ.get("NXD_LOG_LEVEL", default="info").lower()
    if log_level == "off":
        return logging.FATAL + 1
    if log_level == "fatal":
        # fatal added so that log level can take same values for cpp and py
        # fatal in cpp exceptions kills the process
        # so use fatal for that only
        return logging.FATAL
    if log_level == "error":
        return logging.ERROR
    if log_level == "warning":
        return logging.WARNING
    if log_level == "info":
        return logging.INFO
    if log_level in ["debug", "trace"]:
        return logging.DEBUG
    raise ValueError(f"Allowed NXD_LOG_LEVELS are: info, trace, debug, warning, error, fatal, off. Got: {log_level}")


class PackagePathFilter(logging.Filter):
    def filter(self, record: Any) -> bool:
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


def get_logger(name: str = "neuronx_distributed", rank0_only: bool = True) -> logging.Logger:
    logger = logging.getLogger(name + ("[0]" if rank0_only else "[]"))
    if getattr(logger, "initialized", False):
        return logger  # already configured

    hide_time = os.getenv("NXD_LOG_HIDE_TIME", "false").lower()
    time = "" if hide_time in ["true", "1"] else "%(asctime)s.%(msecs)03d: "
    log_formatter = logging.Formatter(
        fmt=f"[{time}%(levelname).1s %(relativepath)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(log_formatter)
    stdout_handler.addFilter(PackagePathFilter())
    logger.handlers = [stdout_handler]  # overwrite

    level = get_log_level()
    if level <= logging.FATAL:
        logger.setLevel(level)
        if rank0_only:
            # this ensures all logging levels get marked with the rank zero decorator
            # otherwise logs would get multiplied for each GPU process in multi-GPU setup
            for level in (
                "debug",
                "info",
                "warning",
                "error",
                "exception",
                "fatal",
                "critical",
            ):
                setattr(logger, level, _rank0_only(getattr(logger, level)))
    else:
        logger.disabled = True
    logger.propagate = False
    logger.initialized = True
    return logger


def _rank0_only(fn: Callable[P, T], default: Optional[T] = None, **extra_kwargs: P.kwargs) -> Callable[P, Optional[T]]:
    """Wrap a logging.Logger function to call internal function only in rank zero.
    Function that can be used as a decorator to enable a function/method being called only on global rank 0.

    Arguments:
       fn: function to decorate
       default: value to return when the global rank is not 0
    Returns:
       Decorated function
    """

    @wraps(fn)
    def wrapped_fn(*args: P.args, **kwargs: P.kwargs) -> Optional[T]:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else int(os.environ.get("RANK", "0"))
        if rank == 0:
            # excluding the wrapper from calling stack for logger to use
            # see https://docs.python.org/3/library/logging.html#logging.Logger.findCaller
            kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
            return fn(*args, **kwargs)
        return default

    return wrapped_fn
