import logging

__all__ = ["Logger"]

DEFAULT_FORMAT = "%(levelname)s [%(asctime)s] %(name)s: %(message)s"


class Logger:
    def __init__(self, name):
        self._logger = logging.getLogger(name)

    def debug(self, msg, /, **kwargs):
        self._log("debug", msg, kwargs)

    def info(self, msg, /, **kwargs):
        self._log("info", msg, kwargs)

    def warning(self, msg, /, **kwargs):
        self._log("warning", msg, kwargs)

    def error(self, msg, /, **kwargs):
        self._log("error", msg, kwargs)

    def _log(self, level, msg, kwargs):
        msg = self._format_msg(msg, kwargs)
        return getattr(self._logger, level)(msg)

    @staticmethod
    def _format_msg(msg, kwargs):
        if len(kwargs) > 0:
            str_in_quotes = lambda v: f'"{v}"' if isinstance(v, str) else v
            extra = " ".join((f"{k}={str_in_quotes(v)}" for k, v in kwargs.items()))
            return f"{msg} {extra}"
        return msg
