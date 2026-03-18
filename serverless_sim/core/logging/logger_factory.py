import logging
import os
import sys


def create_logger(
    module_name: str,
    run_dir: str,
    mode: str = "console",
    level: str = "INFO",
) -> logging.Logger:
    """Create a logger for a module.

    Parameters
    ----------
    module_name : str
        Name used as the logger identifier (e.g. ``"serverless_sim"``).
    run_dir : str
        Directory where the log file will be written (used when *mode*
        includes file output).
    mode : str
        One of ``"console"``, ``"file"``, or ``"both"``.
    level : str
        Logging level string (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, …).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if mode in ("console", "both"):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if mode in ("file", "both"):
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "simulation.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
