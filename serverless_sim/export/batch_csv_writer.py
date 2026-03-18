from __future__ import annotations

import csv
import os


class BatchCSVWriter:
    """Buffered CSV writer with batch flush."""

    def __init__(self, file_path: str, header: list[str], buffer_size: int = 100):
        self.file_path = file_path
        self.header = header
        self.buffer_size = buffer_size
        self._buffer: list[list] = []
        self._file = None
        self._writer = None

    def open(self) -> None:
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self._file = open(self.file_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.header)

    def write_row(self, row: list) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if self._writer and self._buffer:
            self._writer.writerows(self._buffer)
            self._file.flush()
            self._buffer.clear()

    def close(self) -> None:
        self.flush()
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None
