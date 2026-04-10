from collections import deque
from typing import Deque, Tuple, List

import numpy as np
import time


class VideoBuffer:
    """
    Ring buffer storing recent frames and timestamps for a campus camera.

    Stores approximately last N seconds of corridor/hostel footage so that
    incident clips can include moments before and after an alert.
    """

    def __init__(self, max_seconds: float, fps: float) -> None:
        self.max_seconds = max_seconds
        self.fps = fps
        self._buffer: Deque[Tuple[float, np.ndarray]] = deque()

    def add_frame(self, frame: np.ndarray) -> None:
        now = time.time()
        self._buffer.append((now, frame))
        self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - self.max_seconds
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    def get_frames_around(
        self,
        event_time: float,
        pre_sec: float,
        post_sec: float,
    ) -> List[np.ndarray]:
        """
        Return frames from [event_time - pre_sec, event_time + post_sec].
        """
        start = event_time - pre_sec
        end = event_time + post_sec
        frames: List[np.ndarray] = [
            frame for ts, frame in self._buffer if start <= ts <= end
        ]
        return frames

