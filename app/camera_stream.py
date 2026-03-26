import threading
import time
from typing import Generator, Tuple, Optional, Union

import cv2

from config import CONFIG


class CampusCameraStream:
    """
    Handles video capture from a college campus camera (corridor, hostel gate, etc.)
    using a background thread to maintain a real-time (lowest latency) buffer.
    """

    def __init__(self, source: Optional[Union[int, str]] = None) -> None:
        cam_cfg = CONFIG["camera"]
        self.source: Union[int, str] = source if source is not None else cam_cfg["source"]
        self.width = cam_cfg["frame_width"]
        self.height = cam_cfg["frame_height"]
        
        # Reverting to simplest initialization for maximum stability
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"[CampusCameraStream] WARNING: Unable to open campus camera source={self.source}")



        self.last_frame: Optional[object] = None
        self.success: bool = False
        self.running: bool = True
        self.lock = threading.Lock()
        
        # Start background thread to keep buffer empty
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self) -> None:
        """Continuously grab frames in a background thread."""
        while self.running:
            if self.cap is not None and self.cap.isOpened():
                success, frame = self.cap.read()
                with self.lock:
                    self.success = success
                    self.last_frame = frame
            else:
                time.sleep(0.1)

    def frames(self) -> Generator[Tuple[bool, Optional[object]], None, None]:
        """
        Yield (success, frame) as fast as possible (the latest available).
        """
        while self.running:
            with self.lock:
                yield self.success, self.last_frame
            # Small sleep to yield control to the read thread
            time.sleep(0.01)

    def release(self) -> None:
        """Release the campus camera resource."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None


