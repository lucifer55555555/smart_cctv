from typing import Generator, Tuple, Optional, Union

import cv2

from config import CONFIG


class CampusCameraStream:
    """
    Handles video capture from a college campus camera (corridor, hostel gate, etc.).

    Usage:
        stream = CampusCameraStream()
        for ok, frame in stream.frames():
            if not ok:
                break
            # process frame
    """

    def __init__(self, source: Optional[Union[int, str]] = None) -> None:
        cam_cfg = CONFIG["camera"]
        self.source: Union[int, str] = source if source is not None else cam_cfg["source"]
        self.width = cam_cfg["frame_width"]
        self.height = cam_cfg["frame_height"]
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            # Non-crashing: just warn; callers will see (ok, None) from frames().
            print(f"[CampusCameraStream] WARNING: Unable to open campus camera source={self.source}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def frames(self) -> Generator[Tuple[bool, Optional[object]], None, None]:
        """
        Yield (success, frame) continuously from the campus camera.
        """
        if self.cap is None:
            while True:
                yield False, None

        while True:
            success, frame = self.cap.read()
            if not success:
                yield False, None
                break
            yield True, frame

    def release(self) -> None:
        """Release the campus camera resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

