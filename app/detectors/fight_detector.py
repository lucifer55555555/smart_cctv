from collections import deque
from typing import Deque, List, Optional

import cv2
import numpy as np
import os
import torch
import torch.nn as nn

from config import CONFIG


class FightDetector:
    """
    CNN + LSTM Violence Detection.

    Buffers recent frames and predicts probability of a fight.
    """

    def __init__(self) -> None:
        self.seq_len = CONFIG["detection"]["sequence_length"]
        self.prob_threshold = CONFIG["detection"]["fight_prob_threshold"]
        self.device = CONFIG["models"]["device"]
        self.weights_path = CONFIG["models"]["violence_model_weights"]

        self.model: Optional[nn.Module] = None
        self._load_model()

        self.frame_buffer: Deque[np.ndarray] = deque(maxlen=self.seq_len)

    def _load_model(self) -> None:
        """
        Load TorchScript violence detection model.
        """
        if not os.path.exists(self.weights_path):
            print(f"[FightDetector] Model not found: {self.weights_path}")
            self.model = None
            return

        try:
            self.model = torch.jit.load(self.weights_path, map_location=self.device)
            self.model.eval()
            print(f"[FightDetector] Loaded violence model: {self.weights_path}")
        except Exception as e:
            print(f"[FightDetector] Failed to load model: {e}")
            self.model = None

    def update_frames(self, frame: np.ndarray) -> None:
        """
        Add new frame to buffer.
        """
        self.frame_buffer.append(frame)

    def _preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Convert frames into tensor (1, seq_len, 3, 120, 120)
        """
        processed = []

        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224))

            arr = resized.astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW

            processed.append(arr)

        seq = np.stack(processed, axis=0)  # (T, C, H, W)
        seq = np.expand_dims(seq, axis=0)  # (1, T, C, H, W)

        tensor = torch.from_numpy(seq).float()
        return tensor.contiguous().to(self.device)

    def predict_fight(self) -> float:
        """
        Returns fight probability (0 → no fight, 1 → fight)
        """

        if self.model is None:
            return 0.0

        if len(self.frame_buffer) < self.seq_len:
            return 0.0

        frames = list(self.frame_buffer)

        with torch.no_grad():
            x = self._preprocess_frames(frames)

            logits = self.model(x)
            prob = torch.sigmoid(logits)

            fight_prob = float(prob.item())

        return fight_prob