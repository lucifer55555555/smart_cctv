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
    CNN + LSTM Violence Detection with confidence smoothing.

    Buffers recent frames, predicts probability of a fight,
    and smooths predictions over a rolling window to prevent flickering.
    """

    # Number of recent predictions to average for smoothing
    SMOOTHING_WINDOW = 5

    def __init__(self) -> None:
        self.seq_len = CONFIG["detection"]["sequence_length"]
        self.prob_threshold = CONFIG["detection"]["fight_prob_threshold"]
        self.device = CONFIG["models"]["device"]
        self.weights_path = CONFIG["models"]["violence_model_weights"]

        self.model: Optional[nn.Module] = None
        self._load_model()

        # Buffer stores preprocessed (3, 224, 224) tensors
        self.tensor_buffer: Deque[torch.Tensor] = deque(maxlen=self.seq_len)

        # Rolling window of recent probabilities for smoothing
        self._prob_history: Deque[float] = deque(maxlen=self.SMOOTHING_WINDOW)

        # Cache the last raw and smoothed probabilities for debugging/overlay
        self.last_raw_prob: float = 0.0
        self.last_smoothed_prob: float = 0.0

    def _preprocess_single_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Convert a single BGR frame into a preprocessed tensor.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        arr = resized.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(arr).float()

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
        Add new frame to buffer after preprocessing.
        """
        try:
            tensor = self._preprocess_single_frame(frame)
            self.tensor_buffer.append(tensor)
        except Exception as e:
            print(f"[FightDetector] Frame preprocessing error: {e}")

    def predict_fight(self) -> float:
        """
        Returns smoothed fight probability (0 → no fight, 1 → fight).

        Uses a rolling average over the last SMOOTHING_WINDOW predictions
        to reduce false-positive flickering.
        """
        if self.model is None:
            return 0.0

        if len(self.tensor_buffer) < self.seq_len:
            return 0.0

        try:
            # Efficiently stack already preprocessed tensors
            # Resulting shape: (1, T, C, H, W)
            x = torch.stack(list(self.tensor_buffer), dim=0).unsqueeze(0).to(self.device).contiguous()

            with torch.no_grad():
                logits = self.model(x)
                prob = torch.sigmoid(logits)
                raw_prob = float(prob.item())

        except Exception as e:
            print(f"[FightDetector] Inference error: {e}")
            raw_prob = 0.0

        self.last_raw_prob = raw_prob
        self._prob_history.append(raw_prob)

        # Smoothed probability = weighted moving average (more recent = higher weight)
        weights = np.arange(1, len(self._prob_history) + 1, dtype=np.float64)
        smoothed = float(np.average(list(self._prob_history), weights=weights))
        self.last_smoothed_prob = smoothed

        return smoothed

    def reset(self) -> None:
        """Clear all buffers and history — useful when switching cameras."""
        self.tensor_buffer.clear()
        self._prob_history.clear()
        self.last_raw_prob = 0.0
        self.last_smoothed_prob = 0.0