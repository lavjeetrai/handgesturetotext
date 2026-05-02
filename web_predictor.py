import base64
import binascii
import math
import os
import threading
import time
from collections import OrderedDict
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from final_pred import (
    Application,
    MODEL_PATH,
    WHITE_IMAGE_PATH,
    ddd,
    hd,
    hd2,
    offset,
)

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", 2 * 1024 * 1024))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", 2_000_000))
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class WebPredictionSession:
    def __init__(self, model):
        self.model = model
        self.lock = threading.Lock()
        self.last_seen = time.time()
        self.reset()

    def reset(self):
        self.ct = {"blank": 0}
        self.blank_flag = 0
        self.space_flag = False
        self.next_flag = True
        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = [" "] * 10
        self.str = " "
        self.ccc = 0
        self.word = " "
        self.current_symbol = " "
        self.photo = "Empty"
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "
        self.pts = []

    def distance(self, x, y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def _first_hand(self, detector, image):
        result = detector.findHands(image, draw=False, flipType=True)
        hands = result[0] if isinstance(result, tuple) else result
        return hands[0] if hands else None

    def _white_canvas(self):
        white = cv2.imread(str(WHITE_IMAGE_PATH))
        if white is None:
            return np.ones((400, 400, 3), np.uint8) * 255
        return cv2.resize(white, (400, 400))

    def _draw_skeleton(self, white, pts, x_offset, y_offset):
        return Application._draw_skeleton(self, white, pts, x_offset, y_offset)

    def _encode_image(self, image):
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")

    def apply_suggestion(self, index):
        words = [self.word1, self.word2, self.word3, self.word4]
        if index < 1 or index > len(words):
            return

        selected = words[index - 1].strip()
        if not selected:
            return

        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        if idx_word == -1:
            return
        self.str = self.str[:idx_word] + selected.upper()

    def snapshot(self, status="ok", skeleton=None):
        return {
            "status": status,
            "symbol": self.current_symbol,
            "sentence": self.str.strip(),
            "suggestions": [
                self.word1.strip(),
                self.word2.strip(),
                self.word3.strip(),
                self.word4.strip(),
            ],
            "skeleton": skeleton,
        }

    def predict_frame(self, frame):
        with self.lock:
            self.last_seen = time.time()
            frame = cv2.flip(frame, 1)
            hand = self._first_hand(hd, frame)
            if not hand:
                return self.snapshot(status="no_hand")

            x, y, w, h = hand["bbox"]
            height, width = frame.shape[:2]
            x1 = max(0, x - offset)
            y1 = max(0, y - offset)
            x2 = min(width, x + w + offset)
            y2 = min(height, y + h + offset)
            image = frame[y1:y2, x1:x2]
            if image.size == 0:
                return self.snapshot(status="empty_crop")

            handz = self._first_hand(hd2, image)
            if not handz:
                return self.snapshot(status="no_hand_crop")

            self.pts = handz["lmList"]
            x_offset = ((400 - w) // 2) - 15
            y_offset = ((400 - h) // 2) - 15
            skeleton = self._draw_skeleton(self._white_canvas(), self.pts, x_offset, y_offset)
            Application.predict(self, skeleton)
            return self.snapshot(skeleton=self._encode_image(skeleton))


class WebSignPredictor:
    def __init__(self):
        self.model = load_model(str(MODEL_PATH), compile=False)
        self.sessions = OrderedDict()
        self.sessions_lock = threading.Lock()
        self.max_sessions = int(os.getenv("MAX_SESSIONS", "64"))

    def session(self, session_id):
        with self.sessions_lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = WebPredictionSession(self.model)
            session = self.sessions[session_id]
            session.last_seen = time.time()
            self.sessions.move_to_end(session_id)
            while len(self.sessions) > self.max_sessions:
                self.sessions.popitem(last=False)
            return session

    def reset(self, session_id):
        session = self.session(session_id)
        with session.lock:
            session.reset()
            return session.snapshot(status="reset")

    def suggest(self, session_id, index):
        session = self.session(session_id)
        with session.lock:
            session.apply_suggestion(index)
            return session.snapshot(status="suggestion_applied")


def decode_data_url(data_url):
    if not isinstance(data_url, str):
        raise ValueError("Image must be a data URL")

    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(data_url, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid image data") from exc

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image payload is too large")

    try:
        image = Image.open(BytesIO(image_bytes))
        image.verify()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("Invalid image file") from exc

    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("Image dimensions are too large")

    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
