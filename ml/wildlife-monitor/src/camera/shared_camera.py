import threading
import cv2


class SharedCamera:
    """
    Single owner of the physical camera device. A background thread
    continuously reads frames and stores only the most recent one.
    Prevents two processes/threads from both opening cv2.VideoCapture
    on the same device — that's the contention problem the live-view
    feature ran into: the detection loop needs continuous camera
    access, and the on-demand live stream needs frames too, without
    either one fighting the other for the same device.
    """

    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {source}")

        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self._lock:
                self._latest_frame = frame

    def get_latest_frame(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.cap.release()