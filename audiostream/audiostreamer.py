import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QThread, pyqtSignal
from collections import deque

CHUNK_DURATION = 3  # seconds
SAMPLE_RATE = 16000
BUFFER_SIZE = SAMPLE_RATE * CHUNK_DURATION

class AudioStreamer(QThread):
    chunk_ready = pyqtSignal(np.ndarray)

    def __init__(self, device: int | None = None):
        """Stream microphone audio and emit fixed-size chunks.

        Parameters
        ----------
        device: int | None
            Optional device index to use for recording. If ``None`` the default
            input device is used.
        """
        super().__init__()
        self.running = False
        self.buffer = deque(maxlen=BUFFER_SIZE * 2)  # Double buffer size to ensure no data loss
        self.device = device
        
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f'Status: {status}')
        self.buffer.extend(indata.flatten())
        while len(self.buffer) >= BUFFER_SIZE:
            chunk = np.array(list(self.buffer)[:BUFFER_SIZE])
            self.buffer.rotate(-BUFFER_SIZE)
            for _ in range(BUFFER_SIZE):
                self.buffer.pop()
            self.chunk_ready.emit(chunk)
        
    def run(self):
        self.running = True
        try:
            device_id = self.device if self.device is not None else sd.default.device[0]

            try:
                sd.check_input_settings(device=device_id, samplerate=SAMPLE_RATE, channels=1)
            except Exception as e:
                print(f"Invalid audio input device {device_id}: {e}")
                self.running = False
                return

            with sd.InputStream(
                device=device_id,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype=np.float32,
                callback=self.audio_callback,
                blocksize=1024,  # Smaller blocksize for more frequent updates
            ):
                while self.running:
                    sd.sleep(100)  # Sleep to prevent high CPU usage
        except Exception as e:
            print(f"Error in AudioStreamer: {str(e)}")
            self.running = False
    
    def stop(self):
        self.running = False
