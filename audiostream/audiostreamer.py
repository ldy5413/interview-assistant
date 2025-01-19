import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QThread, pyqtSignal

CHUNK_DURATION = 3  # seconds
SAMPLE_RATE = 16000
class AudioStreamer(QThread):
    chunk_ready = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.running = False
        
    def run(self):
        self.running = True
        try:
            device_id = sd.default.device[0]  # Use default input device
            with sd.InputStream(device=device_id,
                              samplerate=SAMPLE_RATE,
                              channels=1,
                              dtype=np.float32) as stream:
                while self.running:
                    audio_chunk, _ = stream.read(SAMPLE_RATE * CHUNK_DURATION)
                    self.chunk_ready.emit(audio_chunk.flatten())
        except Exception as e:
            print(f"Error in AudioStreamer: {str(e)}")
            self.running = False
    
    def stop(self):
        self.running = False