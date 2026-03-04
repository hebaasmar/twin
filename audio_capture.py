import pyaudio
import wave
import threading
import os

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
DEFAULT_SECONDS = 8
OUTPUT_DIR = "audio_clips"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def record_clip(filename, seconds=DEFAULT_SECONDS):
    """Record a fixed-length audio clip."""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)

    frames = []
    for _ in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    filepath = os.path.join(OUTPUT_DIR, filename)
    wf = wave.open(filepath, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filepath


class Recorder:
    """Variable-length recorder: call start(), then stop() to get the filepath."""

    def __init__(self, filename="live_query.wav"):
        self.filename = filename
        self._recording = False
        self._frames = []
        self._thread = None
        self._p = None

    def start(self):
        self._recording = True
        self._frames = []
        self._thread = threading.Thread(target=self._record, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop recording and save. Returns the filepath."""
        self._recording = False
        if self._thread:
            self._thread.join(timeout=5)
        return self._save()

    def _record(self):
        self._p = pyaudio.PyAudio()
        stream = self._p.open(
            format=FORMAT, channels=CHANNELS, rate=RATE,
            input=True, frames_per_buffer=CHUNK,
        )
        while self._recording:
            data = stream.read(CHUNK, exception_on_overflow=False)
            self._frames.append(data)
        stream.stop_stream()
        stream.close()
        self._p.terminate()

    def _save(self):
        filepath = os.path.join(OUTPUT_DIR, self.filename)
        p = pyaudio.PyAudio()
        wf = wave.open(filepath, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self._frames))
        wf.close()
        p.terminate()
        return filepath


if __name__ == "__main__":
    print("Recording 8 seconds of audio...")
    record_clip("test_clip.wav")
    print("Done. Check audio_clips/test_clip.wav")
