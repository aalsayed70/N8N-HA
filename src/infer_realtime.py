# src/infer_realtime.py
import numpy as np
import sounddevice as sd
import librosa
import onnxruntime as ort
from collections import deque
import time

SAMPLE_RATE = 16000
WINDOW_SEC = 1.0
STEP_SEC = 0.25  # stride between evaluations
THRESHOLD = 0.5  # adjust from evaluation; detection uses smoothed average
N_MELS = 40
# Min RMS gate to avoid random triggers on silence/background hum.
MIN_RMS = 0.005
# Set to None to use system default input (e.g., laptop mic). Otherwise set to device index or name.
MIC_DEVICE = None

MODEL_ONNX_PATH = "experiments/dary_cnn/dary_wakeword.onnx"

sess = ort.InferenceSession(MODEL_ONNX_PATH, providers=["CPUExecutionProvider"])

def extract_logmel(y):
    # y: 1D numpy, length ~ SAMPLE_RATE
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=512,
        hop_length=160,
        win_length=400,
        n_mels=N_MELS,
        power=2.0,
        center=True
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # normalize
    mean = log_mel.mean()
    std = log_mel.std() + 1e-9
    log_mel = (log_mel - mean) / std
    # shape: (1, 1, n_mels, time)
    return log_mel[np.newaxis, np.newaxis, :, :].astype(np.float32)

def run_model(y):
    log_mel = extract_logmel(y)
    outputs = sess.run(None, {"input": log_mel})
    logits = outputs[0][0]  # scalar
    prob = 1 / (1 + np.exp(-logits))
    return float(prob)

def main():
    buffer_len = int(WINDOW_SEC * SAMPLE_RATE)
    step_len = int(STEP_SEC * SAMPLE_RATE)

    ring_buffer = deque(maxlen=buffer_len)
    prob_smooth = deque(maxlen=4)

    def audio_callback(indata, frames, time_info, status):
        # indata: (frames, channels)
        if status:
            print(status)
        mono = indata[:, 0]
        for sample in mono:
            ring_buffer.append(sample)

    try:
        dev_info = sd.query_devices(MIC_DEVICE, "input")
        dev_name = dev_info["name"]
    except Exception as e:
        print(f"Mic selection issue ({e}); falling back to default input.")
        dev_name = sd.query_devices(None, "input")["name"]
        MIC_DEVICE = None

    with sd.InputStream(
        device=MIC_DEVICE,
        channels=1,
        samplerate=SAMPLE_RATE,
        callback=audio_callback,
        blocksize=step_len,
        dtype="float32"
    ):
        print(f"Listening for 'Dary' ... (device={dev_name})")
        last_trigger_time = 0
        min_trigger_interval = 1.5  # seconds between triggers

        while True:
            if len(ring_buffer) < buffer_len:
                time.sleep(0.05)
                continue

            y = np.array(ring_buffer, dtype=np.float32)
            rms = float(np.sqrt(np.mean(y ** 2) + 1e-9))
            if rms < MIN_RMS:
                # Too quiet; skip inference to avoid random spikes
                time.sleep(STEP_SEC * 0.9)
                continue

            prob = run_model(y)
            prob_smooth.append(prob)
            prob_avg = sum(prob_smooth) / len(prob_smooth)
            print(f"Score: {prob:.3f} (avg {prob_avg:.3f}) rms {rms:.4f}", end="\r")

            now = time.time()
            if prob_avg > THRESHOLD and (now - last_trigger_time) > min_trigger_interval:
                print(f"\nWake word detected! (prob={prob_avg:.3f}, rms={rms:.4f})")
                last_trigger_time = now

            time.sleep(STEP_SEC * 0.9)

if __name__ == "__main__":
    main()
