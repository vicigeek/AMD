#!/usr/bin/env /var/lib/asterisk/agi-bin/venv310/bin/python
import sys, os, time, logging, fcntl, audioop
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.serialization
import librosa
from agi import AGI

# --- CONFIG --- 
LOG_FILE    = "/var/log/amd/amd2.log"
MODEL_PATH  = "/opt/models/cnnVoicemailDetection/cnnModel.pt"
ORIG_SR     = 8000
TARGET_SR   = 16000
MIN_SEC     = 0.5
MIN_BYTES   = int(ORIG_SR * MIN_SEC)
MAX_WAIT    = 2.0
AUDIO_FD    = 3
CHUNK_SIZE  = 4096
N_MFCC      = 20
FIXED_WIDTH = 40
N_FFT       = 400
HOP_LEN     = 160

# --- LOGGING ---
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stderr)]
)
log = logging.getLogger("amd2")

# --- MATCHING CNN ARCHITECTURE ---
class AudioClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # conv1: [1→16], kernel=3, pad=1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        # conv2: [16→32]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        # conv3: [32→64]
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        # fc layers: flatten 64×20×40 = 51200 → 64 → 2
        self.fc1   = nn.Linear(64 * N_MFCC * FIXED_WIDTH, 64)
        self.fc2   = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --- LOAD FULL CHECKPOINT ---
def load_model(path):
    try:
        torch.serialization.add_safe_globals({'AudioClassifierCNN': AudioClassifierCNN})
        model = torch.load(path, map_location="cpu", weights_only=False)
        model.eval()
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        log.info("✅ Model loaded successfully")
        return model
    except Exception as e:
        log.critical("Model load failed: %s", e)
        return None

# --- AGI SCRIPT ---
def main():
    agi = AGI()
    uid = agi.env.get("agi_uniqueid","noid")
    ani = agi.env.get("agi_callerid","unknown")
    log.info(f"AGI start – UID={uid} ANI={ani}")

    model = load_model(MODEL_PATH)
    if not model:
        agi.set_variable("AMDSTATUS","AIERR")
        agi.set_variable("AMDCAUSE","MODEL")
        return

    # Buffer μ-law
    buf = b""
    fcntl.fcntl(AUDIO_FD, fcntl.F_SETFL, os.O_NONBLOCK)
    deadline = time.time() + MAX_WAIT
    while time.time() < deadline and len(buf) < MIN_BYTES:
        try:
            chunk = os.read(AUDIO_FD, CHUNK_SIZE)
            if chunk:
                buf += chunk
                log.debug(f"Buffered {len(buf)}/{MIN_BYTES} bytes")
            else:
                time.sleep(0.01)
        except BlockingIOError:
            time.sleep(0.01)
        except Exception as e:
            log.error("Audio read error: %s", e)
            break

    if len(buf) < MIN_BYTES:
        log.warning("❌ Not enough audio")
        agi.set_variable("AMDSTATUS","NOAUDIO")
        agi.set_variable("AMDCAUSE","NOAUDIO")
        return

    # Decode & resample
    try:
        pcm16 = audioop.ulaw2lin(buf, 2)
        res16,_ = audioop.ratecv(pcm16, 2, 1, ORIG_SR, TARGET_SR, None)
        audio = np.frombuffer(res16, dtype=np.int16).astype(np.float32)/32768.0
        log.debug(f"Decoded to {audio.shape[0]} samples @16kHz")
    except Exception as e:
        log.error("Decode error: %s", e)
        agi.set_variable("AMDSTATUS","AIERR")
        agi.set_variable("AMDCAUSE","DECODE")
        return

    # MFCC extraction
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=TARGET_SR,
                                    n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LEN)
        log.debug(f"Raw MFCC shape: {mfcc.shape}")
        if mfcc.shape[1] < FIXED_WIDTH:
            pad = FIXED_WIDTH - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad)), mode="constant")
        else:
            mfcc = mfcc[:, :FIXED_WIDTH]
        inp = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        log.debug(f"MFCC tensor shape: {tuple(inp.shape)}")
    except Exception as e:
        log.error("MFCC error: %s", e)
        agi.set_variable("AMDSTATUS","AIERR")
        agi.set_variable("AMDCAUSE","MFCC")
        return

    # Inference
    try:
        with torch.no_grad():
            logits = model(inp)
        pred_id = int(logits.argmax(dim=1)[0])
        conf    = float(torch.softmax(logits, dim=1)[0,pred_id])
        label   = "HUMAN" if pred_id==0 else "MACHINE"
        log.info(f"✅ Prediction: {label} (conf={conf:.2f})")
        agi.set_variable("AMDSTATUS",label)
        agi.set_variable("AMDCAUSE",f"{conf:.2f}")
    except Exception as e:
        log.error("Inference error: %s", e)
        agi.set_variable("AMDSTATUS","AIERR")
        agi.set_variable("AMDCAUSE","INFER")

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        log.critical("Fatal error: %s", e)
        sys.exit(1)
