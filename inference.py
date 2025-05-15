#!/usr/bin/env /var/lib/asterisk/agi-bin/venv310/bin/python
import sys
import os
import time
import logging
import fcntl
import audioop
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from agi import AGI

# --- CONFIGURATION ---
LOG_FILE      = "/var/log/amd/amd2.log"
MODEL_DIR     = "/opt/models/wav2vec-vm-finetune"
ORIG_SR       = 8000       # incoming μ-law sample rate
TARGET_SR     = 16000      # model expects 16 kHz
MIN_SEC       = 0.5        # seconds of audio to buffer before inference
MIN_BYTES     = int(ORIG_SR * MIN_SEC)  # ≈4000 bytes
MAX_WAIT      = 2.0        # seconds max to wait for MIN_BYTES
AUDIO_FD      = 3
CHUNK_SIZE    = 4096       # bytes per os.read

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stderr),
    ]
)
log = logging.getLogger("amd2")

def load_local_model(path):
    """Load feature extractor and model from local directory without internet."""
    try:
        fe = Wav2Vec2FeatureExtractor.from_pretrained(path, local_files_only=True)
        m  = Wav2Vec2ForSequenceClassification.from_pretrained(path, local_files_only=True)
        m.eval()
        log.info(f"Loaded model from {path}")
        return fe, m
    except Exception as e:
        log.critical(f"Failed to load local model: {e}")
        return None, None

def stream_ulaw(min_bytes, timeout_s):
    """Read μ-law bytes from FD=3 until we have min_bytes or timeout."""
    fcntl.fcntl(AUDIO_FD, fcntl.F_SETFL, os.O_NONBLOCK)
    buf = b""
    deadline = time.time() + timeout_s
    while time.time() < deadline and len(buf) < min_bytes:
        try:
            chunk = os.read(AUDIO_FD, CHUNK_SIZE)
            if chunk:
                buf += chunk
                log.debug(f"Buffered {len(buf):,}/{min_bytes:,} μ-law bytes")
            else:
                time.sleep(0.02)
        except BlockingIOError:
            time.sleep(0.02)
        except Exception as e:
            log.error(f"Error reading audio: {e}")
            break
    return buf

def ulaw_to_float_array(buf):
    """Convert μ-law→PCM16→upsample to 16 kHz→float32 numpy array."""
    try:
        # Decode μ-law to 16-bit PCM
        pcm16 = audioop.ulaw2lin(buf, 2)
        # Upsample 8 kHz→16 kHz
        res16, _ = audioop.ratecv(pcm16, 2, 1, ORIG_SR, TARGET_SR, None)
        # Convert to float32 in range [-1,1]
        arr = np.frombuffer(res16, dtype=np.int16).astype(np.float32) / 32768.0
        log.info(f"Converted to {arr.shape[0]} samples at {TARGET_SR} Hz")
        return arr
    except Exception as e:
        log.error(f"Decode/resample error: {e}")
        return None

def infer(audio, fe, model):
    """Run Wav2Vec2 inference on float32 audio array."""
    try:
        inputs = fe(audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_id = int(logits.argmax(dim=-1)[0])
        conf    = float(torch.softmax(logits, -1)[0, pred_id])
        label   = model.config.id2label[pred_id]
        log.info(f"Inference result: {label} (conf={conf:.2f})")
        return label, conf
    except Exception as e:
        log.error(f"Inference error: {e}")
        return "AIERR", 0.0

def main():
    agi = AGI()
    uid = agi.env.get("agi_uniqueid", "noid")
    log.info(f"AGI start – ANI={agi.env.get('agi_callerid')} UID={uid}")

    fe, model = load_local_model(MODEL_DIR)
    if model is None:
        agi.set_variable("AMDSTATUS", "AIERR")
        agi.set_variable("AMDCAUSE", "MODELLOAD")
        sys.exit(1)

    buf = stream_ulaw(MIN_BYTES, MAX_WAIT)
    if len(buf) < MIN_BYTES:
        log.warning("Insufficient audio or timeout")
        agi.set_variable("AMDSTATUS", "NOAUDIO")
        agi.set_variable("AMDCAUSE", "NOAUDIO")
        sys.exit(0)

    audio = ulaw_to_float_array(buf)
    if audio is None or audio.size == 0:
        agi.set_variable("AMDSTATUS", "AIERR")
        agi.set_variable("AMDCAUSE", "DECODE")
        sys.exit(1)

    label, conf = infer(audio, fe, model)
    agi.set_variable("AMDSTATUS", label)
    agi.set_variable("AMDCAUSE", f"{conf:.2f}")
    sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.critical(f"Fatal script error: {e}")
        sys.exit(1)
