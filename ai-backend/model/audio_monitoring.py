import sounddevice as sd
import numpy as np
import webrtcvad
import queue
import threading
import time
import wave
from datetime import datetime
from pyAudioAnalysis import audioSegmentation as aS
import os
import whisper

# Settings
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
VAD_MODE = 2  # 0-3, 3 is most aggressive
AUDIO_LOG_FILE = 'audio_log.wav'
EVENT_LOG_FILE = 'audio_events.log'
DIARIZATION_INTERVAL = 30  # seconds
TRANSCRIPT_LOG_FILE = 'audio_transcript.log'

# Initialize VAD
vad = webrtcvad.Vad(VAD_MODE)

# Audio buffer for diarization
audio_buffer = []
q = queue.Queue()

# Logging
def log_event(msg):
    print(msg)
    with open(EVENT_LOG_FILE, 'a') as f:
        f.write(f"{datetime.now().isoformat()} - {msg}\n")

def log_transcript(msg):
    print(f"[TRANSCRIPT] {msg}")
    with open(TRANSCRIPT_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now().isoformat()} - {msg}\n")

# Audio callback
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    q.put(indata.copy())

# Thread: Save audio to buffer and log file
def audio_saver():
    wf = wave.open(AUDIO_LOG_FILE, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    while True:
        data = q.get()
        audio_buffer.append(data)
        wf.writeframes((data * 32767).astype(np.int16).tobytes())

def vad_and_log():
    speech_detected = False
    silence_count = 0
    speech_count = 0
    while True:
        if not audio_buffer:
            time.sleep(0.1)
            continue
        frame = audio_buffer[-1]
        # Convert to 16-bit PCM
        pcm = (frame.flatten() * 32767).astype(np.int16).tobytes()
        is_speech = vad.is_speech(pcm, SAMPLE_RATE)
        if is_speech:
            speech_count += 1
            silence_count = 0
            if not speech_detected:
                log_event('Speech started')
                speech_detected = True
        else:
            silence_count += 1
            if speech_detected and silence_count > 10:
                log_event('Speech ended')
                speech_detected = False
        time.sleep(FRAME_DURATION / 1000)

def diarization_worker():
    while True:
        time.sleep(DIARIZATION_INTERVAL)
        # Save last N seconds of audio to temp file
        if len(audio_buffer) < int(SAMPLE_RATE * DIARIZATION_INTERVAL / FRAME_SIZE):
            continue
        temp_wav = 'temp_diar.wav'
        segment = np.concatenate(audio_buffer[-int(SAMPLE_RATE * DIARIZATION_INTERVAL / FRAME_SIZE):])
        wav_data = (segment * 32767).astype(np.int16)
        with wave.open(temp_wav, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(wav_data.tobytes())
        # Run speaker diarization
        try:
            diar_result = aS.speaker_diarization(temp_wav, n_speakers=2, plot_res=False)
            # diar_result can be (flags, classes, centers) or (flags, -1, -1) on error
            if len(diar_result) == 3:
                _, classes, _ = diar_result
            else:
                classes = diar_result[1] if len(diar_result) > 1 else -1
            if isinstance(classes, (list, np.ndarray)) and len(set(classes)) > 1:
                log_event('Multiple speakers detected!')
        except Exception as e:
            log_event(f'Diarization error: {e}')
        os.remove(temp_wav)

def transcription_worker():
    print("[TRANSCRIBE] Transcription thread started.")
    model = whisper.load_model("base")  # You can use "small", "medium", or "large" for more accuracy
    while True:
        time.sleep(DIARIZATION_INTERVAL)
        needed_frames = int(SAMPLE_RATE * DIARIZATION_INTERVAL / FRAME_SIZE)
        print(f"[TRANSCRIBE] Checking audio buffer: {len(audio_buffer)} frames, need {needed_frames}.")
        if len(audio_buffer) < needed_frames:
            print("[TRANSCRIBE] Not enough audio for transcription, skipping.")
            continue
        temp_wav = 'temp_transcribe.wav'
        segment = np.concatenate(audio_buffer[-needed_frames:])
        wav_data = (segment * 32767).astype(np.int16)
        with wave.open(temp_wav, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(wav_data.tobytes())
        try:
            print(f"[TRANSCRIBE] Transcribing {temp_wav}...")
            result = model.transcribe(temp_wav)
            print(f"[TRANSCRIBE] Transcription result: {result['text']}")
            log_transcript(result['text'])
        except Exception as e:
            print(f"[TRANSCRIBE] Transcription error: {e}")
            log_transcript(f"Transcription error: {e}")
        os.remove(temp_wav)

# Start audio stream
stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback, blocksize=FRAME_SIZE)
stream.start()

# Start threads
threading.Thread(target=audio_saver, daemon=True).start()
threading.Thread(target=vad_and_log, daemon=True).start()
threading.Thread(target=diarization_worker, daemon=True).start()
threading.Thread(target=transcription_worker, daemon=True).start()

print("Audio monitoring started. Press Ctrl+C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting...")
    stream.stop()
    stream.close()

# Instructions:
# 1. Activate your Python 3.10 environment and install requirements: pip install -r requirements.txt
# 2. Run: python audio_monitoring.py
# 3. Speech and multiple speaker events will be shown in the terminal and saved to audio_events.log
# 4. The full audio will be saved to audio_log.wav 