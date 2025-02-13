import pyaudio
import requests
import threading
import numpy as np
import signal
import time


SERVER_URL = 'http://127.0.0.1:8080/convert'

chunk = 72000
format = pyaudio.paFloat32
channels = 1
rate = 16000
silence_threshold = 0.003

audio = pyaudio.PyAudio()
stop_signal = threading.Event()

def get_vb_cable_index():
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(numdevices):
        device_info = audio.get_device_info_by_host_api_device_index(0, i)
        if 'VB-Audio Virtual Cable' in device_info.get('name'):
            return i
    return None

vb_cable_index = 8
if vb_cable_index is None:
    raise Exception("VB-Cable not found.")
else:
    print(f"VB-Cable found at index {vb_cable_index}")

def is_silent(data):
    return np.abs(data).mean() < silence_threshold

def send_to_server(audio_data):
    try:
        start_time = time.time()
        response = requests.post(SERVER_URL, data=audio_data.tobytes())
        latency = time.time() - start_time
        if response.status_code == 200:
            response_data = response.json()
            processed_audio = np.array(response_data['processed_audio'], dtype=np.float32)
            print(f"Total Latency (API): {latency:.3f}s | Chunk Duration: {len(audio_data)/rate:.2f}s")
            time.sleep(0.1)
            play_audio(processed_audio)
        else:
            print(f"Fehler beim Senden der Daten: {response.status_code}")
    except Exception as e:
        print(f"Fehler beim Verbinden mit dem Server: {e}")

def play_audio(data):
    print("Starte die Audiowiedergabe mit VB-Cable...")
    stream = audio.open(format=format, channels=channels, rate=rate, output=True)
    stream.write(data.astype(np.float32).tobytes())
    stream.stop_stream()
    stream.close()
    print("Audiowiedergabe beendet.")

def record_audio():
    print("Starte die Audioaufnahme...")
    stream = audio.open(format=format, channels=channels, rate=rate,
                        input=True, frames_per_buffer=chunk, input_device_index=vb_cable_index)
    while not stop_signal.is_set():
        data = np.frombuffer(stream.read(chunk), dtype=np.float32)
        if not is_silent(data):
            print("Audio-Chuck aufgenommen, Sende an Server...")
            threading.Thread(target=send_to_server, args=(data,)).start()
        else:
            print("Stille erkannt, Chunk wird nicht gesendet.")
    stream.stop_stream()
    stream.close()

def signal_handler(sig, frame):
    print("Beende die Aufnahme...")
    stop_signal.set()

signal.signal(signal.SIGINT, signal_handler)
record_thread = threading.Thread(target=record_audio)
record_thread.start()
record_thread.join()
audio.terminate()
print("Audio-Aufnahme und -Wiedergabe beendet.")
