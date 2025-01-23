import pyaudio
import requests
import threading
import numpy as np
import wave
import signal
from collections import deque

SERVER_URL = 'http://127.0.0.1:8080/convert'


chunk = 10000
format = pyaudio.paFloat32
channels = 1
rate = 16000
buffer_size = 3
silence_threshold = 0.001


audio_buffer = deque(maxlen=buffer_size)


audio = pyaudio.PyAudio()


stop_signal = threading.Event()


# VB-Cable-Index bestimmen
def get_vb_cable_index():
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(numdevices):
        device_info = audio.get_device_info_by_host_api_device_index(0, i)
        if 'VB-Audio Virtual Cable' in device_info.get('name'):
            return i
    return None


vb_cable_index = 7
if vb_cable_index is None:
    raise Exception("VB-Cable not found.")
else:
    print(f"VB-Cable found at index {vb_cable_index}")



def is_silent(data):
    return np.abs(data).mean() < silence_threshold


def save_audio_to_file(data, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(data.astype(np.float32).tobytes())



def record_audio():
    print("Starte die Audio...")
    stream = audio.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk,
                        input_device_index=vb_cable_index)

    while not stop_signal.is_set():
        data = np.frombuffer(stream.read(chunk), dtype=np.float32)


        print(f"Erste 10 Samples der aufgenommenen Audiodaten: {data[:10]}")

        if is_silent(data):
            print("Stille erkannt, keine Daten gesendet...")
            continue


        audio_buffer.append(data)

        if len(audio_buffer) == buffer_size:
            audio_to_process = np.concatenate(list(audio_buffer), axis=0)
            audio_buffer.clear()

            print("Audio-Chunks aufgenommen und gesendet...")
            threading.Thread(target=send_to_server, args=(audio_to_process,)).start()

    stream.stop_stream()
    stream.close()


def send_to_server(audio_data):
    try:
        response = requests.post(SERVER_URL, data=audio_data.tobytes())
        if response.status_code == 200:
            response_data = response.json()
            processed_audio = np.array(response_data['processed_audio'], dtype=np.float32)
            print(f"Erste 10 Samples des empfangenen verarbeiteten Audios: {processed_audio[:10]}")


            save_audio_to_file(processed_audio, 'processed_audio.wav')
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



def signal_handler(sig, frame):
    print("Beende die Aufnahme...")
    stop_signal.set()



signal.signal(signal.SIGINT, signal_handler)


record_thread = threading.Thread(target=record_audio)
record_thread.start()

record_thread.join()

audio.terminate()
print("Audio-Aufnahme und -Wiedergabe beendet.")
