import tkinter as tk
from tkinter import ttk, filedialog
import pyaudio
import requests
import threading
import numpy as np
import time
import subprocess
import os
import json
import cv2
import zmq
import msgpack
import msgpack_numpy as m
import pyvirtualcam

m.patch()

SERVER_URL = "http://127.0.0.1:8080/convert"
UPDATE_URL = "http://127.0.0.1:8080/update_target"
HEALTH_URL = "http://127.0.0.1:8080/health"
ZMQ_SERVER_ADDRESS = "tcp://localhost:5558"
ZMQ_CLIENT_ADDRESS = "tcp://localhost:5559"
ZMQ_UPDATE_ADDRESS = "tcp://localhost:5560"

chunk = 16000
audio_format = pyaudio.paFloat32
channels = 1
rate = 16000
silence_threshold = 0.003

audio = pyaudio.PyAudio()
recording_flag = False
recording_thread = None
selected_device_index = None

video_running = False
video_thread = None
zmq_context = None
sender = None
receiver = None
cap = None
virtual_cam = None
frame_buffer = []


# Utility functions
def get_input_devices():
    devices = []
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            devices.append((i, info.get("name")))
    return devices


def update_selected_device(event=None):
    global selected_device_index
    selection = device_combobox.get()
    for idx, name in get_input_devices():
        if name == selection:
            selected_device_index = idx
            print(f"Selected device updated to index: {selected_device_index} - {name}")
            break


devices = get_input_devices()
if devices:
    selected_device_index = devices[0][0]
else:
    raise Exception("No input devices found.")


# Audio functions
def send_to_server(audio_data):
    try:
        start_time = time.time()
        response = requests.post(SERVER_URL, data=audio_data.tobytes())
        latency = time.time() - start_time
        if response.status_code == 200:
            response_data = response.json()
            processed_audio = np.array(
                response_data["processed_audio"], dtype=np.float32
            )
            print(
                f"Total Latency (API): {latency:.3f}s | Chunk Duration: {len(audio_data) / rate:.2f}s"
            )
            time.sleep(0.1)
            play_audio(processed_audio)
        else:
            print(f"Error sending data: {response.status_code}")
    except Exception as e:
        print(f"Error connecting to server: {e}")


def play_audio(data):
    print("Starting audio playback...")
    stream = audio.open(format=audio_format, channels=channels, rate=rate, output=True)
    stream.write(data.astype(np.float32).tobytes())
    stream.stop_stream()
    stream.close()
    print("Audio playback finished.")


def record_audio():
    global chunk, selected_device_index
    stream = audio.open(
        format=audio_format,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=chunk,
        input_device_index=selected_device_index,
    )
    while recording_flag:
        data = np.frombuffer(stream.read(chunk), dtype=np.float32)
        if not is_silent(data):
            print("Audio chunk recorded, sending to server...")
            threading.Thread(target=send_to_server, args=(data,)).start()
        else:
            print("Silence detected, chunk not sent.")
    stream.stop_stream()
    stream.close()


def start_recording():
    global recording_flag, recording_thread
    if not recording_flag:
        recording_flag = True
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.start()
        print("Recording started.")


def stop_recording():
    global recording_flag, recording_thread
    recording_flag = False
    if recording_thread is not None:
        recording_thread.join()
    print("Recording stopped.")


def toggle_recording(event=None):
    if recording_flag:
        stop_recording()
        record_button.config(text="Push-to-Talk")
    else:
        start_recording()
        record_button.config(text="Stop Recording")


def is_silent(data):
    return np.abs(data).mean() < silence_threshold


def connect_to_server():
    try:
        response = requests.get(HEALTH_URL)
        if response.status_code == 200:
            status_label.config(text="Connected")
        else:
            status_label.config(text="Connection Error")
    except Exception as e:
        status_label.config(text="Connection Error")
        print(f"Error connecting: {e}")


def update_chunk_size():
    global chunk, recording_flag
    try:
        new_value = int(chunk_spinbox.get())
        if new_value != chunk:
            chunk = new_value
            print(f"Chunk size updated to: {chunk} samples")
            if recording_flag:
                stop_recording()
                start_recording()
    except ValueError:
        print("Invalid value for chunk size.")


def browse_target_audio():
    file_path = filedialog.askopenfilename(
        title="Select Target Audio",
        filetypes=(("Audio Files", "*.wav *.mp3 *.flac"), ("All Files", "*.*")),
    )
    if file_path:
        target_audio_entry.delete(0, tk.END)
        target_audio_entry.insert(0, file_path)


def upload_target_audio():
    file_path = target_audio_entry.get()
    if not file_path:
        upload_status_label.config(text="No file selected")
        return
    scp_command = [
        "scp",
        "-i",
        r"Private Key SSH",
        "-P",
        "Port",
        file_path,
        "file_path",
    ]
    try:
        result = subprocess.run(
            scp_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        upload_status_label.config(text="Upload Successful")
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        upload_status_label.config(text="Upload Failed")
        print(e.stderr.decode())


def update_target_on_server():
    file_path = target_audio_entry.get()
    if not file_path:
        update_status_label.config(text="No file selected")
        return
    filename = os.path.basename(file_path)
    payload = {"target_filename": filename}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(UPDATE_URL, data=json.dumps(payload), headers=headers)
        result = response.json()
        if response.status_code == 200:
            update_status_label.config(text="Update Successful")
        else:
            update_status_label.config(text="Update Failed")
        print(result)
    except Exception as e:
        update_status_label.config(text="Update Failed")
        print(f"Error updating target on server: {e}")


# Video functions
def initialize_video():
    global zmq_context, sender, receiver, cap, virtual_cam
    zmq_context = zmq.Context()
    sender = zmq_context.socket(zmq.PUSH)
    sender.setsockopt(zmq.SNDHWM, 1)
    sender.setsockopt(zmq.LINGER, 0)
    sender.connect(ZMQ_SERVER_ADDRESS)
    receiver = zmq_context.socket(zmq.PULL)
    receiver.setsockopt(zmq.RCVHWM, 1)
    receiver.setsockopt(zmq.LINGER, 0)
    receiver.connect(ZMQ_CLIENT_ADDRESS)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            video_status_label.config(text="Webcam Error")
            return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        virtual_cam = pyvirtualcam.Camera(
            width=1280,
            height=720,
            fps=30,
            fmt=pyvirtualcam.PixelFormat.BGR,
        )
        print(f"Virtual camera initialized using {virtual_cam.backend} backend")
        return True
    except Exception as e:
        print(f"Error initializing virtual camera: {e}")
        return False


def process_video():
    global video_running, cap, sender, receiver, virtual_cam
    last_frame_time = time.time()
    frames_sent = 0
    frames_received = 0
    try:
        while video_running:
            for _ in range(5):
                cap.read()
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                time.sleep(0.1)
                continue
            current_time = time.time()
            if frame.shape[:2] != (720, 1280):
                frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
            if (frames_sent <= frames_received) or (
                current_time - last_frame_time > 1.0
            ):
                try:
                    while receiver.poll(timeout=0):
                        receiver.recv(flags=zmq.NOBLOCK)
                        frames_received += 1
                except zmq.ZMQError:
                    pass
                _, encoded_frame = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                sender.send(msgpack.packb(encoded_frame.tobytes()))
                frames_sent += 1
                last_frame_time = current_time
                print(f"Frame sent: {frames_sent}")
            try:
                data = receiver.recv(flags=zmq.NOBLOCK)
                frames_received += 1
                processed_frame_data = np.frombuffer(
                    msgpack.unpackb(data), dtype=np.uint8
                )
                processed_frame = cv2.imdecode(processed_frame_data, cv2.IMREAD_COLOR)
                if processed_frame.shape[:2] != (720, 1280):
                    processed_frame = cv2.resize(
                        processed_frame, (1280, 720), interpolation=cv2.INTER_CUBIC
                    )
                if virtual_cam is not None:
                    virtual_cam.send(processed_frame)
                    print(
                        f"Frame displayed: {frames_received}, delay: {time.time() - last_frame_time:.2f}s"
                    )
            except zmq.ZMQError:
                pass
            time.sleep(0.05)
    except Exception as e:
        print(f"Video processing error: {e}")
    finally:
        cleanup_video()
        video_status_label.config(text="Video Stopped")
        root.after(0, lambda: video_button.config(text="Start Video"))


def cleanup_video():
    global cap, zmq_context, sender, receiver, virtual_cam
    if receiver is not None:
        try:
            while receiver.poll(timeout=0):
                receiver.recv(flags=zmq.NOBLOCK)
        except zmq.ZMQError:
            pass
    if cap is not None:
        cap.release()
    if sender is not None:
        sender.close()
    if receiver is not None:
        receiver.close()
    if zmq_context is not None:
        zmq_context.term()
    if virtual_cam is not None:
        virtual_cam.close()


def start_video():
    global video_running, video_thread
    if not video_running:
        if initialize_video():
            video_running = True
            video_thread = threading.Thread(target=process_video)
            video_thread.daemon = True
            video_thread.start()
            video_button.config(text="Stop Video")
            video_status_label.config(text="Video Running")
        else:
            video_status_label.config(text="Failed to start video")


def stop_video():
    global video_running, video_thread
    video_running = False
    if video_thread is not None:
        video_thread.join(timeout=1.0)
    cleanup_video()
    video_button.config(text="Start Video")
    video_status_label.config(text="Video Stopped")


def toggle_video():
    if video_running:
        stop_video()
    else:
        start_video()


def connect_ssh_for_video():
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        ssh_command = [
            "ssh",
            "-i",
            r"your private Key for SSH",
            "-p",
            "port",
            "XXX@XXX.XXX.XXX.XXX",
            "cd /home/XXXX/Real-Time-Deepfake-Pipeline && python server.py &",
        ]
        subprocess.Popen(
            ssh_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creation_flags,
        )
        ssh_video_status_label.config(text="Connected to Video Server")
    except Exception as e:
        ssh_video_status_label.config(text="Connection Failed")
        print(f"SSH connection error: {e}")


def browse_video_source():
    file_path = filedialog.askopenfilename(
        title="Select Video Source Image",
        filetypes=(("Image Files", "*.jpg *.jpeg *.png"), ("All Files", "*.*")),
    )
    if file_path:
        video_source_entry.delete(0, tk.END)
        video_source_entry.insert(0, file_path)


def upload_video_source():
    file_path = video_source_entry.get()
    if not file_path:
        video_upload_status_label.config(text="No file selected")
        return
    scp_command = [
        "scp",
        "-i",
        r"your private Key for SSH",
        "-P",
        "port",
        file_path,
        "file_path",
    ]
    try:
        result = subprocess.run(
            scp_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        video_upload_status_label.config(text="Upload Successful")
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        video_upload_status_label.config(text="Upload Failed")
        print(e.stderr.decode())


def update_video_config_on_server():
    file_path = video_source_entry.get()
    if not file_path:
        video_update_status_label.config(text="No video source selected")
        return
    filename = os.path.basename(file_path)
    server_path = f"/home/ali/Real-Time-Deepfake-Pipeline/video/{filename}"
    try:
        new_upscale = float(upscale_entry.get())
    except ValueError:
        video_update_status_label.config(text="Invalid upscale value")
        return
    update_thread = threading.Thread(
        target=send_update_config,
        args=(server_path, new_upscale, disable_face_var.get()),
    )
    update_thread.daemon = True
    update_thread.start()
    video_update_status_label.config(text="Sending update...")


def send_update_config(server_path, new_upscale, disable_face):
    try:
        context = zmq.Context()
        update_socket = context.socket(zmq.REQ)
        update_socket.setsockopt(zmq.RCVTIMEO, 5000)
        update_socket.connect(ZMQ_UPDATE_ADDRESS)
        update_data = {
            "source_image": server_path,
            "upscale": new_upscale,
            "disable_face_enhancement": disable_face,
        }
        update_socket.send_string(json.dumps(update_data))
        try:
            reply = update_socket.recv_string()
            root.after(0, lambda: video_update_status_label.config(text=reply))
        except zmq.error.Again:
            root.after(
                0, lambda: video_update_status_label.config(text="Update timed out")
            )
        update_socket.close()
        context.term()
    except Exception as e:
        print(f"Error sending update command: {e}")
        root.after(
            0, lambda: video_update_status_label.config(text=f"Update Failed: {str(e)}")
        )


# GUI
root = tk.Tk()
root.title("Deepfake Audio & Video Client")
root.geometry("900x800")

mainframe = ttk.Frame(root, padding="10")
mainframe.grid(column=0, row=0, sticky="NSEW")
mainframe.columnconfigure(0, weight=1)
mainframe.columnconfigure(1, weight=1)

server_frame = ttk.LabelFrame(mainframe, text="Server Connection", padding="10")
server_frame.grid(column=0, row=0, columnspan=2, padx=5, pady=5, sticky="EW")

connect_button = ttk.Button(
    server_frame, text="Connect Audio Server", command=connect_to_server
)
connect_button.grid(column=0, row=0, padx=5, pady=5, sticky="EW")

status_label = ttk.Label(server_frame, text="Not connected")
status_label.grid(column=1, row=0, padx=5, pady=5, sticky="EW")

ssh_video_button = ttk.Button(
    server_frame, text="Connect Video Server", command=connect_ssh_for_video
)
ssh_video_button.grid(column=0, row=1, padx=5, pady=5, sticky="EW")

ssh_video_status_label = ttk.Label(server_frame, text="Not connected")
ssh_video_status_label.grid(column=1, row=1, padx=5, pady=5, sticky="EW")

audio_frame = ttk.LabelFrame(mainframe, text="Audio Controls", padding="10")
audio_frame.grid(column=0, row=1, columnspan=2, padx=5, pady=5, sticky="EW")

device_label = ttk.Label(audio_frame, text="Audio Input Device:")
device_label.grid(column=0, row=0, padx=5, pady=5, sticky="W")

device_list = [name for idx, name in get_input_devices()]
device_combobox = ttk.Combobox(audio_frame, values=device_list, state="readonly")
device_combobox.current(0)
device_combobox.grid(column=1, row=0, padx=5, pady=5, sticky="EW")
device_combobox.bind("<<ComboboxSelected>>", update_selected_device)

record_button = ttk.Button(audio_frame, text="Push-to-Talk", command=toggle_recording)
record_button.grid(column=0, row=1, padx=5, pady=5, sticky="EW")

chunk_label = ttk.Label(audio_frame, text="Chunk Size:")
chunk_label.grid(column=1, row=1, padx=5, pady=5, sticky="W")

chunk_var = tk.StringVar(value=str(chunk))
chunk_spinbox = ttk.Spinbox(
    audio_frame,
    from_=8000,
    to=32000,
    increment=1000,
    textvariable=chunk_var,
    command=update_chunk_size,
    width=10,
)
chunk_spinbox.grid(column=1, row=1, padx=5, pady=5, sticky="E")

target_frame = ttk.LabelFrame(mainframe, text="Target Audio Upload", padding="10")
target_frame.grid(column=0, row=2, columnspan=2, padx=5, pady=5, sticky="EW")

target_audio_entry = ttk.Entry(target_frame, width=50)
target_audio_entry.grid(column=0, row=0, padx=5, pady=5, sticky="EW")

browse_button = ttk.Button(
    target_frame, text="Browse Audio Source", command=browse_target_audio
)
browse_button.grid(column=1, row=0, padx=5, pady=5)

upload_button = ttk.Button(
    target_frame, text="Upload Target Audio", command=upload_target_audio
)
upload_button.grid(column=0, row=1, padx=5, pady=5, sticky="EW")

upload_status_label = ttk.Label(target_frame, text="")
upload_status_label.grid(column=1, row=1, padx=5, pady=5, sticky="EW")

update_button = ttk.Button(
    target_frame, text="Update Audio Target on Server", command=update_target_on_server
)
update_button.grid(column=0, row=2, padx=5, pady=5, sticky="EW")

update_status_label = ttk.Label(target_frame, text="")
update_status_label.grid(column=1, row=2, padx=5, pady=5, sticky="EW")

vcam_status_frame = ttk.LabelFrame(
    mainframe, text="Virtual Camera Status", padding="10"
)
vcam_status_frame.grid(column=0, row=5, columnspan=2, padx=5, pady=5, sticky="EW")

vcam_status_label = ttk.Label(vcam_status_frame, text="Virtual Camera: Not active")
vcam_status_label.grid(column=0, row=0, padx=5, pady=5, sticky="W")


def update_vcam_status():
    status = (
        "Virtual Camera: Active"
        if virtual_cam is not None and video_running
        else "Virtual Camera: Not active"
    )
    vcam_status_label.config(text=status)
    root.after(1000, update_vcam_status)


update_vcam_status()

video_frame = ttk.LabelFrame(mainframe, text="Video Controls", padding="10")
video_frame.grid(column=0, row=4, columnspan=2, padx=5, pady=5, sticky="EW")

video_button = ttk.Button(video_frame, text="Start Video", command=toggle_video)
video_button.grid(column=0, row=0, padx=5, pady=5, sticky="EW")

video_status_label = ttk.Label(video_frame, text="Video Stopped")
video_status_label.grid(column=1, row=0, padx=5, pady=5, sticky="EW")

video_source_frame = ttk.LabelFrame(mainframe, text="Video Source Update", padding="10")
video_source_frame.grid(column=0, row=6, columnspan=2, padx=5, pady=5, sticky="EW")

video_source_entry = ttk.Entry(video_source_frame, width=50)
video_source_entry.grid(column=0, row=0, padx=5, pady=5, sticky="EW")

video_browse_button = ttk.Button(
    video_source_frame, text="Browse Video Source", command=browse_video_source
)
video_browse_button.grid(column=1, row=0, padx=5, pady=5)

video_upload_button = ttk.Button(
    video_source_frame, text="Upload Video Source", command=upload_video_source
)
video_upload_button.grid(column=0, row=1, padx=5, pady=5, sticky="EW")

video_upload_status_label = ttk.Label(video_source_frame, text="")
video_upload_status_label.grid(column=1, row=1, padx=5, pady=5, sticky="EW")

upscale_label = ttk.Label(video_source_frame, text="Upscale Factor:")
upscale_label.grid(column=0, row=2, padx=5, pady=5, sticky="W")

upscale_entry = ttk.Entry(video_source_frame, width=10)
upscale_entry.insert(0, "0.4")
upscale_entry.grid(column=1, row=2, padx=5, pady=5, sticky="W")


disable_face_var = tk.BooleanVar(value=False)
disable_face_check = ttk.Checkbutton(
    video_source_frame, text="Disable Face Enhancement", variable=disable_face_var
)
disable_face_check.grid(column=0, row=3, padx=5, pady=5, sticky="W")


video_update_button = ttk.Button(
    video_source_frame,
    text="Update Video Config on Server",
    command=update_video_config_on_server,
)
video_update_button.grid(column=0, row=4, padx=5, pady=5, sticky="EW")

video_update_status_label = ttk.Label(video_source_frame, text="")
video_update_status_label.grid(column=1, row=4, padx=5, pady=5, sticky="EW")

root.bind("<x>", toggle_recording)
root.bind("<X>", toggle_recording)
root.bind("<v>", lambda event: toggle_video())
root.bind("<V>", lambda event: toggle_video())

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

root.mainloop()

if recording_flag:
    stop_recording()
if video_running:
    stop_video()
audio.terminate()
