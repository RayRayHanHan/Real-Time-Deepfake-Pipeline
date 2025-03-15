import cv2
import zmq
import msgpack
import msgpack_numpy as m
import numpy as np
from wrapper import Wrapper
import time
import os

m.patch()

ZMQ_RECEIVE_ADDRESS = "tcp://0.0.0.0:5558"
ZMQ_SEND_ADDRESS = "tcp://0.0.0.0:5559"

def process_frame(frame, wrapper):
    # Frame auf 1280x720 skalieren
    frame_resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
    processed_frame = wrapper.generate(frame_resized)
    return processed_frame

def main():
    # Wrapper initialisieren mit dem Standard-Source-Image
    source_path = "./image/Elon Musk.jpg"  # Standard-Source Image
    print("Initializing wrapper...")
    wrapper = Wrapper(source_path)
    last_mod_time = os.path.getmtime(source_path)

    context = zmq.Context()

    # Socket zum Empfangen von Frames
    receiver = context.socket(zmq.PULL)
    receiver.bind(ZMQ_RECEIVE_ADDRESS)

    # Socket zum Senden der verarbeiteten Frames
    sender = context.socket(zmq.PUSH)
    sender.bind(ZMQ_SEND_ADDRESS)

    print("Server is running and waiting for frames...")

    try:
        while True:
            # Prüfe, ob sich das Quellbild geändert hat
            new_mod_time = os.path.getmtime(source_path)
            if new_mod_time != last_mod_time:
                print("Source image updated, reinitializing wrapper...")
                wrapper = Wrapper(source_path)
                last_mod_time = new_mod_time

            start_time = time.time()
            data = receiver.recv()
            compressed_frame = np.frombuffer(msgpack.unpackb(data), dtype=np.uint8)
            frame = cv2.imdecode(compressed_frame, cv2.IMREAD_REDUCED_COLOR_2)

            # Frame verarbeiten
            processed_frame = process_frame(frame, wrapper)

            # Zur Originalgröße zurückskalieren, falls nötig
            processed_frame = cv2.resize(
                processed_frame,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

            # Komprimieren und an den Client senden
            _, encoded_frame = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            sender.send(msgpack.packb(encoded_frame.tobytes()))

            elapsed_time = time.time() - start_time
            print(f"Processing time: {elapsed_time:.4f} seconds")
    except KeyboardInterrupt:
        print("Server interrupted.")
    finally:
        receiver.close()
        sender.close()
        context.term()

if __name__ == "__main__":
    main()
