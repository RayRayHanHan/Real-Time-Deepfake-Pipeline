import cv2
import zmq
import msgpack
import msgpack_numpy as m
import numpy as np
from wrapper import Wrapper
import time
import argparse
import os
import json

m.patch()

ZMQ_RECEIVE_ADDRESS = "tcp://0.0.0.0:5558"
ZMQ_SEND_ADDRESS = "tcp://0.0.0.0:5559"
ZMQ_UPDATE_ADDRESS = "tcp://0.0.0.0:5560"

def process_frame(frame, wrapper):
    frame_resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
    processed_frame = wrapper.generate(frame_resized)
    return processed_frame

def main():
    parser = argparse.ArgumentParser(description="Real-Time Deepfake Pipeline Server")
    parser.add_argument("--source_image", type=str, default="./image.jpg", help="Path to the source image.")
    parser.add_argument("--gfpgan_path", type=str, default="models/GFPGANv1.3.pth", help="Path to the GFPGAN model file.")
    parser.add_argument("--inswapper_path", type=str, default="models/inswapper_128_fp16.onnx", help="Path to the inswapper model file.")
    parser.add_argument("--upscale", type=float, default=0.4, help="Upscale factor for GFPGAN face enhancement.")
    args = parser.parse_args()

    print("Initializing wrapper...")
    wrapper = Wrapper(
        source_image=args.source_image,
        gfpgan_path=args.gfpgan_path,
        inswapper_path=args.inswapper_path,
        upscale=args.upscale,
    )

    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind(ZMQ_RECEIVE_ADDRESS)
    sender = context.socket(zmq.PUSH)
    sender.bind(ZMQ_SEND_ADDRESS)
    update_socket = context.socket(zmq.REP)
    update_socket.bind(ZMQ_UPDATE_ADDRESS)

    poller = zmq.Poller()
    poller.register(receiver, zmq.POLLIN)
    poller.register(update_socket, zmq.POLLIN)

    print("Server is running and waiting for frames and update commands...")

    try:
        while True:
            socks = dict(poller.poll(timeout=50))
            
            if update_socket in socks and socks[update_socket] == zmq.POLLIN:
                message = update_socket.recv()
                try:
                    update_data = json.loads(message.decode("utf-8"))
                    new_image_path = update_data.get("source_image")
                    new_upscale = update_data.get("upscale")
                    if new_image_path and new_upscale is not None:
                        print("Received update command:")
                        print(f" - New source image: {new_image_path}")
                        print(f" - New upscale factor: {new_upscale}")
                        wrapper.update_config(new_image_path, float(new_upscale))
                        update_socket.send_string("Update successful")
                    else:
                        update_socket.send_string("Invalid update command")
                except Exception as e:
                    print(f"Error processing update command: {e}")
                    update_socket.send_string("Error in update command")
            
            if receiver in socks and socks[receiver] == zmq.POLLIN:
                start_time = time.time()
                data = receiver.recv()
                compressed_frame = np.frombuffer(msgpack.unpackb(data), dtype=np.uint8)
                frame = cv2.imdecode(compressed_frame, cv2.IMREAD_REDUCED_COLOR_2)

                processed_frame = process_frame(frame, wrapper)
                processed_frame = cv2.resize(
                    processed_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC
                )

                _, encoded_frame = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                sender.send(msgpack.packb(encoded_frame.tobytes()))
                elapsed_time = time.time() - start_time
                print(f"Processing time: {elapsed_time:.4f} seconds")

    except KeyboardInterrupt:
        print("Server interrupted.")

if __name__ == "__main__":
    main()
