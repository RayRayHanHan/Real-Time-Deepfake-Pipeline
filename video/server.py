import cv2
import zmq
import msgpack
import msgpack_numpy as m
import numpy as np
from wrapper import Wrapper
import time
import argparse


m.patch()

ZMQ_RECEIVE_ADDRESS = "tcp://0.0.0.0:5558"
ZMQ_SEND_ADDRESS = "tcp://0.0.0.0:5559"


# Process received webcam frame
def process_frame(frame, wrapper):
    # Resize frame
    frame_resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
    processed_frame = wrapper.generate(frame_resized)
    return processed_frame


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Real-Time-Deepfake-Pipeline Server Options"
    )
    parser.add_argument(
        "--source_image",  # Replace with file path of source image
        type=str,
        default="./image.jpg",
        help="Path to the source image.",
    )
    parser.add_argument(
        "--gfpgan_path",
        type=str,
        default="models/GFPGANv1.3.pth",
        help="Path to the GFPGAN model file.",
    )
    parser.add_argument(
        "--inswapper_path",
        type=str,
        default="models/inswapper_128_fp16.onnx",
        help="Path to the inswapper model file.",
    )
    parser.add_argument(
        "--upscale",
        type=float,
        default=0.4,
        help="Upscale factor for GFPGAN face enhancement.",
    )
    args = parser.parse_args()

    # Initialize wrapper
    print("Initializing wrapper...")
    wrapper = Wrapper(
        source_image=args.source_image,
        gfpgan_path=args.gfpgan_path,
        inswapper_path=args.inswapper_path,
        upscale=args.upscale,
    )

    context = zmq.Context()

    # Socket to receive frames from the client
    receiver = context.socket(zmq.PULL)
    receiver.bind(ZMQ_RECEIVE_ADDRESS)

    # Socket to send processed frames back to the client
    sender = context.socket(zmq.PUSH)
    sender.bind(ZMQ_SEND_ADDRESS)

    print("Server is running and waiting for frames...")

    try:
        while True:
            # Receive frame
            start_time = time.time()
            data = receiver.recv()
            compressed_frame = np.frombuffer(msgpack.unpackb(data), dtype=np.uint8)
            frame = cv2.imdecode(compressed_frame, cv2.IMREAD_REDUCED_COLOR_2)

            # Process frame
            processed_frame = process_frame(frame, wrapper)

            # Resize back to original resolution if needed (before sending)
            processed_frame = cv2.resize(
                processed_frame,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

            # Compress and send processed frame
            _, encoded_frame = cv2.imencode(
                ".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            sender.send(msgpack.packb(encoded_frame.tobytes()))

            elapsed_time = time.time() - start_time
            print(f"Processing time: {elapsed_time:.4f} seconds")

    except KeyboardInterrupt:
        print("Server interrupted.")


if __name__ == "__main__":
    main()
