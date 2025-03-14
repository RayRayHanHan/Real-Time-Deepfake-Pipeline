import cv2
import zmq
import msgpack
import msgpack_numpy as m
import numpy as np
from wrapper import Wrapper
import time


m.patch()

ZMQ_RECEIVE_ADDRESS = "tcp://0.0.0.0:8888"
ZMQ_SEND_ADDRESS = "tcp://0.0.0.0:8889"


# Process received webcam frame
def process_frame(frame, wrapper):
    # Resize frame
    frame_resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
    processed_frame = wrapper.generate(frame_resized)
    return processed_frame


def main():
    # Initialize wrapper
    source_path = "./image.jpg"  # Replace with file path of source image
    print("Initializing wrapper...")
    wrapper = Wrapper(source_path)

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
