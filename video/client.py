import cv2
import zmq
import msgpack
import msgpack_numpy as m
import numpy as np
import time

m.patch()

ZMQ_SERVER_ADDRESS = "tcp://localhost:5555"
ZMQ_CLIENT_ADDRESS = "tcp://localhost:5556"


def main():
    context = zmq.Context()

    # Socket to send frames to the server
    sender = context.socket(zmq.PUSH)
    sender.connect(ZMQ_SERVER_ADDRESS)

    # Socket to receive processed frames from the server
    receiver = context.socket(zmq.PULL)
    receiver.connect(ZMQ_CLIENT_ADDRESS)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Client started, sending frames to the server...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Compress and send the frame
            _, encoded_frame = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            sender.send(msgpack.packb(encoded_frame.tobytes()))
            print("Sent frame to server")

            # Receive processed frame from the server
            start_time = time.time()
            data = receiver.recv()
            print("Received frame from server")
            elapsed_time = time.time() - start_time
            print(f"Receive time: {elapsed_time:.4f} seconds")
            processed_frame_data = np.frombuffer(msgpack.unpackb(data), dtype=np.uint8)
            processed_frame = cv2.imdecode(
                processed_frame_data, cv2.IMWRITE_JPEG_QUALITY
            )

            # Show the processed frame
            cv2.imshow("Real-Time-Deepfake-Pipeline", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Client interrupted.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
