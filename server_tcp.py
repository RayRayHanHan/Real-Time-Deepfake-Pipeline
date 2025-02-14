import pickle
from wrapper import Wrapper
import socket
import struct
import numpy as np
import cv2


# Helper function receive n bytes
def receive_msg(sock, expected_size):
    data = bytearray()
    # Consider cases where we could read fewer bytes than expected_size
    while len(data) < expected_size:
        packet = sock.recv(expected_size - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def decompress_frame(compressed_data):
    frame_array = np.frombuffer(compressed_data, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    return frame


# Process received webcam frame
def process_frame(frame, wrapper):
    processed_frame = wrapper.generate(frame)
    return processed_frame


def main():
    # Initialize wrapper
    source_path = "./image.jpg"  # Replace with file path of source image
    print("Initializing wrapper...")
    wrapper = Wrapper(source_path)

    # Create TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = "0.0.0.0"
    port = 8080

    # Bind socket to port
    server_socket.bind((host, port))

    # Listen for incoming connections
    server_socket.listen(1)
    print(f"Server listening on {host}:{port}")

    while True:
        print("Waiting for connection...")
        client_socket, client_address = server_socket.accept()
        print(f"Connected to client {client_address}")

        try:
            while True:
                # Receive frame size
                size_data = receive_msg(client_socket, 4)
                if size_data is None:
                    print("Client disconnected")
                    break
                frame_size = struct.unpack(">L", size_data)[0]

                # Receive frame data
                frame_data = receive_msg(client_socket, frame_size)
                if frame_data is None:
                    print("Client disconnected")
                    break

                # Deserialize compressed frame
                compressed_frame = pickle.loads(frame_data)

                # Decompress frame data to numpy array
                frame = decompress_frame(compressed_frame)

                # Process frame
                processed_frame = process_frame(frame, wrapper)

                # Serialize processed frame
                processed_frame_data = pickle.dumps(processed_frame)

                # Send processed frame size
                client_socket.sendall(struct.pack(">L", len(processed_frame_data)))

                # Send processed frame
                client_socket.sendall(processed_frame_data)

        except (ConnectionResetError, BrokenPipeError):
            print("Client disconnected")
        finally:
            client_socket.close()


if __name__ == "__main__":
    main()
