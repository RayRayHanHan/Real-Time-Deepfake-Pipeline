import pickle
from wrapper import Wrapper
import socket
import numpy as np
import cv2


# Helper function to receive a message (UDP version - receives max buffer size)
def receive_msg_udp(sock, buffer_size):
    try:
        data, addr = sock.recvfrom(buffer_size)
        return data, addr
    # Handle timeout if no data received
    except socket.timeout:
        return None, None


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

    # Create UDP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    host = "0.0.0.0"
    port = 8080
    server_address = (host, port)

    # Bind socket to port
    server_socket.bind(server_address)
    print(f"UDP Server listening on {host}:{port}")

    buffer_size = 65535  # Max UDP packet size

    while True:
        print("Waiting for frame...")
        # Receive data and client address
        compressed_frame_data, client_address = receive_msg_udp(
            server_socket, buffer_size
        )
        # Check if data was received
        if compressed_frame_data:
            try:
                # Deserialize compressed frame (which is compressed bytes data)
                compressed_frame = pickle.loads(compressed_frame_data)

                # Decompress frame data to numpy array
                frame = decompress_frame(compressed_frame)

                # Process frame
                processed_frame = process_frame(frame, wrapper)

                # Serialize processed frame
                processed_frame_data = pickle.dumps(processed_frame)

                # Send processed frame back to client address
                server_socket.sendto(processed_frame_data, client_address)

            except Exception as e:
                import traceback

                print(f"Server-side processing error: {e}")
                traceback.print_exc()
        else:
            # Indicate timeout or no data
            print("No frame received or timeout")


if __name__ == "__main__":
    main()
