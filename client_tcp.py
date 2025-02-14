import cv2
import pickle
import socket
import struct

# from sshtunnel import SSHTunnelForwarder
import time

# Create a SSH tunnel
# def create_ssh_tunnel(ssh_host, ssh_port, ssh_username, remote_port):
#     try:
#         tunnel = SSHTunnelForwarder(
#             ssh_host,
#             ssh_port=ssh_port,
#             ssh_username=ssh_username,
#             remote_bind_address=("localhost", remote_port),
#             local_bind_address=("localhost", remote_port),
#         )
#         tunnel.start()
#         print("SSH tunnel established...")
#         return tunnel
#     except Exception as e:
#         print(f"Failed to establish a SSH tunnel: {e}")
#         return None


def optimize_frame(frame):
    # Resize to smaller resolution
    frame = cv2.resize(frame, (640, 480))

    # Compress using JPEG encoding
    _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    compressed_data = encoded.tobytes()

    return compressed_data


# Captures frames from the webcam
def get_webcam_frames():
    # Open webcam (default camera)
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while True:
            # Read a frame
            success, frame = cam.read()

            if not success:
                print("Error: Could not read frame.")
                break

            optimized_frame = optimize_frame(frame)
            print(
                f"Original size: {len(frame.tobytes())}, Compressed: {len(optimized_frame)}"
            )

            # Yield the captured frame
            yield optimized_frame

            # Press 'q' to exit the loop
            if cv2.waitKey(1) == ord("q"):
                print("Exiting...")
                break

    finally:
        cam.release()


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


def main():
    # SSH connection details
    SSH_HOST = ""  # replace with hostname/IP address
    SSH_PORT = 1234  # replace with SSH port
    # SSH_USERNAME = ""
    REMOTE_PORT = 8080

    # Create SSH tunnel
    # tunnel = create_ssh_tunnel(SSH_HOST, SSH_PORT, SSH_USERNAME, REMOTE_PORT)
    # if not tunnel:
    #     print("Exiting due to SSH tunnel craeation failure")
    #     return

    # time.sleep(1)

    # Connect to server through SSH tunnel
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        client_socket.connect(("localhost", REMOTE_PORT))
        print(f"Connected to server through SSH tunnel at {SSH_HOST}:{SSH_PORT}")

        # Capture webcam frames
        for frame in get_webcam_frames():
            # Serialize frame (convert python object to bytes)
            frame_data = pickle.dumps(frame)
            print(f"Len frame data sent: {len(frame_data)}")

            # Send frame size (4 bytes)
            client_socket.sendall(struct.pack(">L", len(frame_data)))

            # Send frame bytes (4 bytes)
            client_socket.sendall(frame_data)
            start_time = time.time()

            # Receive frame size (4 bytes)
            size_data = receive_msg(client_socket, 4)
            if size_data is None:
                print("Connection closed by server")
                break

            # Convert received frame bytes to python object
            processed_frame_size = struct.unpack(">L", size_data)[0]
            print(f"Receive frame size: {processed_frame_size}")

            # Receive processed frame bytes (4 bytes)
            processed_frame_data = receive_msg(client_socket, processed_frame_size)
            if processed_frame_data is None:
                print("Connection closed by server")
                break

            end_time = time.time()
            print(end_time - start_time)

            # Deserialize and display processed frame
            print("Received processed frame...")
            processed_frame = pickle.loads(processed_frame_data)
            cv2.imshow("Deep Live Cam", processed_frame)

            if cv2.waitKey(1) == ord("q"):
                break

    except ConnectionRefusedError:
        print("Couldn't connect to server")
    except Exception as e:
        print(f"Error during processing {e}")
    finally:
        client_socket.close()
        cv2.destroyAllWindows()
        # if tunnel:
        #     print("Closing SSH tunnel")
        #     tunnel.stop()


if __name__ == "__main__":
    main()
