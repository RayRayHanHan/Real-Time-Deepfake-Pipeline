import cv2
import pickle
import socket

# from sshtunnel import SSHTunnelForwarder

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
    frame = cv2.resize(frame, (320, 240))

    # Compress using JPEG encoding
    _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 30])
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

            optimized_frame_bytes = optimize_frame(frame)
            print(
                f"Original size: {len(frame.tobytes())}, Compressed: {len(optimized_frame_bytes)}"
            )
            yield optimized_frame_bytes  # Yield compressed bytes

            # Press 'q' to exit the loop
            if cv2.waitKey(1) == ord("q"):
                print("Exiting...")
                break
    finally:
        cam.release()


# Helper function to receive a message
def receive_msg_udp(sock, buffer_size):
    try:
        # Receive data and server address
        data, server_address = sock.recvfrom(buffer_size)
        return data
    # Handle timeout if no data received
    except socket.timeout:
        return None


def main():
    # SSH connection details (Optional for UDP in local network)
    # SSH_HOST = "" # replace with hostname/IP address
    # SSH_PORT = 1234 # replace with SSH port
    # SSH_USERNAME = ""

    # Create SSH tunnel (Optional for UDP in local network)
    # tunnel = create_ssh_tunnel(SSH_HOST, SSH_PORT, SSH_USERNAME, REMOTE_PORT)
    # if not tunnel:
    #     print("Exiting due to SSH tunnel craeation failure")
    #     return
    # time.sleep(1)

    # UDP server details
    UDP_SERVER_IP = "localhost"
    UDP_SERVER_PORT = 8080

    # Create UDP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (UDP_SERVER_IP, UDP_SERVER_PORT)
    buffer_size = 65535  # Max UDP packet size
    # Optional timeout for receiving data. If server doesn't respond within  1 second, client will continue
    # client_socket.settimeout(
    #     1.0
    # )

    try:
        # Capture webcam frames
        for frame in get_webcam_frames():

            # Serialize frame bytes
            frame_data = pickle.dumps(frame)

            # Send frame data (UDP packet)
            client_socket.sendto(frame_data, server_address)

            # Receive processed frame (UDP packet)
            processed_frame_data = receive_msg_udp(client_socket, buffer_size)
            # Check if data received
            if processed_frame_data:
                # Deserialize and display processed frame
                print("Received processed frame...")
                processed_frame = pickle.loads(processed_frame_data)
                cv2.imshow("Deep Live Cam", processed_frame)
            else:
                # Indicate timeout or no response
                print("No processed frame received from server or timeout")

            if cv2.waitKey(1) == ord("q"):
                break

    except ConnectionRefusedError:
        print("Couldn't connect to server (UDP)")
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        client_socket.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
