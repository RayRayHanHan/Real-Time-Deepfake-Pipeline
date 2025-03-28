# Real-Time Deepfake Voice & Video Conversion

## Goal

This project aims to provide **real-time deepfake** capabilities for both **voice** and **video** in applications such as **Skype**, **Zoom**, and other video calling platforms. Using state-of-the-art audio and video deepfake models, this project builds a complete real-time pipeline that converts your voice and face on the fly. The system is designed with high-quality, low-latency performance in mind to ensure a seamless live interaction experience.

---

## Overview

This repository provides a complete real-time deepfake system for both voice and video conversion. It consists of a **server-client architecture** where:

- **Audio Server (Diff-HierVC):**  
  Runs a Flask-based service to perform real-time voice conversion.  
- **Video Server (insightface + GFPGAN):**  
  Processes webcam frames to perform face detection, swapping, and optional enhancement.

**A single GUI client** manages **both** audio and video connections. From this GUI, you can:
- **Connect/Disconnect** to the audio server.
- **Connect/Disconnect** to the video server.
- **Push-to-Talk** or continuous audio streaming for real-time voice conversion.
- **Configure** audio chunk size, select vocoder (BigVGAN or HiFiGAN), and upload/update target voice references.
- **Upload** and **update** the source image for face swapping on the video server.
- **Start/Stop** the virtual camera stream for real-time face swapping.
- **Adjust** upscale factors and face enhancement settings.

Below is a screenshot of the **single GUI** client:

<img src="https://private-user-images.githubusercontent.com/89353884/427796857-ddd41566-f341-4080-a1a7-4dbefc49b914.PNG?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDMxMjI4MzAsIm5iZiI6MTc0MzEyMjUzMCwicGF0aCI6Ii84OTM1Mzg4NC80Mjc3OTY4NTctZGRkNDE1NjYtZjM0MS00MDgwLWExYTctNGRiZWZjNDliOTE0LlBORz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTAzMjglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwMzI4VDAwNDIxMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM0MDc5YmQ4N2JkYWVlMzFlZWEzNzgzZmZiODEwZjA3MWNkNWIxMDQ1NDNhZGU5NDg0NzhkZGI3Y2VjNzkwYmEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.0b2eSeLz68fmq0sXOO3t5aoAOTdAahca65KJRaRh_84" alt="GUI_Client" width="600">


> **Note:** The pre-trained models are too large for GitHub. Download them manually using the links below and place them into the specified directories.

---

## Audio Server Ssetup 🎤

### Setup

1. **Clone the Repository & Install Dependencies:**
   ```bash
   git clone https://github.com/ali-shariaty/Real-Time-Deepfake-Pipeline.git
   cd Real-Time-Deepfake-Pipeline/audio
   pip install -r requirements.txt
   ```

2. **Download Pre-trained Models** and place them in the following directory:
   - [Diff-HierVC Models](https://drive.google.com/drive/folders/1THkeyDlA7EbZxwnuuxGsUOftV70Fb7h4?usp=sharing)
   
   ```
   .
   ├── ckpt
   │   ├── config.json
   │   └── model_diffhier.pth ✅
   ├── server.py
   ├── infer.sh
   ├── vocoder
   │   ├── voc_hifigan.pth ✅
   │   └── voc_bigvgan.pth ✅
   ```

3. **Start the Audio Server:**
   ```bash
   python server.py
   ```
   By default, the server listens on port **5003**.

4. **VB-Audio Virtual Cable**  
   - Install [VB-CABLE](https://vb-audio.com/Cable/index.htm).  
   - In **Windows Sound Settings**, set **`CABLE Output`** as the default microphone.  
   - In the calling app (Skype/Zoom), select **`CABLE Output`** as the microphone.  
   - The GUI client will play converted audio to your default output device, which VB-CABLE routes into the calling app.

---

## Video Server Setup 🎥

### Setup

1. **Install Dependencies:**
   ```bash
   cd Real-Time-Deepfake-Pipeline/video
   pip install -r requirements.txt
   ```

2. **Download Pre-trained Models and Place Them in the Correct Directory:**
   - [Face Swapping Model](https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx?download=true)
   - [GFPGAN Model v1.3](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth) or/and [GFPGAN Model v1.4](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth)
   - Move them into `Real-Time-Deepfake-Pipeline/video/models/`

3. **Start the Video Server:**
   ```bash
   python server.py
   ```

   With ```--help``` you can see the available options which can be modified for the video server. The options **source image**, **upscale factor** and **disable face enhancement toggle** can also be modified at runtime via the GUI-Client.


4. **Set Up Virtual Camera in OBS**
   - Open **OBS Studio**.
   - Add your webcam as a source by going to `Sources -> Video Capture Device` and selecting your webcam.
   - Click Start Virtual Camera in the `Controls` panel.
   - Close **OBS Studio** after starting the virtual camera. This is necessary to avoid conflicts, as the webcam might otherwise be occupied by OBS when your program is running.
   
   This will allow OBS’s virtual camera to be used by the client later.

---


## Running the Single GUI Client

After both servers are running (locally or remotely), **navigate to the root directory** of the project and run:

```bash
python GUI-Client.py
```

The GUI includes:
- **Server Connection** section to connect to Audio/Video servers.
- **Audio Controls** to choose your audio device, push-to-talk, chunk size, vocoder type, etc.
- **Target Audio Upload** to select and upload new reference audio for voice conversion.
- **Video Controls** to start/stop video streaming, configure the source image for face swapping, and upscale factors.
- **Virtual Camera Status** to indicate whether the OBS virtual camera is active.

---

## SSH Server Compatibility 🔧

If you want to run the server and client on different devices (e.g. a remote SSH server for the server side and your local machine for the client), follow these steps:

1. **Install and Configure the Server:**  
   Ensure that all server components (both audio and video) are installed and configured on your SSH server.

2. **Set Up SSH Port Forwarding:**  
   Open separate terminal windows for port forwarding:

   - **For Video:**  
     ```bash
     ssh -L 5558:localhost:5558 -L 5559:localhost:5559 -L 5560:localhost:5560 -p <SSH_PORT> <USERNAME>@<IP>
     ```
   - **For Audio:**  
     ```bash
     ssh -L 5003:localhost:5003 -p <SSH_PORT> <USERNAME>@<IP>
     ```

   Adjust the port numbers, username, and IP address as needed.

3. **Start the Servers on the SSH Server:**  
   - In the video terminal, run:
     ```bash
     python server.py
     ```
   - In the audio terminal, run:
     ```bash
     python server.py
     ```

4. **GUI File Upload Functions**

For functions such as **upload_target_audio** and **upload_video_source**:
- **Private Key Path (if required):**  
  If your SSH server uses key-based authentication, replace the placeholder (e.g., `your private Key for SSH`) with the path to your actual private key.
- **SSH Port:**  
  Replace the placeholder port value with your actual SSH port.
- **Remote Destination:**  
  Update the remote path to include the proper username and target directory on your SSH server.

5. **Server Connection Command**

For the function that establishes an SSH connection (e.g., **connect_ssh_for_video**):
- **SSH Credentials:**  
  Modify the SSH command to use your own private key (if required), port, username, and IP address.
- **Server Path:**  
  Ensure that the command navigates to the correct directory on your SSH server. Replace any example usernames (like “ali”) with your actual SSH user ID.

6. **Additional Configurations**

If you need to adjust parameters such as target image, chunk size, or other settings in the GUI:
- **Function Parameters:**  
  Make sure the functions handling these parameters are updated with your SSH credentials (port, username, IP) and any necessary file paths or configuration values.
- **Modular Approach:**  
  It is recommended to centralize these settings so they can be easily changed across the application.


7. **Run the Client Locally:**  
   Open a new terminal on your local device and execute:
   ```bash
   python GUI-Client.py
   ```

The client will connect to the forwarded ports and communicate with the servers on the SSH server.

Note: Only the client (e.g. GUI-Client.py) needs to be executed on your local machine. All other files must be installed and run on the server.

---

## Usage Workflow 🛠️

1. **Server Side**  
   - Launch the audio server (`server.py` in `audio/`).  
   - Launch the video server (`server.py` in `video/`).

2. **Client Side**  
   - Run `GUI-Client.py` from the project root.  
   - Configure the **Audio Server** and **Video Server** addresses (local or remote via SSH forwarding).  
   - In the GUI, adjust audio device, chunk size, vocoder, etc.  
   - For video, upload or select a source image, set the upscale factor, and start the video stream.

3. **Virtual Devices**  
   - **VB-CABLE** for audio input.  
   - **OBS Virtual Camera** for video output.

4. **Join a Call**  
   - In Skype/Zoom, select **`CABLE Output`** as your microphone.  
   - In Skype/Zoom, select **`OBS Virtual Camera`** as your webcam.  
   - Experience real-time deepfake voice and video.

---

## Recommended Settings

1. **Chunk Size (Audio)**  
   - Default: `16000` samples.  
   - Decreasing chunk size (e.g., `8000` or `4000`) **reduces latency** but may lower audio quality.  
   - Increasing chunk size **improves audio quality** but adds more delay.  
   - This project also uses **audio smoothing** to reduce audible artifacts between chunks. See:
     ```python
     def smooth_audio_transition(prev_chunk, current_chunk, overlap_ratio=0.15): ...
     def play_audio_with_smoothing(data, buffer_size=3072, overlap_ratio=0.15): ...
     ```
     - **`buffer_size`** (e.g., `3072`) determines how many samples are processed at a time during playback.
     - **`overlap_ratio`** (e.g., `0.15`) controls how much of the chunk overlaps with the previous chunk.  
       - A **larger overlap_ratio** can provide smoother transitions (fewer artifacts), but requires more compute and can introduce slight additional delay.
       - A **smaller overlap_ratio** is faster but may cause more abrupt transitions between chunks.

2. **Vocoder**  
   - **BigVGAN**: Higher audio quality, slightly slower inference.  
   - **HiFiGAN**: Faster inference, slightly lower fidelity.

3. **Diffusion Steps (Audio)**  
   In the code, you may see:
   ```python
   def get_adaptive_diff_params(audio_length):
       return 15, 15
   ```
   which returns two parameters: `diffpitch_ts` and `diffvoice_ts`.  
   - **`diffpitch_ts`**: Number of diffusion steps specifically for pitch. Higher values can lead to smoother pitch transformations but increase compute time.  
   - **`diffvoice_ts`**: Number of diffusion steps for the voice timbre. Higher values can yield more accurate timbre changes, at the cost of extra latency.

4. **Upscale Factor (Video)**  
   - Default: `0.4`.  
   - Higher values may produce sharper faces (when GFPGAN enhancement is enabled) but increase computation time.

---

## Credits & Contributors 🤝

This project was created and is maintained by **Ali Shariaty** and **Mert Arslan**.  
Feel free to reach out via email:

- alishariaty0854@gmail.com 
- mert.arslan517@gmail.com

---

## License 📝
This project is licensed under the **MIT License**.


---

## Citation 🎓
If you use this work, please cite the following paper:
```
@inproceedings{choi23d_interspeech,
  author={Ha-Yeong Choi and Sang-Hoon Lee and Seong-Whan Lee},
  title={{Diff-HierVC: Diffusion-based Hierarchical Voice Conversion with Robust Pitch Generation and Masked Prior for Zero-shot Speaker Adaptation}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={2283--2287},
  doi={10.21437/Interspeech.2023-817}
}
```

## Acknowledgements 💎
- This project is based on **[Diff-HierVC](https://github.com/hayeong0/Diff-HierVC)**, **[HiFiGAN](https://github.com/jik876/hifi-gan)**, **[BigVGAN](https://github.com/NVIDIA/BigVGAN)**, **[Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam)**, **[insightface](https://github.com/deepinsight/insightface)** and **[GFPGAN](https://github.com/TencentARC/GFPGAN)**.

---

🎧 **Enjoy real-time deepfake voice and video conversion!** 🚀

