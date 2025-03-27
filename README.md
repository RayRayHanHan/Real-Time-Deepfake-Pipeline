# Real-Time Deepfake Voice & Video Conversion

## Goal

- Use state-of-the-art audio and video deepfake models and build a real-time pipeline for a live video call.
- Focus on choosing efficient models that ensure high-quality, low-latency performance for both audio and video deepfakes during live interactions.
- Create a pipeline to automate the process, with an user interface (UI) that integrates with video calling platforms such as Skype or other alternatives to facilitate real-time deepfake interaction.


This repository provides a complete real-time deepfake system for both voice and video conversion. It consists of a **server-client architecture** where:

- **Audio System:**  
  - The **client** captures your microphone input, sends it to the server for real-time conversion, and plays the modified voice during live calls (e.g., Skype, VB-Audio).
  - The **server** performs voice conversion using **Diff-HierVC**.
  
- **Video System:**  
  - The **client** captures webcam video, sends frames to the server, and returns modified frames to an OBS virtual camera for real-time deepfake video applications.
  - The **server** performs face detection and face swapping using **insightface** and perofrms face enhancing using **GFPGAN**.

> **Note:** The pre-trained models are too large for GitHub. Download them manually using the links below and place them into the specified directories.

---

## Audio System 🎤

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
   ├── inference.py
   ├── infer.sh
   ├── vocoder
   │   ├── voc_hifigan.pth ✅
   │   └── voc_bigvgan.pth ✅
   ```

3. **Start the Audio Server:**
   ```bash
   python inference.py
   ```

4. **Run the Audio Client** (for real-time voice conversion):
   ```bash
   python GUI-Client.py
   ```
   
   The client has a **GUI**, allowing you to select a different **source audio** for conversion.

## Video System 🎥

### Setup

1. **Clone Real-Time-Deepfake-Pipeline Repository & Install Dependencies:**
   ```bash
   git clone https://github.com/ali-shariaty/Real-Time-Deepfake-Pipeline.git
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

4. **Run the Video Client:**
   ```bash
   python GUI-Client.py
   ```
   
   The client also has a **GUI**, allowing you to select a different **source image** for face-swapping, different **upscale factor** for face enhancing or **disabling** (or enabling) the **face enhancement**.

5. **Set Up Virtual Camera in OBS**
   - Open **OBS Studio**.
   - Add your webcam as a source by going to `Sources -> Video Capture Device` and selecting your webcam.
   - Click Start Virtual Camera in the `Controls` panel.
   - Close **OBS Studio** after starting the virtual camera. This is necessary to avoid conflicts, as the webcam might otherwise be occupied by OBS when your program is running.
   
   This will allow OBS’s virtual camera to be used by the client later.

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
     python inference.py
     ```

4. **Run the Client Locally:**  
   Open a new terminal on your local device and execute:
   ```bash
   python GUI-Client.py
   ```

The client will connect to the forwarded ports and communicate with the servers on the SSH server.

Note: Only the client (e.g. GUI-Client.py) needs to be executed on your local machine. All other files must be installed and run on the server.

---

## Usage 🛠️
- Start both **audio and video servers** (locally or on an SSH server with port forwarding as described).
- Run the **client scripts** to send data to the servers.
- Use **VB-Audio** for real-time voice conversion in Skype or other platforms.
- Use **OBS Virtual Camera** to route real-time deepfake video into your video calling software.
- Enjoy seamless deepfake interaction during live video calls.
---

## Credits & Contributors 🤝

This project was created and is maintained by **Ali Shariaty** and **Mert Arslan**.  
Feel free to reach out via email:

- alishariaty0854@gmail.com 
- mert.arslan517@gmail.com

---

## License 📝
This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Ali & Mert

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[...] (Full MIT License Text)
```

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

