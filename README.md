# Real-Time Deepfake Voice & Video Conversion

This repository provides a complete real-time deepfake system for both voice and video conversion. It consists of a **server-client architecture** where:

- **Audio System:**  
  - The **server** performs voice conversion using **Diff-HierVC**.
  - The **client** captures your microphone input, sends it to the server for real-time conversion, and plays the modified voice during live calls (e.g., Skype, VB-Audio).
  
- **Video System:**  
  - The **server** performs face swapping using **Deep-Live-Cam**.
  - The **client** captures webcam video, sends frames to the server, and returns modified frames to an OBS virtual camera for real-time deepfake video applications.

> **Note:** The pre-trained models are too large for GitHub. Download them manually using the links below and place them into the specified directories.

---

## Audio System 🎤

### Installation

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

### Installation

1. **Clone Real-Time-Deepfake-Pipeline Repository & Install Dependencies:**
   ```bash
   git clone https://github.com/ali-shariaty/Real-Time-Deepfake-Pipeline.git
   cd Real-Time-Deepfake-Pipeline/video
   pip install -r requirements.txt
   ```

2. **Download Pre-trained Models and Place Them in the Correct Directory:**
   - [Face Swapping Model](https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx?download=true)
   - [GFPGAN Model](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth)
   - Move them into `Real-Time-Deepfake-Pipeline/video/models/`

3. **Start the Video Server:**
   ```bash
   python server.py
   ```

4. **Run the Video Client:**
   ```bash
   python GUI-Client.py
   ```
   
   The client also has a **GUI**, allowing you to select a different **source image** for face-swapping.

5. **Set Up Virtual Camera in OBS**
   - Open **OBS Studio** and go to `Settings > Video`.
   - Add a **Virtual Camera**.

---

## SSH Server Compatibility 🔧
This project has been tested using an **SSH server**.


---

## Usage 🛠️
- Start both **audio and video servers**.
- Run the **client scripts** to send data to the servers.
- Use **VB-Audio** for real-time voice conversion in Skype or other platforms.
- Use **OBS Virtual Camera** for real-time video deepfake.

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
- This project is based on **[Diff-HierVC](https://github.com/hayeong0/Diff-HierVC)**, **[HiFiGAN](https://github.com/jik876/hifi-gan)**, **[BigVGAN](https://github.com/NVIDIA/BigVGAN)**, and **[Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam)**.

---

🎧 **Enjoy real-time deepfake voice and video conversion!** 🚀

