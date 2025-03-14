import torch
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import insightface
from PIL import Image
from modules import face_analyser
from modules.processors.frame import face_enhancer, face_swapper
import time
import os

torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# Wrapper class for face detection & analysis (via InsightFace), face swapping (inswapper_128.onnx model from InsightFace) and face enhancement (via GFPGAN)
class Wrapper:

    # Initialize models
    def __init__(
        self,
        source_path,
        gfpgan_path="models/GFPGANv1.4.pth",
        inswapper_path="models/inswapper_128_fp16.onnx",
    ):
        # Load face analyzer model (InsightFace, buffalo_l)
        self.face_analyzer = FaceAnalysis(
            name="buffalo_l",
            providers=[
                "CUDAExecutionProvider",
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ],
            provider_options=[{"device_id": 0}],
        )
        self.face_analyzer.prepare(
            ctx_id=0, det_size=(640, 640), det_thresh=0.5
        )  # change to 640, 360?

        # Load face swapping model (InsightFace, inswapper)
        self.face_swapper = insightface.model_zoo.get_model(
            inswapper_path,
            providers=[
                "CUDAExecutionProvider",
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ],
            provider_options=[{"device_id": 0}],
        )

        # Load GFPGAN model for face enhancement
        self.face_enhancer = self.load_model(gfpgan_path)

        # Path to source face image for face-swap
        self.source_path = source_path

        # Source face: Load and extract the face from the source image
        self.source_face = face_analyser.get_one_face(cv2.imread(self.source_path))

    # Load a model
    def load_model(self, path):
        if torch.cuda.is_available():
            # Use CUDA if NVIDIA GPU is available
            device = "cuda"
            torch.cuda.set_device(0)
        elif torch.backends.mps.is_available():
            # Use MPS for Mac M1/M2 GPU acceleration
            device = "mps"
        else:
            # Fallback to CPU
            device = "cpu"

        print(f"Loading model on: {device.upper()}")
        model = torch.load(path, map_location=device)
        return model

    # Processes input frame (Face detection, Face swapping, Face enhancing)
    def generate(self, frame):
        # Target face: Detect a face in the current (webcam) frame
        start_time = time.time()
        target_face = face_analyser.get_one_face(frame)
        elapsed_time = time.time() - start_time
        print(f"1. Face detection: {elapsed_time:.4f} seconds")

        # Check if faces detected. If yes, perform face swapping. If no, use original frame
        start_time = time.time()
        if self.source_face and target_face:
            # Perform face swapping
            tmp_frame = face_swapper.swap_face(self.source_face, target_face, frame)
        else:
            tmp_frame = frame
        elapsed_time = time.time() - start_time
        print(f"2. Face swapper: {elapsed_time:.4f} seconds")

        # Perform face enhancing
        start_time = time.time()
        processed_frame = face_enhancer.process_frame(None, tmp_frame)
        elapsed_time = time.time() - start_time
        print(f"3. Face enhancer: {elapsed_time:.4f} seconds")

        # Convert to NumPy array if it is a PIL image
        if isinstance(processed_frame, Image.Image):
            processed_frame = np.array(processed_frame)

        # Ensure frame is valid before displaying
        if processed_frame is not None and isinstance(processed_frame, np.ndarray):
            return processed_frame
        else:
            print("Error: Processed frame is invalid.")
            # Return original frame if processing fails
            return frame
