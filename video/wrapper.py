import torch
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import insightface
from PIL import Image
from modules import face_analyser
from modules.processors.frame import face_enhancer, face_swapper


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
        )
        self.face_analyzer.prepare(ctx_id=0)

        # Load face swapping model (InsightFace, inswapper)
        self.face_swapper = insightface.model_zoo.get_model(
            inswapper_path,
            providers=[
                "CUDAExecutionProvider",
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

        # Load GFPGAN model for face enhancement
        self.face_enhancer = self.load_model(gfpgan_path)

        # Path to source face image for face-swap
        self.source_path = source_path

    # Load a model
    def load_model(self, path):
        if torch.cuda.is_available():
            # Use CUDA if NVIDIA GPU is available
            device = "cuda"
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
        # Source face: Load and extract the face from the source image
        source_face = face_analyser.get_one_face(cv2.imread(self.source_path))

        # Target face: Detect a face in the current (webcam) frame
        target_face = face_analyser.get_one_face(frame)

        # Check if faces detected. If yes, perform face swapping. If no, use original frame
        if source_face and target_face:
            # Perform face swapping
            tmp_frame = face_swapper.swap_face(source_face, target_face, frame)
        else:
            tmp_frame = frame

        # Perform face enhancing
        processed_frame = face_enhancer.process_frame(None, tmp_frame)

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
