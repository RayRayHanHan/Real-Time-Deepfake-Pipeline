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


class Wrapper:
    def __init__(
        self,
        source_image="./image.jpg",
        gfpgan_path="models/GFPGANv1.3.pth",
        inswapper_path="models/inswapper_128_fp16.onnx",
        upscale=0.4,
        disable_face_enhancement=False,
    ):
        self.face_analyzer = FaceAnalysis(
            name="buffalo_l",
            providers=[
                "CUDAExecutionProvider",
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ],
            provider_options=[{"device_id": 0}, {"device_id": 0}, {"device_id": 0}],
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

        self.face_swapper = insightface.model_zoo.get_model(
            inswapper_path,
            providers=[
                "CUDAExecutionProvider",
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ],
            provider_options=[{"device_id": 0}, {"device_id": 0}, {"device_id": 0}],
        )
        face_swapper.INSWAPPER_PATH = inswapper_path

        self.disable_face_enhancement = disable_face_enhancement
        if not self.disable_face_enhancement:
            self.face_enhancer = self.load_model(gfpgan_path)
            face_enhancer.FACE_ENHANCER_UPSCALE = upscale
            face_enhancer.GFPGAN_PATH = gfpgan_path

        self.source_face = face_analyser.get_one_face(cv2.imread(source_image))

    def load_model(self, path):
        if torch.cuda.is_available():
            device = "cuda"
            torch.cuda.set_device(0)
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        print(f"Loading model on: {device.upper()}")
        model = torch.load(path, map_location=device)
        return model

    def update_config(self, new_image_path, new_upscale, new_disable_face):
        print("Updating configuration...")
        new_image = cv2.imread(new_image_path)
        if new_image is not None:
            self.source_face = face_analyser.get_one_face(new_image)
            print("Source face updated.")
        else:
            print("Error: Unable to load new image.")

        face_enhancer.FACE_ENHANCER_UPSCALE = new_upscale
        print(f"Upscale factor updated to {new_upscale}")
        self.disable_face_enhancement = new_disable_face
        print(f"Disable face enhancement value updated to {new_disable_face}")

    def generate(self, frame):
        start_time = time.time()
        target_face = face_analyser.get_one_face(frame)
        elapsed_time = time.time() - start_time
        print(f"1. Face detection: {elapsed_time:.4f} seconds")

        start_time = time.time()
        if self.source_face and target_face:
            tmp_frame = face_swapper.swap_face(self.source_face, target_face, frame)
        else:
            tmp_frame = frame
        elapsed_time = time.time() - start_time
        print(f"2. Face swapper: {elapsed_time:.4f} seconds")

        print(self.disable_face_enhancement)
        if not self.disable_face_enhancement:
            start_time = time.time()
            processed_frame = face_enhancer.process_frame(None, tmp_frame)
            elapsed_time = time.time() - start_time
            print(f"3. Face enhancer: {elapsed_time:.4f} seconds")
        else:
            processed_frame = tmp_frame

        if isinstance(processed_frame, Image.Image):
            processed_frame = np.array(processed_frame)

        if processed_frame is not None and isinstance(processed_frame, np.ndarray):
            return processed_frame
        else:
            print("Error: Processed frame is invalid.")
            return frame
