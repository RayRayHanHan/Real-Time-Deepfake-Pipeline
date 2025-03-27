import os
import logging
import time
import torch
import numpy as np
from torch.nn import functional as F
import torchaudio
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
from vocoder.bigvgan import BigvGAN
from vocoder.hifigan import HiFi
from model.diffhiervc import DiffHierVC, Wav2vec2
from utils.utils import MelSpectrogramFixed
import utils.utils as utils
from flask import Flask, request, jsonify
import threading
import queue
from functools import lru_cache

try:
    from torch.cuda.amp import autocast, GradScaler

    amp_available = True
except ImportError:
    amp_available = False

app = Flask(__name__)

logging.basicConfig(
    filename='server_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
logging.info("Server started with optimized configuration.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    logging.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

seed = 1234
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)

current_vocoder_type = 'bigvgan'
noise_floor_dict = {
    'bigvgan': 0.005,
    'hifigan': 0.0003
}


@lru_cache(maxsize=10)
def precompute_target_features(audio_path):
    logging.info(f"Precomputing target audio features for: {audio_path}")
    target_waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        target_waveform = torchaudio.transforms.Resample(sr, 16000)(target_waveform)
    target_waveform = target_waveform.to(device).half()
    target_mel = mel_fn(target_waveform)
    target_length = torch.LongTensor([target_mel.size(-1)]).to(device)
    return target_mel, target_length


def load_models(vocoder_type='bigvgan'):
    global hps, model, net_v, w2v, mel_fn, target_mel, target_length, current_vocoder_type
    current_vocoder_type = vocoder_type
    config_path = './ckpt/config_bigvgan.json' if vocoder_type == 'bigvgan' else './ckpt/config.json'
    hps = utils.get_hparams_from_file(config_path)

    mel_fn = MelSpectrogramFixed(
        sample_rate=hps.data.sampling_rate,
        n_fft=hps.data.filter_length,
        win_length=hps.data.win_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_mels=hps.data.n_mel_channels,
        window_fn=torch.hann_window
    ).to(device)

    w2v = Wav2vec2().to(device).half()
    model = DiffHierVC(
        hps.data.n_mel_channels,
        hps.diffusion.spk_dim,
        hps.diffusion.dec_dim,
        hps.diffusion.beta_min,
        hps.diffusion.beta_max,
        hps
    ).to(device).half()

    model.load_state_dict(torch.load('./ckpt/model_diffhier.pth', map_location=device))

    try:
        model = torch.compile(model, mode='reduce-overhead')
    except Exception as e:
        logging.warning(f"Model compilation failed: {e}")

    model.eval()
    torch.set_grad_enabled(False)

    if vocoder_type == 'bigvgan':
        net_v = BigvGAN(
            hps.data.n_mel_channels,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model
        ).to(device).half()
        utils.load_checkpoint('./vocoder/voc_bigvgan.pth', net_v, None)
    else:  # hifigan
        net_v = HiFi(
            hps.data.n_mel_channels,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model
        ).to(device).half()
        utils.load_checkpoint('./vocoder/voc_hifigan.pth', net_v, None)

    net_v.eval().dec.remove_weight_norm()

    try:
        net_v = torch.compile(net_v, mode='reduce-overhead')
    except Exception as e:
        logging.warning(f"Vocoder compilation failed: {e}")

    current_target_audio_path = './sample/tar_p239_022.wav'
    target_mel, target_length = precompute_target_features(current_target_audio_path)
    logging.info(f"Models loaded successfully with {vocoder_type} vocoder.")


load_models('bigvgan')

f0_cache = {}


def get_yaapt_f0(audio, sr=16000, interp=False):
    audio_hash = hash(audio.tobytes())
    if audio_hash in f0_cache:
        return f0_cache[audio_hash]

    to_pad = int(20.0 / 1000 * sr) // 2
    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        pitch = pYAAPT.yaapt(basic.SignalObj(y_pad, sr),
                             **{'frame_length': 20.0, 'frame_space': 5.0,
                                'nccf_thresh1': 0.25, 'tda_frame_length': 25.0})
        f0s.append(pitch.samp_interp[None, None, :] if interp else pitch.samp_values[None, None, :])

    result = np.vstack(f0s)
    f0_cache[audio_hash] = result

    if len(f0_cache) > 100:
        f0_cache.pop(next(iter(f0_cache)))

    return result


def get_adaptive_diff_params(audio_length):
    return 15, 15


def process_audio(audio_chunk, sr=16000):
    start_total = time.time()

    times = {
        'preprocessing': 0,
        'inference': 0,
        'postprocessing': 0
    }

    try:
        start_preprocessing = time.time()
        audio_tensor = torch.from_numpy(audio_chunk.copy()).float().half().unsqueeze(0).to(device)
        p_val = (audio_tensor.shape[-1] // 1280 + 1) * 1280 - audio_tensor.shape[-1]
        audio_tensor = F.pad(audio_tensor, (0, p_val)).to(device)

        src_mel = mel_fn(audio_tensor)
        src_length = torch.LongTensor([src_mel.size(-1)]).to(device)

        w2v_x = w2v(F.pad(audio_tensor, (40, 40), "reflect"))
        times['preprocessing'] = time.time() - start_preprocessing

        start_f0 = time.time()
        try:
            f0 = get_yaapt_f0(audio_tensor.cpu().numpy(), sr)
        except Exception as e:
            logging.error(f"F0 computation error: {e}")
            f0 = np.zeros((1, audio_tensor.shape[-1] // 80), dtype=np.float32)

        f0_x = f0.copy()
        f0_x = torch.log(torch.FloatTensor(f0_x + 1)).half().to(device)

        ii = f0 != 0
        if np.any(ii):
            f0[ii] = (f0[ii] - f0[ii].mean()) / f0[ii].std()

        f0_norm_x = torch.FloatTensor(f0).half().to(device)

        audio_length_sec = audio_tensor.shape[-1] / sr
        diffpitch_ts, diffvoice_ts = get_adaptive_diff_params(audio_length_sec)

        start_inference = time.time()
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
            c = model.infer_vc(
                src_mel, w2v_x, f0_norm_x, f0_x, src_length,
                target_mel, target_length,
                diffpitch_ts=diffpitch_ts,
                diffvoice_ts=diffvoice_ts
            )
            converted_audio = net_v(c)

        times['inference'] = time.time() - start_inference

        start_postprocessing = time.time()
        converted_audio = converted_audio / torch.max(torch.abs(converted_audio))
        noise_floor = noise_floor_dict.get(current_vocoder_type, 0.001)
        logging.info(f"Applying noise floor: {noise_floor} for vocoder: {current_vocoder_type}")
        converted_audio = torch.where(
            torch.abs(converted_audio) < noise_floor,
            torch.zeros_like(converted_audio),
            converted_audio
        )
        times['postprocessing'] = time.time() - start_postprocessing

        total_time = time.time() - start_total

        logging.info(f"""
        Performance Breakdown:
        - Total Processing Time: {total_time:.3f}s
        - Preprocessing Time: {times['preprocessing']:.3f}s
        - F0 Computation Time: {time.time() - start_f0:.3f}s
        - Inference Time: {times['inference']:.3f}s
        - Postprocessing Time: {times['postprocessing']:.3f}s
        """)

        return converted_audio.squeeze().detach().cpu().numpy()

    except Exception as e:
        logging.error(f"Audio processing error: {e}")
        raise


PROCESSING_THREADS = 12
processing_queue = queue.Queue(maxsize=25)


def worker():
    while True:
        task = processing_queue.get()
        if task is None:
            break
        audio_chunk, sr, response_object = task
        try:
            processed_audio = process_audio(audio_chunk, sr)
            response_object['processed_audio'] = processed_audio.tolist()
            response_object['success'] = True
        except Exception as e:
            logging.error(f"Error in audio processing: {e}")
            response_object['error'] = str(e)
            response_object['success'] = False
        finally:
            processing_queue.task_done()


worker_threads = []
for i in range(PROCESSING_THREADS):
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    worker_threads.append(t)


@app.route('/convert', methods=['POST'])
def convert():
    try:
        data = request.get_data()
        audio_chunk = np.frombuffer(data, dtype=np.float32)
        sr = 16000
        if len(audio_chunk) < 1000:
            processed_audio = process_audio(audio_chunk, sr)
            return jsonify({'processed_audio': processed_audio.tolist()})
        response_object = {'success': False}
        if processing_queue.qsize() >= processing_queue.maxsize - 1:
            return jsonify({'error': 'Server overloaded, please try again later'}), 503
        processing_queue.put((audio_chunk, sr, response_object))
        timeout = 10
        start_wait = time.time()
        while not response_object.get('success', False) and time.time() - start_wait < timeout:
            time.sleep(0.1)
        if response_object.get('success', False):
            return jsonify({'processed_audio': response_object['processed_audio']})
        elif 'error' in response_object:
            return jsonify({'error': response_object['error']}), 500
        else:
            return jsonify({'error': 'Processing timeout'}), 504
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/update_target', methods=['POST'])
def update_target():
    global target_mel, target_length, current_target_audio_path
    try:
        data = request.get_json()
        if "target_filename" in data:
            new_target_path = os.path.join("./sample", data["target_filename"])
            if not os.path.exists(new_target_path):
                return jsonify({'status': 'error', 'message': f'File not found: {new_target_path}'}), 404
            current_target_audio_path = new_target_path
            logging.info(f"Target audio path updated to: {current_target_audio_path}")
            target_mel, target_length = precompute_target_features(current_target_audio_path)
            logging.info("Target audio updated successfully.")
            return jsonify({'status': 'success', 'message': 'Target audio updated successfully'}), 200
    except Exception as e:
        logging.error(f"Error updating target audio: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/switch_vocoder', methods=['POST'])
def switch_vocoder():
    try:
        data = request.get_json()
        vocoder_type = data.get('vocoder_type', 'bigvgan')

        if vocoder_type not in ['bigvgan', 'hifigan']:
            return jsonify({'status': 'error', 'message': 'Invalid vocoder type'}), 400

        load_models(vocoder_type)
        return jsonify({
            'status': 'success',
            'message': f'Switched to {vocoder_type} vocoder',
            'current_vocoder': current_vocoder_type,
            'noise_floor': noise_floor_dict.get(vocoder_type, 0.007)
        }), 200
    except Exception as e:
        logging.error(f"Error switching vocoder: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=False)

