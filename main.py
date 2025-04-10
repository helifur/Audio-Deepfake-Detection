import argparse

import librosa
import numpy as np
import torch
import yaml
from models.model import RawNet


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")

    model = RawNet(config["model"], device=device)
    model.load_state_dict(
        torch.load("./model.pth", map_location="cpu", weights_only=False)
    )
    model.eval()
    model = model.to(device)

    y, _ = process_audio(args.input)

    segments = []
    max_len = 96000
    total_len = len(y)

    if total_len <= max_len:
        y_pad = pad(y, max_len)
        segments.append(torch.tensor(y_pad, dtype=torch.float32))
    else:
        num_chunks = total_len // max_len
        for i in range(num_chunks):
            seg = y[i * max_len : (i + 1) * max_len]
            segments.append(
                torch.tensor(pad(seg, max_len), dtype=torch.float32)
            )

    model.eval()
    probs = []
    with torch.no_grad():
        for segment in segments:
            input_tensor = segment.unsqueeze(0).to(device)
            output = model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            softmax_out = torch.softmax(output, dim=1)[0][1].item()
            probs.append(softmax_out)

    final_prob = float(np.mean(probs))
    label = "Likely Real" if final_prob > 0.5 else "Likely Fake"
    print(f"Score: {final_prob}")
    print(label)


def pad(y, max_len=96000):
    x_len = y.shape[0]
    if x_len >= max_len:
        return y[:max_len]

    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(y, num_repeats)[:max_len]
    return padded_x


def process_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    if sr != 24000:
        y = librosa.resample(y, orig_sr=sr, target_sr=24000)

    return y, sr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthetic voice detector with RawNet3 implementation"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default="", help="Model config yaml file"
    )

    parser.add_argument("--input", type=str, default="", help="Input file")
    args = parser.parse_args()
    main(args)
