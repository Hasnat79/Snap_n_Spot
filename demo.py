import numpy as np
import argparse
from src.vlm_localizer import localize

import torch



import torch
import numpy as np
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from torchvision import transforms
import argparse
from pathlib import Path

device = 'cuda'
model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device=device, is_eval=True,)
vis_processors = transforms.Compose([
    t for t in vis_processors['eval'].transform.transforms if not isinstance(t, transforms.ToTensor)
])

from decord import VideoReader, cpu

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding.')
    parser.add_argument('--video_path', default='uag_oops', type=str, help='Specify the dataset. See supported datasets in data_configs.py.')
    parser.add_argument('--query', default='description', type=str, help='Specify the dataset. See supported datasets in data_configs.py.')
    return parser.parse_args()

def loadvideo(fname, fps=3, stride=None, max_duration=None):
    vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
    duration = len(vr) / vr.get_avg_fps()
    if fps is not None:
        num_sampled_frames = round(duration * fps)
        all_index = np.linspace(0, len(vr)-1, num=num_sampled_frames).round().astype(np.int32)
        if max_duration is not None:
            all_index = all_index[:round(max_duration * fps)]
    else:
        assert stride is not None
        all_index = np.arange(0, len(vr), stride, dtype=np.int32)
        if max_duration is not None:
            all_index = all_index[:round(max_duration * all_index.shape[0] / duration)]
    vr.seek(0)
    buffer = vr.get_batch(all_index).permute(0, 3, 1, 2) / 255.
    return buffer, duration


@torch.no_grad()
def get_visual_features(video_path, fps=None, stride=None, max_duration=None, batch_size=128):
    video,duration = loadvideo(video_path, fps, stride, max_duration)
    img = vis_processors(video)
    features = []
    for bid in range(0, img.size(0), batch_size):
        batch_img = img[bid:bid+batch_size].to(device)
        with model.maybe_autocast():
            image_embeds = model.ln_vision(model.visual_encoder(batch_img))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_feats = model.vision_proj(query_output.last_hidden_state)
        features.append(image_feats.cpu().half())
    features = torch.cat(features, dim=0)
    print(f"Features shape: {features.shape}")
    return features.numpy(),duration


def infer(video_path, query, stride=64, max_stride_factor=1, pad_sec=0.0):
    features, duration = get_visual_features(video_path, fps=3, stride=stride, max_duration=None, batch_size=128)
    np.save('video.npy', features)
    ans = localize('video.npy', duration, [{'descriptions': [query]}], stride, int(features.shape[0] * max_stride_factor))
    print(ans)
    return ans

if __name__=='__main__':
    args = get_args()
    infer(args.video_path, args.query, stride=64, max_stride_factor=1, pad_sec=0.0)

  