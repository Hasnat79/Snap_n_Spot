import json
import os
from tqdm import tqdm
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from torchvision import transforms
import random
import glob
import argparse
from pathlib import Path

device = 'cuda'
model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device=device, is_eval=True,)
vis_processors = transforms.Compose([
    t for t in vis_processors['eval'].transform.transforms if not isinstance(t, transforms.ToTensor)
])

from decord import VideoReader, cpu

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
    return buffer


@torch.no_grad()
def get_visual_features(video_path, fps=None, stride=None, max_duration=None, batch_size=128):
    video = loadvideo(video_path, fps, stride, max_duration)
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
    return features.numpy()

def get_args():
    parser = argparse.ArgumentParser(description='Extract BLIP-2 features for videos')
    parser.add_argument('--input_root', required=False, type=str, default = "/scratch/user/hasnat.md.abdullah/Snap_n_Spot/data/oops_video/val")
    parser.add_argument('--save_root', required=False, type=str, default = "/scratch/user/hasnat.md.abdullah/Snap_n_Spot/data/oops_video/blip_2_features")
    parser.add_argument('--stride', default=None, type=int)
    parser.add_argument('--fps', default=3, type=float)
    parser.add_argument('--batch_size', default=128, type=int)

    return parser.parse_args()



if __name__=='__main__':
    oops_uag_data_path ="/scratch/user/hasnat.md.abdullah/Snap_n_Spot/data/oops_uag_paper_version.json"
    with open(oops_uag_data_path) as f:
        oops_uag_data = json.load(f)
    
    args = get_args()
    # videos = os.listdir(args.input_root)
    # random.shuffle(videos)
    for video_name in tqdm(oops_uag_data.keys()):
        save_path = os.path.join(args.save_root, video_name)+'.npy'
        if os.path.exists(save_path):
            continue
        else: 
            print(f"does not exist: {save_path}")
            video_path = os.path.join(args.input_root, video_name)+'.mp4'
            features = get_visual_features(video_path, fps=args.fps, stride=args.stride, batch_size=args.batch_size)
            # gpu cache available report in GB
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            np.save(save_path, features)

