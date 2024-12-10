import os
import json
import sys
sys.path.append("/scratch/user/hasnat.md.abdullah/Snap_n_Spot/vid_llms/VTimeLLM")
sys.path.append("/scratch/user/hasnat.md.abdullah/Snap_n_Spot")
from data import uag_oops
import argparse
import torch
from vtimellm.constants import IMAGE_TOKEN_INDEX
from vtimellm.conversation import conv_templates, SeparatorStyle
from vtimellm.model.builder import load_pretrained_model, load_lora
from vtimellm.utils import disable_torch_init
from vtimellm.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, VideoExtractor
from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer
from easydict import EasyDict as edict
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import numpy as np
import clip
from joblib import Memory
cachedir = './cachedir'
mem = Memory(cachedir)

def inference(model, image, query, tokenizer):
    


    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image[None,].cuda(),
            do_sample=True,
            temperature=0.05,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True)

        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1295

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--clip_path", type=str, default="/scratch/user/hasnat.md.abdullah/Snap_n_Spot/vid_llms/VTimeLLM/checkpoints/ViT-L-14.pt")
    parser.add_argument("--model_base", type=str, default="/scratch/user/hasnat.md.abdullah/Snap_n_Spot/vid_llms/VTimeLLM/checkpoints/vicuna-7b-v1.5")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="/scratch/user/hasnat.md.abdullah/Snap_n_Spot/vid_llms/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str, default="/scratch/user/hasnat.md.abdullah/Snap_n_Spot/vid_llms/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--stage3", type=str, default="/scratch/user/hasnat.md.abdullah/Snap_n_Spot/vid_llms/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage3")
    parser.add_argument("--video_path", type=str, default="/scratch/user/hasnat.md.abdullah/Snap_n_Spot/vid_llms/VTimeLLM/images/demo.mp4")
    args = parser.parse_args()

    return args
@mem.cache
def load_model (args, stage2,stage3):
    tokenizer, model, context_len = load_pretrained_model(args, stage2, stage3)
    model = model.cuda()
    model.to(torch.float16)
    return tokenizer, model, context_len
def process_video(video_path):
    clip_model, _ = clip.load(args.clip_path)
    clip_model.eval()
    clip_model = clip_model.cuda()
    video_loader = VideoExtractor(N=100)
    _, images = video_loader.extract({'id': None, 'video': video_path})
    transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # print(images.shape) # <N, 3, H, W>
    images = transform(images / 255.0)
    images = images.to(torch.float16)
    with torch.no_grad():
        features = clip_model.encode_image(images.to('cuda'))
    return features
if __name__ == "__main__":
    args = parse_args()
    disable_torch_init()
    tokenizer, model, context_len = load_model(args, args.stage2, args.stage3)
    
    
    output_dir = "/scratch/user/hasnat.md.abdullah/Snap_n_Spot/outputs/vtime_llm_x_oops"
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(f"{output_dir}/vtime_llm_x_oops_results_raw.json"):
      with open(f"{output_dir}/vtime_llm_x_oops_results_raw.json", "r") as f:
        results = json.load(f)
    else:
      results = {}
    processed = 0
    errors = 0
    for i, (k, v) in enumerate(uag_oops.items()):
      print(f"Processing video {i+1}/{len(uag_oops)}")
      video_dir = "/scratch/user/hasnat.md.abdullah/Snap_n_Spot/data/oops_video/val"
      video_path = os.path.join(video_dir, k + ".mp4")
      print(f"Video_path: {video_path}")
      query = f"""You are given a video containing unusual activities that are not typically seen in everyday life. Please watch the video and predict the start and end time of the unusual activity given the description. The format should be: start time - end time. For example, if the unusual activity is from 0:10 to 0:20, you should write 0:10 - 0:20.
      Description: {v['description']}"""
      print(f"prompt: {v['description']}")
      try: 
        if k in results:
          print("Already processed")
          continue
        features = process_video(video_path)


        vtime_llm_generation = inference(model, features, query, tokenizer)
        v['vtime_llm_generation'] = vtime_llm_generation
        results[k] = v
        with open(f"{output_dir}/vtime_llm_x_oops_results_raw.json", "w") as f:
          json.dump(results, f, indent=2)

      except Exception as e:
        print(f"Error: {e}")
        v['vtime_llm_generation'] = "Error"
        results[k] = v
        with open(f"{output_dir}/vtime_llm_x_oops_results_raw.json", "w") as f:
          json.dump(results, f, indent=2)
        
        continue


    # check total      

    
    
    
    
    
    
    
    
    
    
    
    
    video_path = args.video_path
    features = process_video(video_path)
    
    query = "<video>\n What is the person doing?"
    print("query: ", query)
    print(inference(model, features, query, tokenizer))


