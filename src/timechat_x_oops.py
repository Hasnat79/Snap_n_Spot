
import sys
sys.path.append("/home/grads/h/hasnat.md.abdullah/Snap_n_Spot/vid_llms/TimeChat")
sys.path.append("/home/grads/h/hasnat.md.abdullah/Snap_n_Spot")
import torch
import argparse
import os
import random
import json
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='./logs/timechat_x_oops/run_2')
import time
import psutil  # Add this import

from data import uag_oops

import torch.backends.cudnn as cudnn
from timechat.common.config import Config
from timechat.common.dist_utils import get_rank
from timechat.common.registry import registry
from timechat.conversation.conversation_video import Chat, Conversation, default_conversation, SeparatorStyle, conv_llava_llama_2
import decord
import cv2
import time
import subprocess
from decord import VideoReader
from timechat.processors.video_processor import ToTHWC, ToUint8, load_video
decord.bridge.set_bridge('torch')

# imports modules for registration
from timechat.datasets.builders import *
from timechat.models import *
from timechat.processors import *
from timechat.runners import *
from timechat.tasks import *

import random as rnd
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image
import gradio as gr
from joblib import Memory
from torch.utils.tensorboard import SummaryWriter
cachedir = './cachedir'
mem = Memory(cachedir)
def parse_args():
  parser = argparse.ArgumentParser(description="Demo")
  parser.add_argument("--cfg-path", default='/home/grads/h/hasnat.md.abdullah/Snap_n_Spot/vid_llms/TimeChat/eval_configs/timechat.yaml', help="path to configuration file.")
  parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
  parser.add_argument("--num-beams", type=int, default=1)
  parser.add_argument("--temperature", type=float, default=0.2)
  parser.add_argument("--text-query", default="What is he doing?", help="question the video")
  parser.add_argument("--video-path", default='examples/hotdog.mp4', help="path to video file.")
  parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
  )
  args = parser.parse_args(args=[])
  return args

# cache
@mem.cache
def load_model(cfg, args):
  DIR = "/home/grads/h/hasnat.md.abdullah/Snap_n_Spot/vid_llms/TimeChat/ckpt/timechat"
  MODEL_DIR = f"{DIR}/timechat_7b.pth"
  model_config = cfg.model_cfg
  model_config.device_8bit = args.gpu_id
  model_config.ckpt = MODEL_DIR
  model_cls = registry.get_model_class(model_config.arch)
  device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
  model = model_cls.from_config(model_config)
  model = model.to(device)
  model.eval()
  return model, device

@mem.cache
def load_vis_processor(cfg):
  vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
  vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
  return vis_processor
def log_system_utilization(writer, step):
  cpu_usage = psutil.cpu_percent(interval=1)
  writer.add_scalar('CPU Usage', cpu_usage, step)
  if torch.cuda.is_available():
    gpu_usage = torch.cuda.utilization(1)
    writer.add_scalar('GPU Usage', gpu_usage, step)
def process_video(args, model, vis_processor, device, video_path , prompt, video_index):
  start_time = time.time()
  chat = Chat(model, vis_processor, device=device)
  print('Initialization Finished')
  img_list = []
  chat_state = conv_llava_llama_2.copy()
  chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
  msg = chat.upload_video_without_audio(
    video_path=video_path, 
    conv=chat_state,
    img_list=img_list, 
    n_frms=96,
  )
  text_input = prompt
  print(text_input)

  chat.ask(text_input, chat_state)

  num_beams = args.num_beams
  temperature = args.temperature
  llm_message = chat.answer(conv=chat_state,
                img_list=img_list,
                num_beams=num_beams,
                temperature=temperature,
                max_new_tokens=300,
                max_length=2000)[0]

  end_time = time.time()
  processing_time = end_time - start_time
  writer.add_scalar('Processing Time', processing_time, video_index)
  log_system_utilization(writer, video_index)  # Log system utilization
  print(llm_message+"\n")
  return llm_message

def main():
  args = parse_args()
  cfg = Config(args)
  model, device = load_model(cfg, args)
  vis_processor = load_vis_processor(cfg)

  output_dir = "/home/grads/h/hasnat.md.abdullah/Snap_n_Spot/outputs/timechat_x_oops"
  os.makedirs(output_dir, exist_ok=True)

  if os.path.exists(f"{output_dir}/timechat_x_oops_results_raw.json"):
    with open(f"{output_dir}/timechat_x_oops_results_raw.json", "r") as f:
      results = json.load(f)
  else:
    results = {}

  processed = 0
  errors = 0
  for i, (k, v) in enumerate(uag_oops.items()):
    print(f"Processing video {i+1}/{len(uag_oops)}")
    video_dir = "/home/grads/h/hasnat.md.abdullah/Snap_n_Spot/data/oops_video/val"
    video_path = os.path.join(video_dir, k + ".mp4")
    print(f"Video_path: {video_path}")
    prompt = f"""You are given a video containing unusual activities that are not typically seen in everyday life. Please watch the video and predict the start and end time of the unusual activity given the description. The format should be: start time - end time. For example, if the unusual activity is from 0:10 to 0:20, you should write 0:10 - 0:20.
    Description: {v['description']}"""
    print(f"Prompt: {v['description']}")
    try: 
      if k in results:
        print("Already processed")
        processed += 1
        writer.add_scalar('Processed', processed, i)
        continue
      timechat_generation = process_video(args, model, vis_processor, device, video_path, prompt,i)
      v['timechat_generation'] = timechat_generation
      results[k] = v
      with open(f"{output_dir}/timechat_x_oops_results_raw.json", "w") as f:
        json.dump(results, f, indent=2)

    except Exception as e:
      print(f"Error: {e}")
      v['timechat_generation'] = "Error"
      errors += 1
      writer.add_scalar('Errors', errors, i)
      results[k] = v
      with open(f"{output_dir}/timechat_x_oops_results_raw.json", "w") as f:
        json.dump(results, f, indent=2)
      continue
    
    processed += 1
    writer.add_scalar('Processed', processed, i)
    # with open(f"{output_dir}/timechat_x_oops_results_raw.json", "w") as f:
    #   json.dump(results, f, indent=2)
  writer.close()
  # check total 
if __name__ == "__main__":
  main()
