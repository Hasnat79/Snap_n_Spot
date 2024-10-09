import json
import numpy as np
from lavis.models import load_model_and_preprocess
from torchvision import transforms
from decord import VideoReader, cpu
import torch 
def load_model_and_processors(config:object, vlm_name:str, device:str):
  """loads model and processors given the vlm_name 

  Args:
      config (object): 
      vlm_name (str): vlm name from config
      device (str): 'cuda' or 'cpu'

  Returns:
      tuple: model, visprocessor, text_processor
  """
  if vlm_name == 'blip2':
    assert vlm_name in config.vlm, f"VLM {vlm_name} not found in config"
    model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device=device, is_eval=True)
    vis_processors = transforms.Compose([
        t for t in vis_processors['eval'].transform.transforms if not isinstance(t, transforms.ToTensor)
    ])
    model,vis_processors,text_processors= None,None,None
    return model, vis_processors, text_processors

class Dataloader: 
  """ Dataloader class to iterate over dataset
  returns the annotations (e.g. start, end, description) for each video
  """
  def __init__(self, config, dataset_name:str):
    self.dataset = self.__get_dataset(config, dataset_name)
    self.values = config.DATASET[dataset_name]['values']
    self.video_dir = config.DATASET[dataset_name]['video_dir']
  def __iter__(self):
    for video_id in self.dataset:
      data = self.dataset[video_id]
      annotations = {k: data[k] for k in self.values}
      video_path = f"{self.video_dir}/{video_id}.mp4"
      yield video_path, video_id, annotations
  
  
  def __get_dataset(self,config, dataset_name:str):
    """loads dataset from json file

    Args:
        config (Config class): 
        dataset_name (str): 
    Returns:
        dict:
    """
    assert dataset_name in config.DATASET, f"Dataset {dataset_name} not found in config"
    with open(config.DATASET[dataset_name]['file']) as f:
        data = json.load(f)
    
    return data


def load_video(video_path:str, fps = 3):
  """loads video from path

  Args:
      video_path (str): path to video

  Returns:
      numpy.ndarray 
  """
  vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
  duration = len(vr) / vr.get_avg_fps()
  if fps is not None: 
    num_sampled_frames = round(duration * fps)
    print(f"num_sampled_frames: {num_sampled_frames}")
    all_index = np.linspace(0, len(vr)-1, num=num_sampled_frames).round().astype(np.int32)
    print(f"length of all_index: {len(all_index)}")
    vr.seek(0)
    print(f"vr.get_batch(all_index).shape: {vr.get_batch(all_index).shape}")
    # buffer = vr.get_batch(all_index).permute(0, 3, 1, 2) / 255.
    buffer = np.transpose(vr.get_batch(all_index).asnumpy(), (0, 3, 1, 2)) / 255.0
    # print(f"buffer shape: {buffer.shape}")
    # print(f"type of buffer: {type(buffer)}")
    return buffer # (frames, channels, height, width)

@torch.no_grad()
def get_video_features(video_path:str, fps:int, model, vis_processors,batch_size:int):
  """extracts video features

  Args:
      video_path (str): path to video
      fps (int): frames per second
      model (object): model
      vis_processors (object): visual processors

  Returns:
      numpy.ndarray: video features
  """
  video = load_video(video_path, fps = fps)
  print(f"video/image list shape before vis_processor: {video.shape}") #[11, 3, 720, 1280]
  img = vis_processors(video)
  print(f"image shape after going through vis_processors: {img.shape}") #[11, 3, 364, 364]
  features = []

  for idx in range(0, img.size(0), batch_size):
    batch_img = img[idx:idx+batch_size].to(device)
    print(f"\tbatch_img shape: {batch_img.shape}")
    with model.maybe_autocast():
      image_embeds = model.ln_vision(model.visual_encoder(batch_img)) 
    image_embeds = image_embeds.float()
    print(f"\timage_embeds shape after model.lnvision: {image_embeds.shape}")
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
    print(f"\timage_atts shape: {image_atts.shape}")
    print(f"\tshapes of model.query_tokens: {model.query_tokens.shape}")
    query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_output = model.Qformer.bert(
        query_embeds=query_tokens, 
        encoder_hidden_states=image_embeds, 
        encoder_attention_mask=image_atts, 
        return_dict=True,
    )
    image_feats = model.vision_proj(query_output.last_hidden_state)
    print(f"\timage_feats shape after vision projection: {image_feats.shape}")
    features.append(image_feats.cpu().half())
  features = torch.cat(features, dim=0)
  print(f"features shape: {features.shape}")
  return features.numpy()


def calculate_snippet_query_scores (video_path:str, model, vis_processors, text_processors, fps:int, query:str):
  """calculates snippet query scores

  Args:
      video_path (str): path to video
      fps (int): frames per second
      query (str): description
      model (object): model
      vis_processors (object): visual processors
      text_processors (object): text processors

  Returns:
      list: snippet query scores
  """
  # video features
  video_features = get_video_features(video_path, fps, model, vis_processors,batch_size=128)
  print(f"video_features shape: {video_features.shape}")
 
  
  return None
