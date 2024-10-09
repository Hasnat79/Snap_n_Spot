from config import Config
from utils import load_model_and_processors,Dataloader,load_video




if __name__ == '__main__':
  config = Config()

  device = 'cuda' #arg
  vlm_name = 'blip2' #arg
  model, vis_processors, text_processors = load_model_and_processors(config, vlm_name, device)



  dataset_name = 'uag_oops' #arg
  data_loader = Dataloader(config, dataset_name)
  
  # run zero_vmr for one video 
  for video_path, video_id, values in data_loader:
    print(video_path, video_id, values)
    # video = load_video(video_path)
    # print(video)
    description = values['description']
    # video = load_video(video_path, fps = 1)
    snipped_query_scores = calculate_snippet_query_scores(video_path,model, vis_processors, text_processors, fps =1, query = description)
    break
