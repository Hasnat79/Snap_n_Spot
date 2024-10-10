from config import Config
from utils import load_model_and_processors,Dataloader,calculate_snippet_query_scores
import numpy as np
import matplotlib.pyplot as plt


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
    # description ="When the guy jumps, the son flies up and hits the wall."
    # video = load_video(video_path, fps = 1)
    snippet_query_scores = calculate_snippet_query_scores(video_path,model, vis_processors, text_processors, fps =3, query = description, device=device) #[1, 11]

    print(f"type(snippet_query_scores): {type(snippet_query_scores)}")
    
    snippet_query_scores = snippet_query_scores.flatten().tolist()

    differences = []
    for i in range(len(snippet_query_scores)):
      d_q = []
      w_q = []
      for j in range(len(snippet_query_scores)):
        d_q.append((snippet_query_scores[i]-snippet_query_scores[j])**2)
      
      print(f"d_q: {d_q}")
      # find argmax of d_q
      m_q = np.argmax(np.array(d_q))
      for j in range(len(snippet_query_scores)):
        w_q.append(1 - (d_q[j]/((snippet_query_scores[i]-snippet_query_scores[m_q])**2)))
      # print(f"m_q: {m_q}")
      # w_q.append(1 - (d_q[i])/ (( snippet_query_scores[i] - snippet_query_scores[m_q] )**2)  )
      print(f"w_q: {w_q}")
      print("\n\n====================\n\n")
      # Plot each probability w_q for each frame
      plt.figure()
      plt.bar(range(len(w_q)), w_q)
      plt.xlabel('Frame Index')
      plt.ylabel('Probability')
      plt.title(f'Probability Distribution for Frame {i}')
      plt.savefig(f'frame_{i}_prob_dist.png')
    exit()


    print(f"differences: {differences}")
    print(f"max_diff_index: {np.argmax(np.array(differences))}")
    
    # find the index of the maximum difference
    max_diff_index = np.argmax(np.array(differences))

    #probability of a snippet being in the same moment with s conditioned on queery

    probabilities = []
    for i in range(len(differences)):
      if i == max_diff_index:
        probabilities.append(0)
        continue
      print(f"differences [i]: {differences[i]}")
      print(f"snippet_query_scores[i]: {snippet_query_scores[i]}")
      prob = 1 - (differences[i]/((snippet_query_scores[i]-snippet_query_scores[max_diff_index])**2))
      probabilities.append(prob)
    print(f"probabilities: {probabilities}")
    print(f"probabilities: {np.argmax(np.array(probabilities))}")


    break
