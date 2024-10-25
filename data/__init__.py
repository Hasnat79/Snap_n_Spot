import json
import os
oops_path = "/home/grads/h/hasnat.md.abdullah/Snap_n_Spot/data/oops_uag_paper_version.json"

uag_oops = json.load(open(oops_path))
print(f"Number of videos: {len(uag_oops)}")

# val_dir = "/home/grads/h/hasnat.md.abdullah/Snap_n_Spot/data/oops_video/val"
# missing_videos = []

# for key in uag_oops.keys():
#   video_file = os.path.join(val_dir, f"{key}.mp4")
#   if not os.path.exists(video_file):
#     missing_videos.append(f"{key}.mp4")

# missing_videos_path = "/home/grads/h/hasnat.md.abdullah/Snap_n_Spot/data/missing_videos.json"
# with open(missing_videos_path, 'w') as f:
#   json.dump(missing_videos, f,indent=2)

# print(f"Missing videos saved to {missing_videos_path}")
# print(f"Number of missing videos: {len(missing_videos)}")  #688
