import json
timechat_x_oops_path = "/home/grads/h/hasnat.md.abdullah/Snap_n_Spot/outputs/timechat_x_oops/timechat_x_oops_results_raw.json"
timechat_x_oops_processed_path= "/home/grads/h/hasnat.md.abdullah/Snap_n_Spot/outputs/timechat_x_oops/timechat_x_oops_results_processed.json"

timechat_x_oops_raw = json.load(open(timechat_x_oops_path))
timechat_x_oops_processed = json.load(open(timechat_x_oops_processed_path))
