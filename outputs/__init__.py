import json
dir_path = "/scratch/user/hasnat.md.abdullah/Snap_n_Spot/outputs/"
timechat_x_oops_path = dir_path+"timechat_x_oops/timechat_x_oops_results_raw.json"
timechat_x_oops_processed_path= dir_path+"timechat_x_oops/timechat_x_oops_results_processed.json"

vtime_llm_x_oops_path = dir_path+"vtime_llm_x_oops/vtime_llm_x_oops_results_raw.json"

timechat_x_oops_raw = json.load(open(timechat_x_oops_path))
timechat_x_oops_processed = json.load(open(timechat_x_oops_processed_path))

vtime_llm_x_oops_raw = json.load(open(vtime_llm_x_oops_path))
print(f"len(timechat_x_oops_raw): {len(timechat_x_oops_raw)}")
print(f"len(timechat_x_oops_processed): {len(timechat_x_oops_processed)}")
print(f"len(vtime_llm_x_oops_raw): {len(vtime_llm_x_oops_raw)}")
