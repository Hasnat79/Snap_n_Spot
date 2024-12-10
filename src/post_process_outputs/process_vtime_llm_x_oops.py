from utils import LightweightTimeExtractor
import sys
import json
sys.path.append("/scratch/user/hasnat.md.abdullah/Snap_n_Spot")
from outputs import vtime_llm_x_oops_raw

if __name__ == "__main__":
  extractor = LightweightTimeExtractor()
  for k, v in vtime_llm_x_oops_raw.items():
    vtime_generation = v['vtime_llm_generation']
    # start,end = get_timestamps(timechat_generation)
    
    print(f"vtime_generation: {vtime_generation}")
    
    result = extractor.extract_vtime_times(vtime_generation)
    print(f"result: {result}")
    v['start'] = result['start']
    v['end'] = result['end']
    # break
    vtime_llm_x_oops_raw[k] = v
  output_path = "/scratch/user/hasnat.md.abdullah/Snap_n_Spot/outputs/vtime_llm_x_oops/vtime_llm_x_oops_results_processed.json"
  with open(output_path, 'w') as f:
    json.dump(vtime_llm_x_oops_raw, f, indent=2)
