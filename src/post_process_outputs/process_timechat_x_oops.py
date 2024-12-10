import sys
sys.path.append("/home/grads/h/hasnat.md.abdullah/Snap_n_Spot/")
from outputs import timechat_x_oops_processed
import re
import json
from transformers import pipeline
class LightweightTimeExtractor:
    def __init__(self):
        """
        Initialize with a question-answering pipeline using a smaller BERT model
        """
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/tinyroberta-squad2",  # Much smaller model
            device="cpu",
            cache_dir = "cache"
        )
    
    def extract_times(self, text: str) -> dict:
        """
        Extract time ranges using a series of targeted questions
        """
        result = {}
        
        # Extract start time
        start_question = "What is the start time?"
        start_answer = self.qa_pipeline(
            question=start_question,
            context=text
        )
        # print(f"text: {text}")
        print(f"start_answer: {start_answer}")
        try: 
          start = float(start_answer['answer'].split()[0])
        except:
          start = None
        
        result ['start'] = start

        
              
        # Extract end time
        end_question = "What is the end time?"
        end_answer = self.qa_pipeline(
            question=end_question,
            context=text
        )
        print(f"end_answer: {end_answer}")
        try :
          end = float(end_answer['answer'].split()[0])
        except:
          end = None
  
        result['end'] = end
        
        
        return result if result else None
# def get_timestamps(timechat_generation):
#   pattern_1 = re.compile(r'\d+ - \d+ seconds')
#   pattern_2 = re.compile(r'\d+\.\d+ - \d+\.\d+ seconds')
#   pattern_3 = re.compile(r'Start - \d+\.\d+ seconds, End - \d+\.\d+ seconds')
#   pattern_4 = r"Start - (\d+\.\d+) seconds\. End - (\d+\.\d+) seconds\."
#   times = re.findall(pattern_1, timechat_generation)
#   if not times:
#     times = re.findall(pattern_2, timechat_generation)
#   if not times:
#     times = re.findall(pattern_3, timechat_generation)
  
#   print(f"times found: {times}")
#   if times:
#     if 'Start' in times[0]:
#       start, end = times[0].replace('Start - ', '').replace('End - ', '').split(", ")
#     else:
#       start, end = times[0].split(" - ")
#     start = start.replace(" seconds", "")
#     end = end.replace(" seconds", "")
#     print(f"start: {start}, end: {end}")
#     print(f"type(start): {type(start)}, type(end): {type(end)}")
#     return start, end
#   else:
#     return None, None
def get_timestamps(timechat_generation):
  # Define all regex patterns
  pattern_1 = re.compile(r'\d+ - \d+ seconds')
  pattern_2 = re.compile(r'\d+\.\d+ - \d+\.\d+ seconds')
  pattern_3 = re.compile(r'Start - \d+\.\d+ seconds, End - \d+\.\d+ seconds')
  pattern_4 = re.compile(r"Start - (\d+\.\d+) seconds\. End - (\d+\.\d+) seconds\.")
  pattern_5 = re.compile(r"Start position: (\d+\.\d+)\. End position: (\d+\.\d+)\.")
  pattern_6 = re.compile(r"Start position: (\d+\.\d+) seconds\. End position: (\d+\.\d+) seconds\.")

  # Search for timestamps using all patterns
  match_1 = pattern_1.search(timechat_generation)
  match_2 = pattern_2.search(timechat_generation)
  match_3 = pattern_3.search(timechat_generation)
  match_4 = pattern_4.search(timechat_generation)
  match_5 = pattern_5.search(timechat_generation)
  match_6 = pattern_6.search(timechat_generation)

  # Check for matches and extract accordingly
  if match_6:
    start = match_6.group(1)
    end = match_6.group(2)
    print(f"start: {start}, end: {end}")
    print(f"type(start): {type(start)}, type(end): {type(end)}")
    return start, end
  elif match_5:
    start = match_5.group(1)
    end = match_5.group(2)
    print(f"start: {start}, end: {end}")
    print(f"type(start): {type(start)}, type(end): {type(end)}")
    return start, end
  elif match_4:
    start = match_4.group(1)
    end = match_4.group(2)
    print(f"start: {start}, end: {end}")
    print(f"type(start): {type(start)}, type(end): {type(end)}")
    return start, end
  elif match_3:
    times = match_3.group(0).split(", ")
    start = times[0].replace('Start - ', '').replace(' seconds', '')
    end = times[1].replace('End - ', '').replace(' seconds', '')
    print(f"start: {start}, end: {end}")
    return start, end
  elif match_2:
    times = match_2.group(0).split(" - ")
    start = times[0]
    end = times[1].replace(" seconds", "")
    print(f"start: {start}, end: {end}")
    return start, end
  elif match_1:
    times = match_1.group(0).split(" - ")
    start = times[0]
    end = times[1].replace(" seconds", "")
    print(f"start: {start}, end: {end}")
    return start, end
  
  print("No match found.")
  return None, None
  
extractor = LightweightTimeExtractor()
for k, v in timechat_x_oops_processed.items():
  timechat_generation = v['timechat_generation']
  # start,end = get_timestamps(timechat_generation)
  if v['start'] == None and v['end'] == None:
    print(f"timechat_generation: {timechat_generation}")
    
    result = extractor.extract_times(timechat_generation)
    print(f"result: {result}")
    v['start'] = result['start']
    v['end'] = result['end']
  
    # timechat_x_oops_processed[k] = v
output_path = "/home/grads/h/hasnat.md.abdullah/Snap_n_Spot/outputs/timechat_x_oops/timechat_x_oops_results_processed.json"
print(f"len(timechat_x_oops_processed): {len(timechat_x_oops_processed)}")
with open(output_path, "w") as f:
  json.dump(timechat_x_oops_processed, f, indent=2)
print(f"Processed results saved to {output_path}")
