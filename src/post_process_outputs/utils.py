import sys
# sys.path.append("/home/grads/h/hasnat.md.abdullah/Snap_n_Spot/")
from transformers import pipeline
import os
# os.environ['HF_HOME'] = "/scratch/user/hasnat.md.abdullah/Snap_n_Spot/src/cachedir"
class LightweightTimeExtractor:
    def __init__(self):
        """
        Initialize with a question-answering pipeline using a smaller BERT model
        """
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/tinyroberta-squad2",  # Much smaller model
            device="cpu",
            cache_dir = "/scratch/user/hasnat.md.abdullah/Snap_n_Spot/src/cachedir"
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
    def extract_vtime_times(self, text: str) -> dict:
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
              start = float(start_answer['answer'].split(":")[1])
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
              end = float(end_answer['answer'].split(":")[1])
            except:
              end = None
      
            result['end'] = end
            
            
            return result if result else None
