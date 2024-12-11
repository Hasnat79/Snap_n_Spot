import os
import sys
sys.path.append("/scratch/user/hasnat.md.abdullah/Snap_n_Spot")
import argparse
import json

def calculate_recall_at_k(predictions, ground_truth, k, iou_threshold):
  recall_count = 0

  for i in range(len(predictions)):
    gt_start, gt_end = ground_truth[i]
    pred_start, pred_end = predictions[i]
    intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
    union = max(gt_end, pred_end) - min(gt_start, pred_start)
    iou = intersection / union
    if iou >= iou_threshold:
      recall_count += 1
  print(f"Recall@1, IoU >= {iou_threshold}: {recall_count / len(ground_truth):.4f}")
  return recall_count / len(ground_truth)    
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Evaluate Recall@1 for different IoU thresholds")
  parser.add_argument('--input', type=str, required=True, help='Path to the input JSON file')
  args = parser.parse_args()

  with open (args.input) as f:
    predictions = json.load(f)
  # print(f"prediction.keys(): {predictions.keys()}")
  tag = args.input.split('/')[-1].split('.')[0]
  print(f"_________{tag}_________")
  ground_truth = [(float(gt['start_time']), float(gt['end_time'])) for _, gt in predictions.items()]
  predictions = [(float(pred['start']) if pred['start'] is not None else 0.0, 
            float(pred['end']) if pred['end'] is not None else 0.0) 
            for _, pred in predictions.items()]
  recall_at_1_iou_0_5 = calculate_recall_at_k(predictions, ground_truth, 1, 0.5)
  recall_at_1_iou_0_7 = calculate_recall_at_k(predictions, ground_truth, 1, 0.7)

  # for n in vid_llms:
  #   print(f"__________{n}__________")
  #   ground_truth = [(float(gt['start_time']), float(gt['end_time'])) for _, gt in timechat_x_oops_processed.items()]
  #   predictions = [(float(pred['start']) if pred['start'] is not None else 0.0, 
  #           float(pred['end']) if pred['end'] is not None else 0.0) 
  #           for _, pred in timechat_x_oops_processed.items()]

  #   recall_at_1_iou_0_5 = calculate_recall_at_k(predictions, ground_truth, 1, 0.5)
  #   recall_at_1_iou_0_7 = calculate_recall_at_k(predictions, ground_truth, 1, 0.7)
