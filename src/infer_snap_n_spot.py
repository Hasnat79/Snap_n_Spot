import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from lavis.models import load_model_and_preprocess
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json
import os
from data_configs import DATASETS
from eval import abs_dist
def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]
class EnhancedVTG:
    def __init__(self,
                 nms_threshold: float = 0.3,
                 score_threshold: float = 0.2,
                 dynamic_threshold: float = 0.0005):
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.dynamic_threshold = dynamic_threshold
        self.visual_feature_weight = 0.7
        self.semantic_feature_weight = 0.3
        
        # Initialize models
        self.text_model = SentenceTransformer('paraphrase-distilroberta-base-v2')
        self.blip_model, _, _ = load_model_and_preprocess(
            "blip2_image_text_matching", "coco", device='cuda', is_eval=True)

    def get_semantic_scores(self, query: str, caption_list: List[str]) -> torch.Tensor:
        """Get semantic similarity scores from captions (1 fps)."""
        with torch.no_grad():
            embed_query = self.text_model.encode(query, convert_to_tensor=True)
            embed_caption_list = self.text_model.encode(caption_list, convert_to_tensor=True)
            scores = F.cosine_similarity(embed_query.unsqueeze(0), embed_caption_list)
            return scores.cuda()

    def get_visual_scores(self, video_features: np.ndarray, query: str) -> torch.Tensor:
        """Get visual-textual similarity scores (3 fps)."""
        # with torch.no_grad():
        #     text = self.blip_model.tokenizer(
        #         query, padding='max_length', truncation=True,
        #         max_length=35, return_tensors="pt").to('cuda')
            
        #     text_output = self.blip_model.Qformer.bert(
        #         text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        #     text_feat = self.blip_model.text_proj(text_output.last_hidden_state[:,0,:])
            
        # v1 = F.normalize(text_feat, dim=-1)
        # v2 = F.normalize(torch.tensor(video_features, device='cuda'), dim=-1)
        # print(f"v1.shape: {v1.shape}, v2.shape: {v2.shape}")
        # scores = torch.einsum('md,npd->mnp', v1, v2)
        # scores = scores.max(dim=-1)[0].mean(dim=0)
        with torch.no_grad():
            text = self.blip_model.tokenizer(query, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to('cuda')                    
            text_output = self.blip_model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
            text_feat = self.blip_model.text_proj(text_output.last_hidden_state[:,0,:])
    
        v1 = F.normalize(text_feat, dim=-1)
        v2 = F.normalize(torch.tensor(video_features, device='cuda', dtype=v1.dtype), dim=-1)
        scores = torch.einsum('md,npd->mnp', v1, v2)
        scores, _ = scores.max(dim=-1)
        scores = scores.mean(dim=0, keepdim=True)
        return scores

    def get_dynamic_scores(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute dynamic scores by detecting gradual increases in similarity.
        Specifically designed to detect start timestamps.
        """
        # Apply Gaussian smoothing
        kernel_size = 7
        sigma = 1.0
        gaussian_weights = torch.tensor([
            np.exp(-(x - kernel_size//2)**2/float(2*sigma**2))
            for x in range(kernel_size)
        ], dtype=torch.float32).cuda()
        gaussian_weights = gaussian_weights / gaussian_weights.sum()
        
        # Pad and convolve
        padded_scores = F.pad(scores.view(1, 1, -1), (kernel_size//2, kernel_size//2), mode='replicate')
        smoothed_scores = F.conv1d(padded_scores, gaussian_weights.view(1, 1, -1)).view(-1)
        
        # Compute differences to detect increases
        diffs = torch.diff(smoothed_scores)
        diffs = F.pad(diffs, (1, 0), value=0)
        # print(f"diffs shape: {diffs.shape}")
        # print(f"diffs: {diffs}")
        # Detect gradual increases using 3-frame window
        dynamic_scores = torch.zeros_like(scores).squeeze(0)
        dynamic_idxs = torch.zeros_like(scores).squeeze(0)
        # print(f"dynamic_scores shape: {dynamic_scores.shape}")
        # print(f"dynamic_idxs shape: {dynamic_idxs.shape}")
        # for i in range(2, len(diffs)):
        # print(f"dynamic_scores: {dynamic_scores}")
        for i in range(2, len(diffs)):
            # print(f"i: {i}")
            f1, f2, f3 = diffs[i], diffs[i-1], diffs[i-2]
            # print(f"f1: {f1}, f2: {f2}, f3: {f3}")
            # Check for consistent increase patterns
            # print(f"3*f1: {3*f1}, 2*f1+f2: {2*f1+f2}, f1+f2+f3: {f1+f2+f3}")
            # print(f"dynamic_threshold: {self.dynamic_threshold}")
            
            if ((3 * f1) > self.dynamic_threshold or 
                (2 * f1 + f2) > self.dynamic_threshold or 
                (f1 + f2 + f3) > self.dynamic_threshold):
                dynamic_scores[i] = max(3 * f1, 2 * f1 + f2, f1 + f2 + f3)
                # print(f"dynamic_scores[i]: {dynamic_scores[i]}")
                dynamic_idxs[i] = i
                # print(f"dynamic_scores: {dynamic_scores}")
        return dynamic_idxs, dynamic_scores

    def locate_spans(self, 
                    video_features: np.ndarray,
                    query: List[str],
                    caption_list: Optional[List[str]] = None,
                    duration: Optional[float] = None) -> Dict:
        """
        Main method to locate video spans.
        Uses dynamic scores for start time and static scores for end time.
        Handles different frame rates (3 fps for visual, 1 fps for captions).
        """
        # Get visual scores (3 fps)
        visual_scores = self.get_visual_scores(video_features, query)
        # print(f"visual_scores: {visual_scores}")
        visual_scores = (visual_scores - visual_scores.min()) / (visual_scores.max() - visual_scores.min())
        # print(f"normalized_visual_scores: {visual_scores}")
        # Get semantic scores if captions available (1 fps)
        if caption_list:
            semantic_scores = self.get_semantic_scores(query, caption_list)
            semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min())
            
            # Interpolate semantic scores to match visual scores frame rate
            semantic_scores_interp = F.interpolate(
                semantic_scores.view(1, 1, -1),
                size=len(visual_scores),
                mode='linear'
            ).view(-1)
        else:
            semantic_scores_interp = torch.zeros_like(visual_scores)
        # print(f"semantic_scores_interp: {semantic_scores_interp}")
        # Combine scores with weights
        
        combined_scores = (self.visual_feature_weight * visual_scores + self.semantic_feature_weight * semantic_scores_interp)
            # combined_scores = (1.0 * visual_scores + 1.0 * semantic_scores_interp)
       
        print(f"combined_scores: {combined_scores}")
        # Get dynamic scores for start time detection
        dynamic_idxs, dynamic_scores = self.get_dynamic_scores(combined_scores)
        
        # print(f"dynamic_idxs: {dynamic_idxs}")
        print(f"dynamic_scores: {dynamic_scores}")
        # Find potential start points (from dynamic scores)
        # print(f"self.dynamic_threshold: {self.dynamic_threshold}")
        start_candidates = torch.nonzero(dynamic_scores > self.dynamic_threshold).view(-1)
        if len(start_candidates) == 0:
            start_candidates = torch.cat((start_candidates, torch.tensor([0], device=start_candidates.device)))
        print(f"start_candidates: {start_candidates}")
        # Find potential end points (from static scores)
        # print(f"combined_scores: {combined_scores}")
        print(f"self.score_threshold: {self.score_threshold}")
        end_mask = (combined_scores < self.score_threshold).squeeze(0)
        print(f"end_mask: {end_mask}")
        # Generate spans
        spans = []
        for start_idx in start_candidates:
            print(f"start_idx: {start_idx}")
            # print(f"torch.nonzero(end_mask[start_idx:]).view(-1): {end_mask[start_idx:]}")
            # Look for end points after start point
            possible_ends = torch.nonzero(end_mask[start_idx:]).view(-1) + start_idx
            # possible_ends = end_mask[start_idx:] + start_idx
            # possible_ends = end_mask[start_idx:]
            print(f"possible_ends: {possible_ends}")
            
            if len(possible_ends) > 0:
                # Take the furthest continuous segment
                current_end = possible_ends[0]
                for end_idx in possible_ends[1:]:
                    if end_idx - current_end <= 3:  # Allow small gaps (1 second at 3fps)
                        current_end = end_idx
                    else:
                        break

            else:
                current_end = len(combined_scores) - 1
                # print(f"span : {start_idx}, {current_end}")
                # Calculate span score
                # print(f"combined_scores[start_idx:current_end+1]: {combined_scores[start_idx:current_end+1]}")
            span_score = combined_scores.squeeze(0)[start_idx:current_end+1].mean()
            spans.append((int(start_idx), int(current_end), float(span_score)))
        # print(f"spans: {spans}")
        
        # Apply NMS
        spans = self.apply_nms(spans)
        # print(f"spans after nms: {spans}")
        # Convert to timestamps if duration provided
        if duration and spans:
            # print(f"len(visual_scores): {len(visual_scores.squeeze(0))}")
            # print(f"duration: {duration}")
            fps = (len(visual_scores.squeeze(0))/ duration)
            # print(f"fps: {fps}")
            spans = [(start/fps, end/fps, score) for start, end, score in spans]
        
        # exit()
        spans.sort(key=lambda x: x[2], reverse=True)
        # print(f"sorted spans: {spans}")
        return {
            "query": query,
            "spans": spans,
            "scores": combined_scores.cpu().tolist()
        }

    def apply_nms(self, spans: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """Apply Non-Maximum Suppression to spans."""
        if not spans:
            return []
        
        spans = sorted(spans, key=lambda x: x[2], reverse=True)
        kept_spans = []
        
        while spans:
            current_span = spans.pop(0)
            kept_spans.append(current_span)
            
            # Calculate IoU with remaining spans
            spans = [
                span for span in spans
                if self._calculate_iou(current_span, span) < self.nms_threshold
            ]
        
        return kept_spans

    def _calculate_iou(self, span1: Tuple[int, int, float], 
                      span2: Tuple[int, int, float]) -> float:
        """Calculate IoU between two spans."""
        start1, end1, _ = span1
        start2, end2, _ = span2
        
        intersection = min(end1, end2) - max(start1, start2)
        if intersection <= 0:
            return 0.0
        
        union = max(end1, end2) - min(start1, start2)
        return intersection / union

def main(args):
    caption_dir = "/scratch/user/hasnat.md.abdullah/Snap_n_Spot/vid_llms/VTG-GPT/data/uag_oops/caption/test"


    model = EnhancedVTG(
        nms_threshold=args.nms_threshold,
        score_threshold=args.score_threshold,
        dynamic_threshold=args.dynamic_threshold
    )
    model.visual_feature_weight = 1.0
    model.semantic_feature_weight = 0.0
    # Add your evaluation code here
    dataset = DATASETS[args.dataset]
    feature_path_dir= dataset['feature_path']
    with open(dataset['splits'][args.split]['annotation_file']) as f:
        data = json.load(f)
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    ground_truth = []
    predictions = []

    pbar = tqdm(data.items())
    for vid, ann in pbar:

        query_json = []
        for i in range(len(ann['sentences'])):
            query_json.append({'descriptions': [ann['sentences'][i]]})

        feature_path = os.path.join(feature_path_dir, vid+'.npy')
        video_feature = np.load(feature_path)
        if args.dataset == 'uag_oops':
          caption = load_jsonl(os.path.join(caption_dir, vid+'.jsonl'))
          caption_list = [c['description'] for c in caption]
        if args.dataset == 'uag_oops':
            duration = ann['timestamps'][0][1]
        if args.dataset == "charades":
            duration = ann['duration']
        # print(f"feature.shape: {video_feature.shape}")
        # print(f"query_json: {query_json}")
        # print(f"duration: {duration}")
        for query in query_json: 
            # print(f"query: {query['descriptions'][0]}")

            # with caption list
            # ans = model.locate_spans(video_feature, query=query['descriptions'], duration=duration, caption_list=caption_list)
            # without caption list
            if args.use_caption:
                ans = model.locate_spans(video_feature, query=query['descriptions'], duration=duration, caption_list=caption_list)
            ans = model.locate_spans(video_feature, query=query['descriptions'], duration=duration)

            print(f"ans: {ans}")
        
            s,e = ann['timestamps'][0]
            ground_truth.append((s,e))
            sp, se = ans['spans'][0][:2]
            predictions.append((sp, se))
            iou_ = (min(e, se) - max(s, sp)) / (max(e, se) - min(s, sp))
            ious.append(max(iou_, 0))
            recall += thresh <= iou_
        # ans = model.locate_spans(video_feature, query = query_json[0], duration=duration)
        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})
        break
        # break
    print(f"len(ious): {len(ious)}")
    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))
    abs_dist(ground_truth, predictions)

    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Video Temporal Grounding")
    # hyper parameters for uag_oops
    parser.add_argument('--nms_threshold', default=0.3, type=float)
    parser.add_argument('--score_threshold', default=0.5, type=float)
    parser.add_argument('--dynamic_threshold', default=0.0005, type=float)
    #hyper parameters for charades
    # parser.add_argument('--nms_threshold', default=0.3, type=float)
    # parser.add_argument('--score_threshold', default=0.6, type=float)
    # parser.add_argument('--dynamic_threshold', default=0.0005, type=float)


    parser.add_argument('--dataset', default='uag_oops', type=str, help='Specify the dataset. See supported datasets in data_configs.py.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split. See supported splits in data_configs.py.')
    parser.add_argument('--use_caption', action='store_true', help='Use caption for evaluation')
    
    args = parser.parse_args()
    main(args)
# if __name__=='__main__':
#     args = get_args()
#     assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
#     dataset = DATASETS[args.dataset]
#     assert args.split in dataset['splits'], 'Unsupported split. To evaluate other split, please add the configuration in data_configs.py.'
    
#     print('Evaluating', args.dataset, args.split, 'stride', dataset['stride'], 'max_stride_factor', dataset['max_stride_factor'])


#     with open(dataset['splits'][args.split]['annotation_file']) as f:
# #         data = json.load(f)
#     eval(data, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'], dataset['splits'][args.split]['pad_sec'])
        
