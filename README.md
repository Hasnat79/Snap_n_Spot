<div align="center">

# Snap&Spot: Leveraging Large Vision Language Models for Train-Free Zero-Shot Localization of Unusual Activities in Video
[**Hasnat Md Abdullah**](https://github.com/Hasnat79)

Texas A&M University

</div>

<!-- ## 📜 Abstract -->


<div align="center">
  <img src="Figure/overview.png" alt="example" width="700"/>
</div>

## 🔧 Getting Started
- Clone this repository
```bash
git clone https://github.com/Hasnat79/Snap_n_Spot
```
- init the submodules (foundation_models)
```bash
git submodule update --init --recursive
```
## 🚀 Installation

To install the necessary dependencies, run:

```bash
conda create -n snap
conda activate snap
pip install -r requirements.txt
```


## 📂 Dataset


[/data](data) directory contains [charades-sta](data/charades-sta/charades_test.json) and [uag_oops](data/uag_oops_charades_format.json) annnotation files. [oops_video/val](data/oops_video/val) contains the videos of [UAG-OOPS dataset](https://huggingface.co/datasets/hasnat79/ual_bench). [charades-sta](data/charades-sta) contains the videos of the [Charades-STA dataset](https://huggingface.co/datasets/hasnat79/ual_bench). 


### ⚙️ Blips2 feature generation
```bash 
cd src
python feature_extraction.py 
```
- genrates blip2 features for the videos in the [data](data) directory in numpy format


## 🧠 Methodology __ [Colab demo](https://colab.research.google.com/drive/1QoMa01UGrx71p838uAXrTAokb1xPEct8?usp=sharing)
```bash 
cd src
python evaluate.py --dataset uag_oops
```
- generates the metrics for zero-shot unusual activity localization on UAG-OOPS dataset using the Snap&Spot pipeline.
<details>
  <summary> Click to see the output format </summary>
  
  Expected output format:
  ```bash 
  R@0.3: 0.6620967741935484
  R@0.5: 0.49489247311827955
  R@0.7: 0.23951612903225805
  ```
 </details>

## Try for a single video and query
```bash
python demo.py \
 --video_path "/Snap_n_Spot/data/oops_video/34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)0.mp4" \
 --query "A guy jumps onto a bed where his son is. When the guy jumps, the son flies up and hits the wall."
 ```

## 📝 Evaluate any dataset using Our methodology
- set up the dataset in the [data_configs](src/data_configs.py) file
- generate the features using the [feature_extraction.py](src/feature_extraction.py) file
- run the evaluation using the [evaluate.py](src/evaluate.py) file

**Note**: You need to make sure you are updating the paths correctly in the config file and inside the scripts.





