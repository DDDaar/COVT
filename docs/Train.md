# CoVT Training Instructions

## Preparations

1. Firstly install the dependencies. To ensure compatibility between the training environment of Qwen2.5-VL and other visual models (e.g., [Segment Anything Model](https://github.com/facebookresearch/segment-anything), [Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2), [DINO v2](https://github.com/facebookresearch/dinov2), etc.), we run:
    ```
    cd train
    bash install_training.sh
    ```

2. Download the visual models' checkpoints. 

* SAM is the anchor model for segmentation model. To prepare its checkpoint, download the [VITiT-H SAM Model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) to the `train/src/anchors/segment_anything/ckpt/`.


* DepthAnthing v2 is the anchor model for depth estimation. Download [DepthAnthing v2 Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) to the `train/src/anchors/DepthAnything/ckpt/`.

* PIDINet is for edge detection. It is very lightweight, to download the ckpt file, view [PIDINet checkpoint](https://github.com/hellozhuo/pidinet/blob/master/trained_models/table5_baseline.pth), download it and put it under `train/src/anchors/pidinet/ckpt/`.

3. Prepare the training data.

We have released training data on [huggingface](https://huggingface.co/datasets/Wakals/CoVT-Dataset). Our training script is based on the implementation from [this GitHub repository](https://github.com/2U1/Qwen-VL-Series-Finetune). Therefore, to keep the consistency, the dataset formatted according to the LLaVA specification. The dataset should be a JSON file where each entry contains information about conversations and images. Ensure that the image paths in the dataset match the provided `--image_folder`.

The example is like:
<details>
<summary>Example</summary>

```json
[
  {
    "id": "000000033471",
    "image": "000000033471.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      },
      {
        "from": "human",
        "value": "What feature can be seen on the back of the bus?"
      },
      {
        "from": "gpt",
        "value": "The back of the bus features an advertisement."
      },
      {
        "from": "human",
        "value": "Is the bus driving down the street or pulled off to the side?"
      },
      {
        "from": "gpt",
        "value": "The bus is driving down the street, which is crowded with people and other vehicles."
      }
    ]
  }
  ...
]
```
Make sure that `<image>` appears at the beginning of the first row in `conversations`.
</details>

## Training

The training process consists of four stages. By using `4 x A6000`, they use around `20 hours` in total.
The first stage is the **understanding stage**, after which we perform the first LoRA merge.  
The second stage is the **generation stage**, the third stage is the **reasoning stage**, and the fourth stage is the **efficient reasoning stage**.  
After completing all stages, we merge the LoRA parameters again.  
We have integrated the entire workflow, so you only need to run:
```
bash scripts/finetune.sh
```
The final checkpoint will be saved under `train/output/lora_merged` as `lora_stage234_merged`.

You can customize the configure in `scripts/finetune.sh`. Here are the parameters you need to configure and their definitions:

- **DATA_PATH**: The path of the data json file.
- **VISUAL_MODEL_ID**: The visual models you want to align.
    - `sam`: segmentation tokens
    - `depth`: depth tokens
    - `dino`: DINO tokens
    - `pidinet`: edge tokens
    - `metaclip`: MetaCLIP tokens
    - `siglip`: SigLIP tokens