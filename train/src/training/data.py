import copy
import os
from dataclasses import dataclass, field
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from PIL import Image
from transformers import AutoImageProcessor
import re
import numpy as np
import cv2
from torchvision import transforms
import random

from segment_anything import build_sam_vit_h, sam_model_registry, SamPredictor
from src.anchors.DepthAnything.depth_anything_v2.dpt import DepthAnythingV2
from diffusers import AutoencoderKL
from transformers import AutoModel, CLIPImageProcessor

from .params import DataArguments
from .constants import *


# 超长序列截断机制，都是只有一个维度的向量
def truncate_sequence(input_ids, labels, max_length, eos_token_id):
    # 超长截断，留一个位置给eos
    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length-1]
        labels = labels[:max_length-1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()  # 第一个序列的形状
    trailing_dims = max_size[1:]    # 除序列长度外的其他维度
    max_len = max(len(seq) for seq in sequences)  # 最大序列长度
    batch_size = len(sequences)  # 批次大小
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value) # 创建全填充值的张量，形状为 [batch_size, max_len, *trailing_dims]
    # (batch_size, max_len) + trailing_dims 为新的维度元组
   
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        # 将序列数据放在张量的开头
        if padding_side == 'right':
            output.data[i, :length] = seq
        # 将序列数据放在张量的末尾
        else:
            output.data[i, -length:] = seq
    return output


# 图像信息处理
# python -c "
# import requests
# from io import BytesIO
# from qwen_vl_utils import process_vision_info
# from PIL import Image
# import numpy as np

# url = 'https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg'

# # 1. 获取原始维度
# resp = requests.get(url)
# original_img = Image.open(BytesIO(resp.content))
# orig_w, orig_h = original_img.size

# # 2. 在 process 时指定新维度 (例如 448x448)
# messages = [{
#     'role': 'user', 
#     'content': [{
#         'image': url,
#         'resized_height': 448,
#         'resized_width': 448
#     }]
# }]

# image_inputs, _ = process_vision_info(messages)

# if image_inputs:
#     img = image_inputs[0]
#     img_np = np.array(img)
#     new_w, new_h = img.size
    
#     print(f'--- 维度变化分析 ---')
#     print(f'处理前 (原始尺寸): {orig_w}x{orig_h} (Wxh)')
#     print(f'处理后 (指定尺寸): {new_w}x{new_h} (Wxh)')
#     print(f'Numpy 数组形状 (H, W, C): {img_np.shape}')
#     print(f'数据类型: {img_np.dtype}')
# "


# PIL 原生$(W, H)$$(448, 392)$图像预处理、裁剪、缩放
# NumPy/OpenCV$(H, W, C)$$(392, 448, 3)$
# 矩阵运算、颜色空间转换PyTorch (NPU/GPU)$(C, H, W)$$(3, 392, 448)$深度学习模型输入
def get_image_info(image_path, min_pixel, max_pixel, width, height):
    # Using this because of process_vision_info function
    # Need to fix this in the future


    content = {
        "type": "image", 
        "image": image_path,
        "min_pixel": min_pixel,
        "max_pixel": max_pixel
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    messages = [
        {"role": "user", 
         "content": [content]
        }
    ]

    image_input, _ = process_vision_info(messages)

    # 取出的就是这个列表里的第一个（也是通常唯一的）元素，返回的还是图片形式
    return image_input[0]




# 视频信息处理
def get_video_info(video_path, min_pixels, max_pixels, fps):
    # Using this because of process_vision_info function
    # Need to fix this in the future

    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "video", 
                "video": video_path,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "fps": fps
            }
            ]
        }
    ]

    _, video_input, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    return video_input[0], video_kwargs


def add_anchor_pad(user_input, anchor_nums, anchor_tokens):
    # add anchor pad after VISION_END_TOKEN or ANCHOR_END_TOKEN
    # 将各类的anchor pad token 加入user_input中
    anchor_pads = []
    for anchor_num, anchor_token in zip(anchor_nums, anchor_tokens):
        anchor_pad = ANCHOR_START_TOKEN + anchor_token * anchor_num + ANCHOR_END_TOKEN
        anchor_pads.append(anchor_pad)
    anchor_pads = "".join(anchor_pads)
    if VISION_END_TOKEN in user_input:
        user_input = user_input.replace(VISION_END_TOKEN, VISION_END_TOKEN + anchor_pads)
    return user_input


# 示例："The apples of the image is [△△△], and the oranges of the image is [○○]. How many fruits are there?
def add_cot_anchor_pad_in_user_input(user_input, anchor_nums, anchor_tokens):
    if len(anchor_nums) == 0:
        return user_input
    
    anchor_pads = []
    for anchor_num, anchor_token in zip(anchor_nums, anchor_tokens, anchor_names):
        anchor_pad = ANCHOR_START_TOKEN + anchor_token * anchor_num + ANCHOR_END_TOKEN
        anchor_pads.append(anchor_pad)
    CoT_pad = ""
    # 自然语言描述，加入input中
    # 多个锚点："The [特征名1] of the image is [锚点块1], the [特征名2] of the image is [锚点块2], 
    # and the [特征名3] of the image is [锚点块3]。
    if len(anchor_pads) == 1:
        CoT_pad = f"The {anchor_names[0]} of the image is {anchor_pads[0]}. "
    else:
        for i, (anchor_name, anchor_pad) in enumerate(zip(anchor_names, anchor_pads)):
            if i == 0:
                CoT_pad += f"The {anchor_name} of the image is {anchor_pad}, "
            elif i == len(anchor_names) - 1:
                CoT_pad += f"and the {anchor_name} of the image is {anchor_pad}. "
            else:
                CoT_pad += f"the {anchor_name} of the image is {anchor_pad}, "
    user_input = CoT_pad + user_input
    return user_input


# 在模型响应中添加基于锚点的推理
# "Because the apples of the image is [△△△], and the oranges of the image is [○○]. So there are 5 fruits in total."
def get_cot_data_in_response(response, anchor_nums, anchor_tokens, anchor_names):
    if len(anchor_nums) == 0:
        return response
    
    anchor_pads = []
    for anchor_num, anchor_token in zip(anchor_nums, anchor_tokens):
        anchor_pad = ANCHOR_START_TOKEN + anchor_token * anchor_num + ANCHOR_END_TOKEN
        anchor_pads.append(anchor_pad)
    CoT_start = "Because "
    if len(anchor_names) == 1:
        CoT_start += f"the {anchor_names[0]} of the image is {anchor_pads[0]}. "
    else:
        for anchor_name, anchor_pad in zip(anchor_names, anchor_pads):
            CoT_start += f"the {anchor_name} of the image is {anchor_pad}"
            if anchor_name == anchor_names[-2]:
                CoT_start += ", and "
            elif anchor_name == anchor_names[-1]:
                CoT_start += ". "
            else:
                CoT_start += ", "
    response = CoT_start + response
    return response


COT_TEMPLATES = [
    {
        "name": "basic_causal",
        "single": "Because the {anchor_name} of the image is {anchor_pad}. ",
        "multiple": "Because the {anchor_name} of the image is {anchor_pad}{connector}",
        # 多个特征时的连接词逻辑
        "connectors": {
            "middle": ", ",
            "second_last": ", and ",
            "last": ". "
        }
    },
    
    {
        "name": "observational",
        "single": "I can observe that the {anchor_name} of the image is {anchor_pad}. ",
        "multiple": "I can observe that the {anchor_name} of the image is {anchor_pad}{connector}",
        "connectors": {
            "middle": ", ",
            "second_last": ", and ",
            "last": ". "
        }
    },
    
    {
        "name": "analytical",
        "single": "After analyzing the image, the {anchor_name} is {anchor_pad}. ",
        "multiple": "After analyzing the image, the {anchor_name} is {anchor_pad}{connector}",
        "connectors": {
            "middle": ", ",
            "second_last": ", and ",
            "last": ". "
        }
    },
    
    {
        "name": "descriptive",
        "single": "The image shows that the {anchor_name} is {anchor_pad}. ",
        "multiple": "The image shows that the {anchor_name} is {anchor_pad}{connector}",
        "connectors": {
            "middle": ", ",
            "second_last": ", and ",
            "last": ". "
        }
    },
    
    {
        "name": "conditional",
        "single": "Given that the {anchor_name} of the image is {anchor_pad}. ",
        "multiple": "Given that the {anchor_name} of the image is {anchor_pad}{connector}",
        "connectors": {
            "middle": ", ",
            "second_last": ", and ",
            "last": ". "
        }
    },
    
    {
        "name": "evidence_based",
        "single": "Based on the visual evidence, the {anchor_name} of the image is {anchor_pad}. ",
        "multiple": "Based on the visual evidence, the {anchor_name} of the image is {anchor_pad}{connector}",
        "connectors": {
            "middle": ", ",
            "second_last": ", and ",
            "last": ". "
        }
    }
]

# 随机选择上述的一个cot模版
def get_random_cot_template():
    return random.choice(COT_TEMPLATES)


# 使用cot模版，将pad token填充进去
def apply_cot_template(template, anchor_names, anchor_pads):
    if len(anchor_names) == 1:
        return template["single"].format(
            anchor_name=anchor_names[0], 
            anchor_pad=anchor_pads[0]
        )
    else:
        result = ""
        for i, (anchor_name, anchor_pad) in enumerate(zip(anchor_names, anchor_pads)):
            if i == len(anchor_names) - 1: 
                connector = template["connectors"]["last"]
            elif i == len(anchor_names) - 2: 
                connector = template["connectors"]["second_last"]
            else: 
                connector = template["connectors"]["middle"]
            
            result += template["multiple"].format(
                anchor_name=anchor_name,
                anchor_pad=anchor_pad,
                connector=connector
            )
        return result

# 结合上述构造cot内容，将答案也加进去，构造完整的格式。即"<think>" + cot_text + "</think>" + "<answer>" + response + "</answer>"
def get_templates_comt_data_in_response(response, anchor_nums, anchor_tokens, anchor_names):
    if len(anchor_nums) == 0:
        return response
    
    anchor_pads = []
    for anchor_num, anchor_token in zip(anchor_nums, anchor_tokens):
        anchor_pad = ANCHOR_START_TOKEN + anchor_token * anchor_num + ANCHOR_END_TOKEN
        anchor_pads.append(anchor_pad)
    
    template = get_random_cot_template()
    
    cot_text = apply_cot_template(template, anchor_names, anchor_pads)
    
    response = "<think>" + cot_text + "</think>" + "<answer>" + response + "</answer>"
    return response
        

# 不随机使用模版，构造固定的cot推理结构
def get_comt_data_in_response(response, anchor_nums, anchor_tokens, anchor_names):
    if len(anchor_nums) == 0:
        return response
    
    anchor_pads = []
    for anchor_num, anchor_token in zip(anchor_nums, anchor_tokens):
        anchor_pad = ANCHOR_START_TOKEN + anchor_token * anchor_num + ANCHOR_END_TOKEN
        anchor_pads.append(anchor_pad)
    CoT_start = "<think> Because "
    if len(anchor_names) == 1:
        CoT_start += f"the {anchor_names[0]} of the image is {anchor_pads[0]}. "
    else:
        for anchor_name, anchor_pad in zip(anchor_names, anchor_pads):
            CoT_start += f"the {anchor_name} of the image is {anchor_pad}"
            if anchor_name == anchor_names[-2]:
                CoT_start += ", and "
            elif anchor_name == anchor_names[-1]:
                CoT_start += ". "
            else:
                CoT_start += ", "
    response = CoT_start + " </think>\n" + "<answer> " + response + " </answer>"
    return response
    
# 将视觉任务的特征描述转换为特定的对话格式，用于训练或评估多模态模型。它创建了一个模拟的对话场景，其中模型需要识别图像中的特定特征。
# 提问各个特征是什么，回答是各个anchor pad
def get_feature_data(user_input, gpt_response, anchor_nums, anchor_tokens, anchor_names):
    anchor_pads = []
    for anchor_num, anchor_token, anchor_name in zip(anchor_nums, anchor_tokens, anchor_names):
        anchor_pad = ANCHOR_START_TOKEN + anchor_token * anchor_num + ANCHOR_END_TOKEN
        anchor_pads.append(anchor_pad)
    anchor_name = ", ".join(anchor_names)
    anchor_pads = "".join(anchor_pads)
    user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN}What is the {anchor_name} of the image?\n{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
    gpt_response = f"{anchor_pads}\n{DEFAULT_IM_END_TOKEN}\n"
    return user_input, gpt_response

def replace_pad_with_anchor_tokens(gpt_response):
    # 将response中的pad token替换为实际的特征token
    token_dict = {
        "<segmentation>": SAM_PAD_TOKEN * 8,
        "<depth>": DEPTH_PAD_TOKEN * 4,
        "<dino>": DINO_PAD_TOKEN * 4,
        "<pidinet>": PIDINET_PAD_TOKEN * 4,
        "<siglip>": SIGLIP_PAD_TOKEN * 4,
        "<metaclip>": METACLIP_PAD_TOKEN * 4,
    }
    for token, anchor_token in token_dict.items():
        gpt_response = gpt_response.replace(token, anchor_token)
    return gpt_response
    

# 不同的token对应不同数目，写死
def get_token_num(anchor_model_id):
    token_nums = []
    for anchor_model in anchor_model_id:
        if anchor_model == "sam":
            token_nums.append(8)
        elif anchor_model == "dino":
            token_nums.append(4)
        elif anchor_model == "depth":
            token_nums.append(4)
        elif anchor_model == "SD":
            token_nums.append(4)
        elif anchor_model == "InternViT":
            token_nums.append(4)
        elif anchor_model == "pidinet":
            token_nums.append(4)
        elif anchor_model == "siglip":
            token_nums.append(4)
        elif anchor_model == "metaclip":
            token_nums.append(4)
    return token_nums

# 根据不同anchor id得到对应的pad_token
def get_anchor_token(anchor_model_id):
    anchor_tokens = []
    for anchor_model in anchor_model_id:
        if anchor_model == "sam":
            anchor_tokens.append(SAM_PAD_TOKEN)
        elif anchor_model == "dino":
            anchor_tokens.append(DINO_PAD_TOKEN)
        elif anchor_model == "depth":
            anchor_tokens.append(DEPTH_PAD_TOKEN)
        elif anchor_model == "SD":
            anchor_tokens.append(SD_PAD_TOKEN)
        elif anchor_model == "InternViT":
            anchor_tokens.append(INTERN_PAD_TOKEN)
        elif anchor_model == "pidinet":
            anchor_tokens.append(PIDINET_PAD_TOKEN)
        elif anchor_model == "siglip":
            anchor_tokens.append(SIGLIP_PAD_TOKEN)
        elif anchor_model == "metaclip":
            anchor_tokens.append(METACLIP_PAD_TOKEN)
    return anchor_tokens

# 根据不同anchor id得到对应的任务名称
def get_anchor_task_name(anchor_model_id):
    anchor_task_names = []
    for anchor_model in anchor_model_id:
        if anchor_model == "sam":
            anchor_task_names.append("segmentation")
        elif anchor_model == "dino":
            anchor_task_names.append("perception feature")
        elif anchor_model == "depth":
            anchor_task_names.append("depth map")
        elif anchor_model == "SD":
            anchor_task_names.append("style")
        elif anchor_model == "InternViT":
            anchor_task_names.append("caption")
        elif anchor_model == "pidinet":
            anchor_task_names.append("edge map")
        elif anchor_model == "siglip":
            anchor_task_names.append("clip feature")
        elif anchor_model == "metaclip":
            anchor_task_names.append("metaclip feature")
    return anchor_task_names
    

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
        shuffle=True,
        random_seed=42,
        anchor_model_id=None,
    ):
        super(SupervisedDataset, self).__init__()
        # 加载数据：支持文件路径字符串或预加载的列表
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        # anchors、model、processor等基础信息
        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.fps = data_args.fps
        self.anchor_model_id = anchor_model_id
        self.anchor_token_nums = get_token_num(anchor_model_id)
        self.anchor_tokens = get_anchor_token(anchor_model_id)
        self.anchor_task_names = get_anchor_task_name(anchor_model_id)
        
        # 不同阶段的step数目
        self.cur_step = 0
        self.stage_0_step = data_args.stage_0_step
        self.stage_1_step = data_args.stage_1_step
        self.stage_2_step = data_args.stage_2_step
        
        # for shuffle
        # 数据打乱
        self.rng = np.random.default_rng(seed=random_seed)
        
        if shuffle:
            self.rng.shuffle(self.list_data_dict)
    
    # 设定当前steps
    def set_cur_step(self, step: int):
        self.cur_step = step
        print(f"[Dataset] cur_step has been set to {step}")
        
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        
        # import ipdb; ipdb.set_trace()
        
        sources = self.list_data_dict[i]
        
        is_video = False

        processor = self.processor
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            # 统一转换成 PIL Image 列表
            if isinstance(image_files, str):
                image_files = Image.open(image_files).convert("RGB")
                image_files = [image_files]
            else:
                image_files = [Image.open(image_file).convert("RGB") for image_file in image_files]

            images = []
            
            for image_file in image_files:
                
                # if not os.path.exists(image_file):
                #     if not image_file.startswith("http"):
                #         image_file = os.path.join(image_folder, image_file)
                #         images.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

                # else:
                # 每张图片都通过get_image_info（process_vision_info）进行处理，如resize等操作
                images.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))
        
        # 对于视频分支处理
        elif "video" in sources:
            is_video = True
            images=None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            videos = []
            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(video_file, self.video_min_pixel, self.video_max_pixel, self.data_args.fps)
                videos.append(video_input)
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None
           
        # images为空的时候添加全黑占位；videos为空时跳过
        if images is None:
            print("No image or video found in the data.")
            images = []
            # Create a black image as a placeholder
            black_image = Image.new("RGB", (self.image_resized_w, self.image_resized_h), (0, 0, 0))
            images.append(get_image_info(black_image, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

        elif len(images) == 0:
            print("No image or video found in the data.")
            # Create a black image as a placeholder
            black_image = Image.new("RGB", (self.image_resized_w, self.image_resized_h), (0, 0, 0))
            images.append(get_image_info(black_image, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))
        
        if videos is not None:
            # import ipdb; ipdb.set_trace()
            pass
            
        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))

        all_input_ids = []        # 存储所有token ID
        all_labels = []           # 存储所有标签
        all_pixel_values = []     # 存储图像像素值
        all_image_grid_thw = []   # 存储图像网格特征
        all_second_gird = []      # 存储视频时间信息
        # all_dino_encoded_values = []

        # Qwen2-VL uses a default system message so I've added this.
        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}\n{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            # 将系统消息的标签设为IGNORE_INDEX（不参与损失计算）
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
            # 移除第一个维度
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))
            

        # 按用户-助手对话对处理数据（每次处理一对）
        for _, j in enumerate(range(0, len(sources), 2)):
            
            # 目前只考虑单轮对话情况，跳过多轮
            if j >= 2:
                break
            
            # 原始的input和response，每个元素可能是字典，包含 'role' 和 'content' 字段
            user_input = sources[j]
            gpt_response = sources[j + 1]
                        
            if (DEFAULT_IMAGE_TOKEN not in user_input['content']) and (DEFAULT_VIDEO_TOKEN not in user_input['content']) and (LLAVA_IMAGE_TOKEN in user_input['content']):
                user_input = f"{DEFAULT_IM_START_TOKEN}{VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                # 在vision_end token后面添加anchor_pad token
                user_input = add_anchor_pad(user_input, self.anchor_token_nums, self.anchor_tokens)
                gpt_response = f"{gpt_response['content']}\n{DEFAULT_IM_END_TOKEN}\n"
                raise ValueError('Every man is a poet when he is in love')
            else:
                #根据 self.cur_step 动态修改 user_input 和 gpt_response
                # 阶段 0 (简单 SFT):
                # 在用户输入的视觉 Token 后注入 anchor_pads（即占位符）。
                # gpt_response 就是普通的答案。
                # 阶段 1 (特征对齐):
                # 强制修改用户问题为：“图像的 [特征名] 是什么？”
                # 强制修改回答为：对应的 anchor_pads。这让模型学会将视觉特征编码进这些特殊 token。
                # 阶段 2 (隐式推理 CoT):
                # 使用 get_comt_data_in_response。
                # 格式变更为： <think> Because the segmentation of the image is [PAD]... </think> <answer> Final Answer </answer>。
                # 最终阶段 (随机混合):
                # 随机选择是进行普通问答，还是进行带 think 标签的推理。
                if self.cur_step < self.stage_0_step:
                    # 第一阶段，在原始的input后面add_anchor_pad，添加各个pad token，答案不变
                    user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                    user_input = add_anchor_pad(user_input, self.anchor_token_nums, self.anchor_tokens)
                    gpt_response = f"{gpt_response['content']}\n{DEFAULT_IM_END_TOKEN}\n"
                elif self.cur_step < self.stage_1_step:
                    # 第二阶段，直接询问各个的pad token是什么，回答对应的token
                    user_input, gpt_response = get_feature_data(user_input, gpt_response, self.anchor_token_nums, self.anchor_tokens, self.anchor_task_names)
                elif self.cur_step < self.stage_2_step:
                    # 第三阶段，问题不变。回答的时候先think，要求给出中间的特殊token特征
                    user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                    gpt_response = f"{gpt_response['content']}"
                    if DEFAULT_IMAGE_TOKEN in user_input:
                        gpt_response = get_comt_data_in_response(gpt_response, self.anchor_token_nums, self.anchor_tokens, self.anchor_task_names)
                    gpt_response = f"{gpt_response}\n{DEFAULT_IM_END_TOKEN}\n"
                    # print(f"\033[92m gpt_response: {gpt_response}\033[0m")
                else:
                    # user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                    # gpt_response = f"{gpt_response['content']}\n{DEFAULT_IM_END_TOKEN}\n"
                    # gpt_response = replace_pad_with_anchor_tokens(gpt_response)
                    # 第四阶段
                    import random
                    xxx = random.randint(0, 5)
                    if xxx == 0:
                        # 标准问答形式
                        user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                        gpt_response = f"{gpt_response['content']}\n{DEFAULT_IM_END_TOKEN}\n"
                    else:
                        user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
                        gpt_response = f"{gpt_response['content']}"
                        if DEFAULT_IMAGE_TOKEN in user_input:
                            # INSERT_YOUR_CODE
                            total = len(self.anchor_tokens)
                            if total == 0:
                                # 如果一个专家token都不使用
                                selected_anchor_token_nums = []
                                selected_anchor_tokens = []
                                selected_anchor_task_names = []
                            else:
                                # 随机从使用的专家token中选择
                                x = random.randint(1, total)
                                # 无放回地随机抽取x个不同的索引，从小到大排序
                                idxs = sorted(random.sample(range(total), x)) if x > 0 else []
                                selected_anchor_token_nums = [self.anchor_token_nums[i] for i in idxs]
                                selected_anchor_tokens = [self.anchor_tokens[i] for i in idxs]
                                selected_anchor_task_names = [self.anchor_task_names[i] for i in idxs]
                            # 构造特殊的response
                            gpt_response = get_comt_data_in_response(gpt_response, selected_anchor_token_nums, selected_anchor_tokens, selected_anchor_task_names)
                        gpt_response = f"{gpt_response}\n{DEFAULT_IM_END_TOKEN}\n"
            
            # print(f'the user_input is {user_input}')
            # print(f'the gpt_response is {gpt_response}')
                    

            # print("-----------------")
            # print(user_input, gpt_response)
            # print("-----------------")
            
            # import ipdb; ipdb.set_trace()
            
            # 图像分支经过processor
            if DEFAULT_IMAGE_TOKEN in user_input:
                inputs = processor(text=[user_input], images=images, videos=videos, padding=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                # raise ValueError('Every man is a poet when he is in love')
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                
                # del dino_val
                torch.cuda.empty_cache()
                
            # 视频分支经过processor
            elif DEFAULT_VIDEO_TOKEN in user_input:
                if "Qwen2.5" in self.model_id:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, return_tensors='pt', **video_kwargs)
                    all_second_gird.extend(inputs["second_per_grid_ts"])
                else:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])

            else:
                # 纯文本
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            # 将prompt和response拼接
            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        # # 拼接所有对话片段
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

        attention_mask = (input_ids > -1000000).to(torch.long)
        
        
        
        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # 拼接后的像素矩阵、各图/帧的尺寸元数据
        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw
            # 原始文件路径/对象
            data_dict["image_files"] = image_files

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        self.cur_step += 1
        
        return data_dict

# 将一个 Batch（批次） 的样本整理成模型可以直接读取的格式
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    # 当 DataLoader 加载数据时会调用它。它接收一个 examples 列表（每个元素是一个样本字典），并返回一个处理后的 data_dict
    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        
        batch_image_files = []
        
        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
            
            if "image_files" in keys:
                batch_image_files.append(example["image_files"])
            
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

        # 将input和label进行padding操作   
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        # 非pad的地方，掩码为1；防止计算填充位置（掩码为0）的注意力权重
        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts
            
        if len(batch_image_files) > 0:
            data_dict["image_files"] = batch_image_files

        return data_dict
    
# 将 LLaVA 特有的图像/视频标记替换为通用格式的标记
def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r'\n?' + re.escape(LLAVA_VIDEO_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN

    return re.sub(pattern, replacement, input_string)


# 将 LLaVA 格式的对话数据转换为 OpenAI API 兼容的格式
# 主要转换：
# 角色映射（human→user, gpt→assistant）
# 图像/视频标记替换
def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data


# SupervisedDataset：定义“怎么读数据”。
# DataCollator：定义“怎么组合数据成 Batch”。
# 输出：一个字典。这个字典通常直接传给 HuggingFace 的 Trainer 类
def make_supervised_data_module(model_id, processor, data_args, anchor_model_id):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id, anchor_model_id=anchor_model_id
    )
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)