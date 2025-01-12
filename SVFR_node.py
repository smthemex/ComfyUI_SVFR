# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import os
import torch
from .infer import main_loader,main_sampler
from .node_utils import nomarl_upscale,tensor_upscale

import folder_paths

MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
current_path = os.path.dirname(os.path.abspath(__file__))

weigths_SVFR_current_path = os.path.join(folder_paths.models_dir, "SVFR")
if not os.path.exists(weigths_SVFR_current_path):
    os.makedirs(weigths_SVFR_current_path)

try:
    folder_paths.add_model_folder_path("SVFR", weigths_SVFR_current_path, False)
except:
    folder_paths.add_model_folder_path("SVFR", weigths_SVFR_current_path)


class SVFR_LoadModel:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        yolo_ckpt_list = [i for i in folder_paths.get_filename_list("SVFR") if
                          "yolo" in i]
        insightface_ckpt_list = [i for i in folder_paths.get_filename_list("SVFR") if
                          "insightface" in i]
        unet_ckpt_list = [i for i in folder_paths.get_filename_list("SVFR") if
                                 "unet" in i]
        id_ckpt_list = [i for i in folder_paths.get_filename_list("SVFR") if
                                 "id" in i]
        return {
            "required": {
                "I2V_repo": ("STRING", {"default": "F:/test/ComfyUI/models/diffusers/stabilityai/stable-video-diffusion-img2vid-xt-1-1"}),
                "unet": (["none"] + unet_ckpt_list,),
                "yolo_ckpt": (["none"] + yolo_ckpt_list,),
                "id_ckpt": (["none"] + id_ckpt_list,),
                "insightface": (["none"] + insightface_ckpt_list,),
                "dtype": (["fp16","bf16","fp32"],),
            }
        }
    
    RETURN_TYPES = ("MODEL_SVFR",)
    RETURN_NAMES = ("model",)
    FUNCTION = "main_loader"
    CATEGORY = "SVFR"
    
    def main_loader(self,I2V_repo, unet, yolo_ckpt, id_ckpt, insightface,dtype):
        
        if dtype == "fp16":
            weight_dtype = torch.float16
        elif dtype == "fp32":
            weight_dtype = torch.float32
        else:
            weight_dtype = torch.bfloat16
            
        if unet == "none" or yolo_ckpt == "none" or id_ckpt == "none" or insightface == "none":
           raise "need choice  ckpt in menu"
        else:
            unet_path=folder_paths.get_full_path("SVFR",unet)
            det_path=folder_paths.get_full_path("SVFR",yolo_ckpt)
            id_path=folder_paths.get_full_path("SVFR",id_ckpt)
            face_path=folder_paths.get_full_path("SVFR",insightface)
            pipe,id_linear,net_arcface,align_instance=main_loader(weight_dtype,I2V_repo,unet_path,det_path,id_path,face_path,device)
        print("****** Load model is done.******")
        return (
        {"pipe": pipe, "id_linear": id_linear, "net_arcface": net_arcface, "align_instance": align_instance, "weight_dtype": weight_dtype},)


class SVFR_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        
        return {
            "required": {
                "image": ("IMAGE",),  # [B,H,W,C], C=3,B>1
                "model": ("MODEL_SVFR",),
                "seed": ("INT", {"default": 77, "min": 0, "max": MAX_SEED}),
                "width": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64, "display": "number"}),
                "decode_chunk_size": ("INT", {"default":16, "min": 4, "max": 128, "step": 4,}),
                "n_sample_frames": ("INT", {"default": 16, "min": 8, "max": 100, "step": 1,}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 4096, "step": 1, "display": "number"}),
                "noise_aug_strength": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 1.00, "step": 0.01, "round": 0.001}),
                "overlap": ("INT", {"default":3, "min": 1, "max": 64}),
                "min_appearance_guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1, "round": 0.01}),
                "max_appearance_guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1, "round": 0.01}),
                "i2i_noise_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1, "round": 0.01}),
                "infer_mode":(["bfr","colorization","inpainting","bfr_color","bfr_color_inpaint"],),
                "save_video": ("BOOLEAN", {"default": False},),
            },
            "optional": {"mask": ("MASK",),
                         },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "sampler_main"
    CATEGORY = "SVFR"
    
    def sampler_main(self, image, model, seed, width, height,decode_chunk_size,n_sample_frames
                     , steps,noise_aug_strength, overlap,min_appearance_guidance_scale,max_appearance_guidance_scale, i2i_noise_strength,infer_mode,save_video,**kwargs):
        
        pipe = model.get("pipe")
        id_linear = model.get("id_linear")
        net_arcface = model.get("net_arcface")
        align_instance = model.get("align_instance")
        weight_dtype = model.get("weight_dtype")
        mask=kwargs.get("mask")
        
        if isinstance(mask,torch.Tensor): #缩放至图片尺寸
            if mask.shape[-1]==64 and mask.shape[-2]==64:
                raise "input mask is not a useful,looks like a default comfyUI mask"

            if len(mask.shape) == 3:  # 1,h,w
                mask_array = mask.squeeze().mul(255).clamp(0, 255).byte().numpy()
            elif len(mask.shape)==2 and mask.shape[0]!=1:  # h,w
                mask_array=mask.mul(255).clamp(0, 255).byte().numpy()
            else:
                raise "check input mask's shape"
            mask_array=np.where(mask_array > 0, 255, 0)
        else:
            mask_array=None
        video_len, _, _, _ = image.size()
        if video_len < 8:
            raise "input video has not much frames below 8 frame,change your input video!"
        else:
            tensor_list = list(torch.chunk(image, chunks=video_len))
            input_frames_pil = [nomarl_upscale(i,width, height) for i in tensor_list]  # tensor to pil
        
        if infer_mode=="bfr":
            task_ids=[0]
        elif infer_mode=="colorization":
            task_ids = [1]
        elif infer_mode=="inpainting":
            task_ids = [2]
        elif infer_mode=="bfr_color":
            task_ids = [0,1]
        else:
            task_ids = [0, 1,2 ]
        
        if not isinstance(mask_array,np.ndarray) and 2 in task_ids:
            raise "If use inpainting need link a mask or a batch mask in the front."
        
        print("******** start infer *********")
        images=main_sampler(pipe, align_instance, net_arcface, id_linear, folder_paths.get_output_directory(), weight_dtype,
                          seed,input_frames_pil,task_ids,mask_array,save_video,decode_chunk_size,noise_aug_strength,
                          min_appearance_guidance_scale,max_appearance_guidance_scale,
                          overlap,i2i_noise_strength,steps,n_sample_frames,device)
        
        #model.to("cpu")#显存不会自动释放，手动迁移，不然很容易OOM
        torch.cuda.empty_cache()
        return (images,)


class SVFR_img2mask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),  # [B,H,W,C], C=3,B=1
                "threshold": ("INT", {"default": 0, "min": 0, "max": 254, "step": 0, "display": "number"}),
                "center_crop": ("BOOLEAN", {"default": False},),
                "width": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64, "display": "number"}),
        }
        }
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "main"
    CATEGORY = "SVFR"
    
    def main(self, image,threshold,center_crop,width,height):
        if center_crop:
            image=tensor_upscale(image, width, height)
        np_img=image.squeeze().mul(255).clamp(0, 255).byte().numpy()
        np_img=np.mean(np_img, axis=2).astype(np.uint8)
        
        black_threshold = 50  # 黑色阈值，小于这个值的像素被认为是黑色
        white_threshold = 200  # 白色阈值，大于这个值的像素被认为是白色
        black_pixels = np.sum(np_img < black_threshold)    # 计算黑色像素的数量
        white_pixels = np.sum(np_img > white_threshold) # 计算白色像素的数量
        if black_pixels>white_pixels: #黑多白少，按白色为遮罩
            out = np.where(np_img > threshold, 255, 0).astype(np.float32) / 255.0
        else:
            out = np.where(np_img > threshold, 0, 255).astype(np.float32) / 255.0
        return (torch.from_numpy(out).unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "SVFR_LoadModel": SVFR_LoadModel,
    "SVFR_Sampler": SVFR_Sampler,
    "SVFR_img2mask":SVFR_img2mask,
    
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SVFR_LoadModel": "SVFR_LoadModel",
    "SVFR_Sampler": "SVFR_Sampler",
    "SVFR_img2mask":"SVFR_img2mask"
}
