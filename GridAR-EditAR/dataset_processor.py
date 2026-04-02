import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
import json
import torch


def resolve_mapping_file_path(dataset_path):
    candidate = os.path.join(dataset_path, "mapping_file.json")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Missing mapping file in {dataset_path}")


class PIE_Bench_Dataset(Dataset):
    '''
    '''
    def __init__(self,
                 args,
                 dataset_path,
                 sample_index=None,
                 ):

        self.args = args

        self.dataset_path = dataset_path
        with open(resolve_mapping_file_path(dataset_path), 'r') as file:
            self.dataset = json.load(file)
        
        if sample_index is not None:
            if sample_index in self.dataset:
                self.dataset = {sample_index: self.dataset[sample_index]}
            elif len(sample_index) == 1:
                self.dataset = {idx: self.dataset[idx] for idx in self.dataset if idx.startswith(sample_index)}
                
            
        self.mapping = {}
        for idx, key in enumerate(self.dataset.keys()):
            self.mapping[idx] = key
        


    def __len__(self,):
        return len(self.dataset)

    def _whiten_transparency(self, img: PIL.Image) -> PIL.Image:
        # Check if it's already in RGB format.
        if img.mode == "RGB":
            return img

        vals_rgba = np.array(img.convert("RGBA"))

        # If there is no transparency layer, simple convert and return.
        if not (vals_rgba[:, :, 3] < 255).any():
            return img.convert("RGB")

        # There is a transparency layer, blend it with a white background.

        # Calculate the alpha proportion for blending.
        alpha = vals_rgba[:, :, 3] / 255.0
        # Blend with white background.
        vals_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[
            :, :, np.newaxis
        ] * vals_rgba[:, :, :3]
        return PIL.Image.fromarray(vals_rgb.astype("uint8"), "RGB")

    def _vqgan_input_from(self, img: PIL.Image, target_image_size=512) -> torch.Tensor:
        # Resize with aspect ratio preservation.
        s = min(img.size)
        scale = target_image_size / s
        new_size = (round(scale * img.size[0]), round(scale * img.size[1]))
        img = img.resize(new_size, PIL.Image.LANCZOS)

        # Center crop.
        x0 = (img.width - target_image_size) // 2
        y0 = (img.height - target_image_size) // 2
        img = img.crop((x0, y0, x0 + target_image_size, y0 + target_image_size))

        # Convert to tensor.
        np_img = np.array(img) / 255.0  # Normalize to [0, 1]
        np_img = np_img * 2 - 1  # Scale to [-1, 1]
        tensor_img = (
            torch.from_numpy(np_img).permute(2, 0, 1).float()
        )  # (Channels, Height, Width) format.

        # Add batch dimension.
        return tensor_img

    def mask_decode(self, encoded_mask, image_shape=[512,512]):
        length=image_shape[0]*image_shape[1]
        mask_array=np.zeros((length,))
        
        for i in range(0,len(encoded_mask),2):
            splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
            for j in range(splice_len):
                mask_array[encoded_mask[i]+j]=1
                
        mask_array=mask_array.reshape(image_shape[0], image_shape[1])
        # to avoid annotation errors in boundary
        mask_array[0,:]=1
        mask_array[-1,:]=1
        mask_array[:,0]=1
        mask_array[:,-1]=1
                
        return mask_array

    def __getitem__(self, index):

        # ['image_path', 'original_prompt', 'editing_prompt', 'editing_instruction', 'editing_type_id', 'blended_word', 'mask']
        
        data_path = self.dataset[self.mapping[index]]['image_path']

        input_img = Image.open(os.path.join(self.dataset_path, 'annotation_images', data_path)).convert('RGB')
        _input_img = input_img
        _mask = self.mask_decode(self.dataset[self.mapping[index]]["mask"])
        _mask = _mask[:,:,np.newaxis].repeat([3],axis=2)
        _original_prompt = self.dataset[self.mapping[index]]["original_prompt"].replace("[", "").replace("]", "")
        _editing_prompt = self.dataset[self.mapping[index]]["editing_prompt"].replace("[", "").replace("]", "")

        edit_txt = self.dataset[self.mapping[index]]['editing_instruction']

        input_img = self._whiten_transparency(input_img)
        input_img = self._vqgan_input_from(input_img)
        edited_img = - torch.ones(input_img.shape)

        save_input_img = (input_img.permute(1,2,0)+1)/2 * 255.
        save_input_img = Image.fromarray(np.array(save_input_img).astype(np.uint8))

        return {
                'index': index,
                'image_path': data_path,
                'dataset': 'PIE',
                'mode': 1,
                'input_img': input_img,
                'edited_img': edited_img,
                '_input_img': np.array(_input_img),
                '_original_prompt': _original_prompt,
                '_editing_prompt': _editing_prompt,
                '_mask': np.array(_mask),
                '_edit_txt': edit_txt,
                }
