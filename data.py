import torch
import torch.utils.data as data
import os
import numpy as np
from PIL import Image
from vist import VIST
from transformers import AutoTokenizer, AutoProcessor

tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
processor_clip = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")  
tokenizer_clip = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")  

class VistDataset(data.Dataset):
    def __init__(self, image_dir, sis_path):
        self.image_dir = image_dir
        self.vist = VIST(sis_path)
        self.ids = list(self.vist.stories.keys())
        
    def __getitem__(self, index):
        vist = self.vist
        story_id = self.ids[index]

        sentence = []
        images = []
        order = []

        story = vist.stories[story_id]
        image_formats = ['.jpg', '.gif', '.png', '.bmp']
        for annotation in story:
            image = Image.new('RGB', (224, 224))
            image_id = annotation["photo_flickr_id"]
            order.append(annotation["worker_arranged_photo_order"])

            for image_format in image_formats:
                try:
                    image = Image.open(os.path.join(self.image_dir, str(image_id) + image_format)).convert('RGB')
                except Exception:
                    continue
            
            image_pro = processor_clip(images=image, return_tensors='pt')
            image_p = image_pro.pixel_values
            images.append(image_p.squeeze(0))
            text = annotation["text"]
            sentence.append(text)

        return torch.stack(images), sentence

    def __len__(self):
        return len(self.ids) 

def collate_fn(data):
    image_stories, caption_stories = zip(*data)
    captions_set = []
    captions_set_clip = []

    for captions in caption_stories:
        word_ids = tokenizer_bert(captions, padding='longest', max_length = 40, truncation=True, return_tensors="pt")
        captions_set.append(word_ids)
    
    for captions in caption_stories:
        word_ids = tokenizer_clip(captions, padding='longest', max_length = 40, truncation=True, return_tensors="pt")
        captions_set_clip.append(word_ids)

    return image_stories, captions_set, captions_set_clip

def get_loader(root, sis_path, batch_size, shuffle, num_workers):
    vist = VistDataset(image_dir=root, sis_path=sis_path)
    data_loader = torch.utils.data.DataLoader(dataset=vist, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader