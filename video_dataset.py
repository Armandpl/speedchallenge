import os
import os.path
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from collections.abc import Callable

class VideoFrameDataset(torch.utils.data.Dataset):

    def __init__(self,
                 root_path: str,
                 seq_len: int,
                 step: int,
                 skip_frames = 8,
                 imagefile_template: str='{:05d}.jpg',
                 transform = None):
        super(VideoFrameDataset, self).__init__()

        self.root_path = root_path
        self.imagefile_template = imagefile_template
        self.transform = transform
        self.seq_len = seq_len
        self.step = step
        self.skip_frames = skip_frames

        self._parse_list()

    # def _load_image(self, directory, idx):
        # return [Image.open(os.path.join(directory, self.imagefile_template.format(idx))).convert('RGB')]

    def _parse_list(self):
        frame_list = []

        with open(os.path.join(self.root_path, "annotations.txt")) as f:
            annotations = f.readlines()

        image_nb = 0
        for fname in os.listdir(self.root_path):
            if fname.endswith(".jpg"):
                image_nb += 1
        
        for i in range(image_nb):
            frame_list.append(os.path.join(self.root_path, str(i).zfill(5)+".jpg"))

        if len(frame_list) > len(annotations):
            print(len(frame_list), " frames, ", len(annotations), " annotations")
            raise IndexError

        self.sample_list = []

        def keep_1_frames_every_y(frames, y):
            res = []
            
            i = 0
            for f in frames:
                i+=1
                if (i == y):
                    res.append(f)
                    i = 0
            
            return res

        for i in range(0, len(frame_list)-self.seq_len*self.skip_frames, self.step):
            
            frames = frame_list[i:i+self.seq_len*self.skip_frames]
            frames = keep_1_frames_every_y(frames, self.skip_frames)

            item = (frames, float(annotations[i+self.seq_len*self.skip_frames]))
            self.sample_list.append(item)

    def __getitem__(self, index):
        # load images in list
        sample = self.sample_list[index]

        images = []
        for pth in sample[0]:
            images.append(Image.open(pth))

        # if self.transform is not None:
        #    images = self.transform(images)
        # images = ImglistToTensor().forward(images)
        tfms = transforms.Compose([
            transforms.Resize((224, 224)),
        ])
        images = [tfms(pic) for pic in images]
        if self.transform is not None:
            images = [self.transform(pic) for pic in images]

        images = [transforms.functional.to_tensor(pic) for pic in images]

        label = sample[1]
        images = torch.stack(images, 0).permute(1, 0, 2, 3)
        # images = torch.stack(images, 0).permute(0, 1, 2, 3)
        return images, label

    def __len__(self):
        return len(self.sample_list)

class ImglistToTensor(torch.nn.Module):
    """
    Converts a list of PIL images in the range [0,255] to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1].
    Can be used as first transform for ``VideoFrameDataset``.
    """
    def forward(self, img_list):
        """
        Converts each PIL image in a list to
        a torch Tensor and stacks them into
        a single tensor.
        Args:
            img_list: list of PIL images.
        Returns:
            tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])