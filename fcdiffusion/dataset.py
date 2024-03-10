import json
import cv2
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('datasets/training_data.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img_path = item['img_path']
        prompt = item['prompt']

        img = cv2.imread(img_path)

        # resize img to 512 x 512
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

        # Do not forget that OpenCV read images in BGR order.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize images to [-1, 1].
        img = (img.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=img, txt=prompt, path=img_path.split('6.5')[1])


class TestDataset(Dataset):
    def __init__(self, img_path: str, prompt: str, res_num: int):
        self.data = []
        self.img_path = img_path
        self.prompt = prompt
        self.res_num = res_num

        for i in range(self.res_num):
            self.data.append({'jpg': img_path,
                              'txt': prompt})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img_path = item['jpg']
        prompt = item['txt']

        img = cv2.imread(img_path)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=img, txt=prompt, path=img_path)

