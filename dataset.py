import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import captcha_util

class CaptchaDataset(Dataset):
    def __init__(self, mode=None, data_dir=None, img_height=32, img_width=100):
        if mode == 'train' and not data_dir:
            data_dir = 'data/train/'
        elif mode == 'valid' and not data_dir:
            data_dir = 'data/valid/'
        elif mode == 'pred' and not data_dir:
            data_dir = 'data/pred/'
        
        self.paths = []
        self.texts = []
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self._load_from_raw_files(mode)

    def _load_from_raw_files(self, mode):
        if type(self.data_dir) is list:
            for _data_dir in self.data_dir:
                self.paths.extend(glob.glob(_data_dir + '*'))
            assert len(self.paths) != 0, '請確認資料夾內檔案是否存在 %s' % mode
        else:
            self.paths = glob.glob(self.data_dir + '*')
            assert len(self.paths) != 0, '請確認資料夾內檔案是否存在 %s' % mode
        
        if mode == 'train':
            print('此次訓練用 %d 張' %(len(self.paths)))
        elif mode == 'valid':
            print('此次檢驗用 %d 張' %(len(self.paths)))
        elif mode == 'pred':
            print('此次預測用 %d 張' %(len(self.paths)))

        for imagePath in self.paths:
            self.texts.append(imagePath.split('\\')[-1].split('.')[0])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        paths = self.paths[index]

        try:
            image = Image.open(paths)
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        #需改用CV2讀圖，可用可不用。
        #preprocessedImage = captcha_util.preprocess(imageOri)
        
        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((3, self.img_height, self.img_width))

        image = torch.FloatTensor(image)
        
        text = ""
        if self.texts:
            text = self.texts[index]
            target = [captcha_util.CHAR2LABEL[c] for c in text]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image

def CaptchaDatasetFn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths
