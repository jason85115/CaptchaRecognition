import config
from glob import glob
import numpy as np
import shutil
import os 
import captcha_util
from tqdm import tqdm

# 參數分別為 95%資料為訓練集 5%資料為檢驗集，validPrepared = true 時代表已經手動將檢驗集分出則將train設定為100%
def split_images(train = 0.95, valid = 0.05, validPrepared = False):
    if not os.path.exists(config.common_config['data_dir']):
        os.mkdir(config.common_config['data_dir'])
    if not os.path.exists(config.common_config['valid_data_dir']):
        os.mkdir(config.common_config['valid_data_dir'])
    if not os.path.exists(config.common_config['train_data_dir']):
        os.mkdir(config.common_config['train_data_dir'])
    if validPrepared:
        train = 1
        valid = 0
    else:
        assert len(glob(config.common_config['valid_data_dir'] + '*')) == 0, '請先刪除valid資料夾內檔案'
        
    assert len(glob(config.common_config['data_dir'] + '*')) != 0, '請確認image資料夾內有檔案，只接受(.png/.jpg)'
    imageList = np.array(glob(config.common_config['data_dir'] + '*'))
    
    assert len(glob(config.common_config['train_data_dir'] + '*')) == 0, '請先刪除train資料夾內檔案'
    
    trainIndex = np.random.choice(len(imageList), int(len(imageList) * train), replace=False)   
    validIndex = np.ones(len(imageList), dtype=np.bool_)
    validIndex[trainIndex] = False
    validIndex = np.where(validIndex == True)[0].astype(np.int32)
    
    print('Start Copy valid Images.')
    for validImage in tqdm(imageList[validIndex]):
        imageName = validImage.split('\\')[-1]
        try:
            imageSubName = imageName.split('.')[0]
            #assert len(imageSubName) == 5, '檔名長度有問題'
            [captcha_util.CHAR2LABEL[c] for c in imageSubName]
        except:
            print ('含有非法字元：' + imageSubName)
            raise ValueError
        shutil.copy(str(validImage), config.common_config['valid_data_dir'] + imageName)
        
    print('Start Copy Train Images.')        
    for trainImage in tqdm(imageList[trainIndex]):
        imageName = trainImage.split('\\')[-1]
        try:
            imageSubName = imageName.split('.')[0]
            #assert len(imageSubName) == 5, '檔名長度有問題'
            [captcha_util.CHAR2LABEL[c] for c in imageSubName]
        except:
            print ('含有非法字元：' + imageSubName)
            raise ValueError
        shutil.copy(str(trainImage), config.common_config['train_data_dir'] + imageName)
        


if __name__ == '__main__':
    np.random.seed(config.split_config['random_seed'])
    split_images(config.split_config['train'], config.split_config['valid'], config.split_config['validPrepared'])