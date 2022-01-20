import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import predict_config as config
from dataset import CaptchaDataset, CaptchaDatasetFn
from model import CRNN
from captcha_util import ctc_decode
from glob import glob
from PIL import Image
import numpy as np
import captcha_util

def predict_by_path(image_path):
    best_checkpoint = config['best_checkpoint']
    decode_method = config['decode_method']
    beam_size = config['beam_size']
    num_class = len(captcha_util.LABEL2CHAR) + 1

    img_height = config['img_height']
    img_width = config['img_width']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    crnn = CRNN(3, img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    
    crnn.load_state_dict(torch.load(best_checkpoint, map_location=device))
    crnn.to(device)
    crnn.eval()
    
    image = Image.open(image_path)

    image = image.resize((img_width, img_height), resample=Image.BILINEAR)
    image = np.array(image)
    image = image.reshape((3, img_height, img_width))

    image = torch.FloatTensor(image).to(device)
    image = image.view(1, 3, image.size()[1], image.size()[2])
    
    logits = crnn(image)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    pred = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                       label2char=captcha_util.LABEL2CHAR)

    result = ''.join(pred[0]) 
    
    return result
    
def predict_by_directory(images_path):
    best_checkpoint = config['best_checkpoint']
    decode_method = config['decode_method']
    beam_size = config['beam_size']

    img_height = config['img_height']
    img_width = config['img_width']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    
    predict_dataset = CaptchaDataset(mode='pred', data_dir=images_path,
                                    img_height=img_height, img_width=img_width)

    predict_loader = DataLoader(
        dataset=predict_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=CaptchaDatasetFn)

    num_class = len(captcha_util.LABEL2CHAR) + 1
    crnn = CRNN(3, img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    
    crnn.load_state_dict(torch.load(best_checkpoint, map_location=device))
    crnn.to(device)
    crnn.eval()

    pbar = tqdm(total=len(predict_loader), desc="Predict")

    all_preds = []
    with torch.no_grad():
        for data in predict_loader:
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'

            images, targets, target_lengths = [d.to(device) for d in data]

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                               label2char=captcha_util.LABEL2CHAR)
            all_preds += preds

            pbar.update(1)
        pbar.close()
    
    images = glob(images_path + '*')
    show_result(images, all_preds)
    
    return all_preds

def predict_and_label(images_path, subFileName):
    imagePaths = glob(images_path + '*' + subFileName)
    for imagePath in imagePaths:
        imagePath = imagePath.replace('\\', '/')
        predLabel = predict_by_path(imagePath)
        print(imagePath)
        captcha_util.rename2label(imagePath, predLabel)

def show_result(paths, preds):
    print('\n===== result =====')
    recordCorrectNum = [0,0,0,0,0,0]
    for path, pred in zip(paths, preds):
        gt = path.split('\\')[-1].split('.')[0]
        answer = ''.join(pred)
        correctNum = 0
        for idx, partialGT in enumerate(gt):
            try:
                if partialGT == pred[idx]:
                    correctNum += 1
            except:
                break
        if correctNum == 1:
            print(f'Correct 1, {gt} > {answer}')
        if correctNum == 2:
            print(f'Correct 2, {gt} > {answer}')
        if correctNum == 3:
            print(f'Correct 3, {gt} > {answer}')    
        #if correctNum == 4:
        #    print(f'Correct 4, {gt} > {answer}')
        #if gt == answer:
        #    print(f'{gt} > {answer}')
        recordCorrectNum[correctNum] += 1;
        
    print('\n===== statistical result ======')
    print(f'正確0個字:{recordCorrectNum[0]}個')
    print(f'正確1個字:{recordCorrectNum[1]}個')
    print(f'正確2個字:{recordCorrectNum[2]}個')
    print(f'正確3個字:{recordCorrectNum[3]}個')
    print(f'正確4個字:{recordCorrectNum[4]}個')
    print(f'正確5個字:{recordCorrectNum[5]}個')

def main(args):
    #test = r'D:\Desktop\Spyder\CaptchaRecognition\data\pred\3EXN2.png'
    #print(predict_by_path(r'./Captcha.jfif'))
    #predict_by_directory('./data/valid/')
    predict_and_label('./待標記/', '.png')

if __name__ == '__main__':
    main(sys.argv)

