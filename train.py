import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss

from dataset import CaptchaDataset, CaptchaDatasetFn
from model import CRNN
from config import train_config as config
import captcha_util 
from tqdm import tqdm

def evaluate(crnn, dataloader, criterion,
             max_iter=None, decode_method='beam_search', beam_size=10):
    crnn.eval()

    tot_count = 0
    tot_loss = 0
    tot_correct = 0
    wrong_cases = []

    pbar_total = max_iter if max_iter else len(dataloader)
    pbar = tqdm(total=pbar_total, desc="Evaluate")

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if max_iter and i >= max_iter:
                break
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'

            images, targets, target_lengths = [d.to(device) for d in data]

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            batch_size = images.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            preds = captcha_util.ctc_decode(log_probs, method=decode_method, beam_size=beam_size)
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()

            tot_count += batch_size
            tot_loss += loss.item()
            target_length_counter = 0
            for pred, target_length in zip(preds, target_lengths):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length
                if pred == real:
                    tot_correct += 1
                else:
                    wrong_cases.append((real, pred))

            pbar.update(1)
        pbar.close()

    evaluation = {
        'loss': tot_loss / tot_count,
        'acc': tot_correct / tot_count,
        'wrong_cases': wrong_cases
    }
    return evaluation


def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, targets, target_lengths = [d.to(device) for d in data]

    logits = crnn(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)
    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    epochs = config['epochs']
    train_batch_size = config['train_batch_size']
    eval_batch_size = config['eval_batch_size']
    lr = config['lr']
    show_interval = config['show_interval']
    valid_interval = config['valid_interval']
    save_interval = config['save_interval']
    cpu_workers = config['cpu_workers']
    reload_checkpoint = config['reload_checkpoint']

    img_width = config['img_width']
    img_height = config['img_height']
    train_data_dir = config['train_data_dir']
    valid_data_dir = config['valid_data_dir']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    train_dataset = CaptchaDataset(mode='train', data_dir=train_data_dir,
                                    img_height=img_height, img_width=img_width)
    valid_dataset = CaptchaDataset(mode='valid', data_dir=valid_data_dir,
                                    img_height=img_height, img_width=img_width)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=cpu_workers,
        collate_fn=CaptchaDatasetFn)
    
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=eval_batch_size,
        shuffle=True,
        num_workers=cpu_workers,
        collate_fn=CaptchaDatasetFn)
    
    num_class = len(captcha_util.LABEL2CHAR) + 1
    crnn = CRNN(3, img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    if reload_checkpoint:
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    optimizer = optim.RMSprop(crnn.parameters(), lr=lr)
    criterion = CTCLoss(reduction='mean')
    criterion.to(device)

    assert save_interval % valid_interval == 0
    i = 1
    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}')
        tot_train_loss = 0.
        tot_train_count = 0
        for train_data in train_loader:
            loss = train_batch(crnn, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size
            if i % show_interval == 0:
                print('train_batch_loss[', i, ']: ', loss / train_size)

            if i % valid_interval == 0:
                evaluation = evaluate(crnn, valid_loader, criterion,
                                      decode_method=config['decode_method'],
                                      beam_size=config['beam_size'])
                print('valid_evaluation: loss={loss}, acc={acc}'.format(**evaluation))

                if i % save_interval == 0:
                    prefix = 'crnn'
                    loss = evaluation['loss']
                    acc = evaluation['acc']
                    save_model_path = os.path.join(config['checkpoints_dir'],
                                                   f'{prefix}_{i:06}_loss{loss}_acc{acc}.pt')
                    torch.save(crnn.state_dict(), save_model_path)
                    print('save model at ', save_model_path)

            i += 1

        print('train_loss: ', tot_train_loss / tot_train_count)


if __name__ == '__main__':
    main()