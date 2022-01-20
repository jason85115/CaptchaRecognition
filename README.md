# CaptchaRecognition
# Python 驗證碼辨識
###### tags: `Tutorials` `Python` `Pytorch` `Captcha`

## 使用教學

### 前置準備
* 準備Python(3.10以下)、pytorch、tqdm、numpy、PIL、opencv、scipy 各函示庫。
* 準備驗證碼圖片(.jpg, .png)，且將檔名取為正確答案，範例：A1B1C.png。
* 設定設定檔，字元集、圖片寬高詳請見檔案解釋。
* 開始訓練前，請清空 train、valid 內圖片。 (若無要使用 <span>Split.py</span> 則免)
### 訓練開始
* 執行順序如下：
    1. <span>split.py</span> (可用可不用)
    2. <span>train.py</span>
* 定時查看訓練結果，若是達到滿意結果可中斷。
### 訓練結束
* 取出 checkpoints 資料夾下準確率最高的.pt檔。
* 設定 config 讀取該最高的.pt檔。
* 執行 ``python predict.py "欲辨識驗證碼圖片路徑"``。

```python=43
predict_config = {
    'best_checkpoint': '請輸入路徑', # 此處輸入.pt檔路徑
    'decode_method': 'beam_search',
    'beam_size': 10
}
```


## 可調整程式碼描述
### <span>config.py</span>

此 split_config 主要給 <span>split.py</span> 使用，可隨意更改，依照資料總數可設定訓練(train)、檢驗(valid)不同比例，至於 validPrepared 正常情況下設 false 讓系統自己去切分，但若資料會一直慢慢丟進去資料集裡，每次 split 訓練集可能會涵蓋到上次的檢驗集資料，這是可以考慮檢驗集設固定，自己蒐集一份放在 valid 資料夾下，將此變數設為True，則會將資料集所有資料拿去訓練。

```python=1
split_config = {
    'random_seed': 1,       # 隨機種子
    'train': 0.95,          # 訓練集占比 0.95 => 95%
    'valid': 0.05,          # 檢驗集占比 0.05 => 5%
    'validPrepared': False  # 若是 valid 資料已有準備，設為True，Split.py 將不會切分 valid 部分
}
```

圖片寬高取與要辨識的驗證碼大小接近的4和16的倍數，關於Lstm的兩個參數建議不要動，動了可能會執行不了，調整後需調整網路的架構，再來leaky_relu可以調整True或False看哪個訓練起來效果較好，每次調整都要重新訓練，最後字元集依照所要辨識的驗證碼圖片有哪些字元填入於此。

```python=8
common_config = {
    'data_dir': 'data/images/',        # 資料集的資料夾路徑
    'train_data_dir': 'data/train/',   # 訓練集的資料夾路徑
    'valid_data_dir': 'data/valid/',   # 檢驗集的資料夾路徑
    'img_width': 80,                   # 圖片寬 需4的倍數
    'img_height': 32,                  # 圖片高 需16倍數
    'map_to_seq_hidden': 64,           # lstm 的 input seq 大小
    'rnn_hidden': 256,                 # lstm 的 hidden Layer層數
    'leaky_relu': False,               # False = relu, True = leaky_relu
    'chars': '-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', # 字元集，第一個一定要放-代表空白。
}
```

主要都是一些訓練時會用到的參數，都可以隨意調整，lr愈大訓練愈快，盡量維持小一點比較不會訓練出問題。

```python=20
train_config = {
    'epochs': 10000,                  # 訓練幾次，一次代表訓練集全都跑過一遍
    'train_batch_size': 128,          # 訓練集一次幾張圖片一起訓練，若記憶體不夠可調小
    'eval_batch_size': 64,            # 測試集一次幾張圖片一起檢驗，若記憶體不夠可調小
    'lr': 0.0005,                     # 學習的速度(learning rate)
    'show_interval': 10,              # N 個 Batch 印一次 Loss
    'valid_interval': 500,            # N 個 Batch 檢驗一次
    'save_interval': 500,             # N 個 Batch 儲存一次，與 valid_interval 不同會錯誤
    'cpu_workers': 4,                 # 訓練時讀取資料 CPU 數量
    'reload_checkpoint': None,        # 輸入讀取的網路參數(.pt檔)路徑，把訓練過的模型再抓出來進行訓練, 沒有填None
    'decode_method': 'greedy',        # decode lstm 出來的結果要用甚麼方法，有三個 greedy、beam_search、prefix_beam_search
    'beam_size': 10,                  # beam_search、prefix_beam_search 用到的參數
    'checkpoints_dir': 'checkpoints/' # 儲存訓練網路參數(.pt檔)的位置
}
```

主要是 <span>predict.py</span> 用到的參數設定，最重要的 best_checkpoint 記得要填入路徑即可。

```python=36
predict_config = {
    'best_checkpoint': '',                # 網路參數(.pt檔)的路徑，像checkpoints/xxx.pt
    'decode_method': 'beam_search',       # decode lstm 出來的結果要用甚麼方法，有三個 greedy、beam_search、prefix_beam_search
    'beam_size': 10                       # beam_search、prefix_beam_search 用到的參數
}
```

### <span>model.py</span>

channels[0] 為image input channel 灰階為1、彩圖為3，後面數字愈大可調參數愈多，但需要的訓練及數量、訓練的時間、電腦的要求也愈高，但如果以上都滿足，參數愈多網路模型愈厲害。

```python=24
channels = [img_channel, 16, 32, 32, 64, 64, 128, 128]  # 可修改，數字愈大參數愈多。
```

## Finetune 模型

### 使用時機

1. 當訓練資料不夠多時，可拿取先前其它驗證碼網站的資料訓練出來的模型參數當作基礎。
2. 同一驗證碼網站有訓練過後但中途中斷想接續訓練時。
3. 同一驗證碼網站新增訓練圖片時。

### 設定方法
#### 將下方程式碼的None的部分更換為當初.pt檔存放的路徑即可囉!!
``` python=1
train_config = {
    'reload_checkpoint': None # (.pt檔)路徑，把訓練過的模型再抓出來進行訓練, 沒有填None
}
```

## 資料擴充

### 使用時機
1. 資料過少，且也懶得標記時，可參考 <span>GenFake.py</span>。

### 使用須知
:::danger
生成出來的圖片要與真實資料相近，且需訓練測試是否可以Work。
:::









