split_config = {
    'random_seed': 1,       # 隨機種子
    'train': 0.95,          # 訓練集占比 0.95 => 95%
    'valid': 0.05,          # 檢驗集占比 0.05 => 5%
    'validPrepared': False  # 若是 valid 資料已有準備，設為True，Split.py 將不會切分 valid 部分
}

common_config = {
    'data_dir': '../data/images/',        # 資料集的資料夾路徑
    'train_data_dir': ['../data/images_IDCard/', '../data/images_Aug/'],  # 訓練集的資料夾路徑，可字串'xxxx/xxxx/' 可List ['xxx/xxx/','xxx/xxx/'...]
    'valid_data_dir': '../data/valid_IDCard/',   # 檢驗集的資料夾路徑
    'img_width': 80,                   # 圖片寬 需4的倍數
    'img_height': 32,                  # 圖片高 需16倍數
    'map_to_seq_hidden': 64,           # lstm 的 input seq 大小
    'rnn_hidden': 256,                 # lstm 的 hidden Layer層數 64 128 256
    'leaky_relu': False,               # False = relu, True = leaky_relu
    'chars': '-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', # 字元集，第一個一定要放-代表空白。
}

train_config = {
    'epochs': 10000,                  # 訓練幾次，一次代表訓練集全都跑過一遍
    'train_batch_size': 64,           # 訓練集一次幾張圖片一起訓練，若記憶體不夠可調小
    'eval_batch_size': 64,            # 測試集一次幾張圖片一起檢驗，若記憶體不夠可調小
    'lr': 0.0005,                     # 學習的速度(learning rate) 0.0005  0.005 0.05
    'show_interval': 10,              # N 個 Batch 印一次 Loss
    'valid_interval': 500,            # N 個 Batch 檢驗一次
    'save_interval': 500,             # N 個 Batch 儲存一次，與 valid_interval 不同會錯誤
    'save_interval_model': False,     # N 個 Batch 是否要儲存
    'cpu_workers': 4,                 # 訓練時讀取資料 CPU 數量
    'reload_checkpoint': None,        # 輸入讀取的網路參數(.pt檔)路徑，把訓練過的模型再抓出來進行訓練, 沒有填None
    'decode_method': 'greedy',        # decode lstm 出來的結果要用甚麼方法，有三個 greedy、beam_search、prefix_beam_search
    'beam_size': 10,                  # beam_search、prefix_beam_search 用到的參數
    'checkpoints_dir': 'checkpoints/' # 儲存訓練網路參數(.pt檔)的位置
}
train_config.update(common_config)

predict_config = {
    'best_checkpoint': 'checkpoints/best_IDCard_36.6.pt',             # 網路參數(.pt檔)的路徑，像checkpoints/xxx.pt
    'decode_method': 'beam_search',    # decode lstm 出來的結果要用甚麼方法，有三個 greedy、beam_search、prefix_beam_search
    'beam_size': 10                   # beam_search、prefix_beam_search 用到的參數
}
predict_config.update(common_config)

