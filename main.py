import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import random
import numpy as np
from src.models.evflownet import EVFlowNet
from src.datasets import DatasetProvider
from enum import Enum, auto
from src.datasets import train_collate
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
from timm.scheduler import CosineLRScheduler
import pandas as pd
import pickle
import os
import datetime
from pytz import timezone
# trainデータの中身を確認する

class RepresentationType(Enum):
    PATH = '/content/drive/MyDrive/DL_lesson/DLlast/checkpoints'
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor):
    '''
    end-point-error (ground truthと予測値の二乗誤差)を計算
    pred_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 予測したオプティカルフローデータ
    gt_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 正解のオプティカルフローデータ
    '''
    epe = torch.mean(torch.mean(torch.norm(pred_flow - gt_flow, p=2, dim=1), dim=(1, 2)), dim=0)

    return epe

# 中間層のサイズをtorch.Size([B, 2, 480, 640]に変更する

def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    '''
    optical flowをnpyファイルに保存
    flow: torch.Tensor, Shape: torch.Size([2, 480, 640]) => オプティカルフローデータ
    file_name: str => ファイル名
    '''
    np.save(f"{file_name}.npy", flow.cpu().numpy())

def get_time():
    time = datetime.datetime.now(timezone('Asia/Tokyo'))
    return time.strftime("%Y%m%d%H%M")

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    set_seed(args.seed)
    SAVE_NAME = Path(args.save_name)
    LOAD_NAME = Path(args.load_name)    
    current_time = get_time()
    model_load_path = f"{LOAD_NAME}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    '''
        ディレクトリ構造:

        data
        ├─test
        |  ├─test_city
        |  |    ├─events_left
        |  |    |   ├─events.h5
        |  |    |   └─rectify_map.h5
        |  |    └─forward_timestamps.txt
        └─train
            ├─zurich_city_11_a
            |    ├─events_left
            |    |       ├─ events.h5
            |    |       └─ rectify_map.h5
            |    ├─ flow_forward
            |    |       ├─ 000134.png
            |    |       |.....
            |    └─ forward_timestamps.txt
            ├─zurich_city_11_b
            └─zurich_city_11_c
        '''
        
    # TODO:transformer追加
    
    # ------------------
    #    Dataloader
    # ------------------
    """
    VOXEL
    3次元ボクセルデータを使用する場合は、このオプションを選択します。
    delta_t_ms:
    イベントデータの時間間隔（ms）。
    num_bins:
    イベントデータのビン数。
    """
    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        visualize=True,
        num_bins=args.train.num_bins,
        sequenceRecurrent=args.sequenceRecurrent,
        config=None
    )
    print(f"train data: {len(loader.get_train_dataset())}, test data: {len(loader.get_test_dataset())}")
    # print(f'summary: {loader.summary()}')
    train_set = loader.get_train_dataset()
    test_set = loader.get_test_dataset()
    train_data = DataLoader(train_set,
                                batch_size=args.data_loader.train.batch_size,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                shuffle=args.data_loader.train.shuffle,
                                collate_fn=train_collate, # collate_fnはデータをバッチにまとめる関数
                                drop_last=False)
    test_data = DataLoader(test_set,
                                batch_size=args.data_loader.test.batch_size,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                shuffle=args.data_loader.test.shuffle,
                                collate_fn=train_collate,
                                drop_last=False)

    '''
    train data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
        Key: flow_gt, Type: torch.Tensor, Shape: torch.Size([Batch, 2, 480, 640]) => オプティカルフローデータのバッチ
        Key: flow_gt_valid_mask, Type: torch.Tensor, Shape: torch.Size([Batch, 1, 480, 640]) => オプティカルフローデータのvalid. ベースラインでは使わない
        TODO: flow_gt_valid_mask多分何かに使える

    
    test data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
    '''
    # ------------------
    #       Model
    # ------------------
    if os.path.exists(model_load_path):
        model = pickle.load(open(f'{model_load_path}', 'rb'))
        # model.load_state_dict(torch.load(model_load, map_location=device))
        print(f"Model loaded from {model_load_path}")
    else:
        print("First training model")
        model = EVFlowNet(args.train).to(device,non_blocking=True)

    # ------------------
    #   optimizer
    # ------------------
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)
    if args.train.epochs != 0:
        scheduler = CosineLRScheduler(optimizer, t_initial=args.train.epochs, lr_min=1e-4, 
                                warmup_t=4, warmup_lr_init=5e-5, warmup_prefix=True)
    # ------------------
    #   Start training
    # ------------------
    """
    訓練データは合計2015データあり, 
    イベントデータ，タイムスタンプ，正解オプティカルフローのRGB画像が与えられる．
    flow:
        オプティカルフローデータを格納するための変数?
    flow_gt:
        訓練データの正解のオプティカルフローデータ?
    compute_epe_error:
        予測されたオプティカルフローと正解データのend point errorを計算する.
    
    """

    # if os.path.exists(model_load_path):
    #     model.load_state_dict(torch.load(model_load_path, map_location=device))
    #     print(f"Model loaded from {model_load_path}")
    # else:
    #     print("First training model")
    model.train()
    for epoch in range(args.train.epochs):
        model_save_path = f'checkpoints/{current_time}_{epoch}_{SAVE_NAME}'
        scheduler.step(epoch+1)
        total_loss = 0
        batch_loss = 0
        step_count = 0
        # print(f"Epoch {epoch+1} start")
        for i, batch in enumerate(tqdm(train_data)):
            optimizer.zero_grad()
            step_count += 1
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device,non_blocking=True) # [B, 4, 480, 640]
            ground_truth_flow = batch["flow_gt"].to(device,non_blocking=True) # [B, 2, 480, 640]
            flow_dict, _ = model(event_image) # [B, 2, 480, 640]
            loss = 0
            for key in flow_dict.keys():
                loss += compute_epe_error(flow_dict[key], ground_truth_flow)/args.batch_extend
            if args.detail==True:
                print(f"batch:{i}, loss: {loss}")
            loss.backward()
            batch_loss += loss.item()
            if step_count % args.batch_extend == 0 :  # イテレーションごとに更新することで，擬似的にバッチサイズを大きくしている
                print(f'step_update_{i//args.batch_extend}_loss: {batch_loss/args.batch_extend}')
                print(f'learning_late: {optimizer.param_groups[0]["lr"]}')
                batch_loss = 0
                step_count = 0
                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item()


        print(f"epoch:{epoch} batch {i} loss: {total_loss / len(train_data)}")
        print(f"final_train_loss: {loss.item()}")
        # torch.save(model.state_dict(), model_save_path)
        # save model by pickle


        # Create the directory if it doesn't exist
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        
        # torch.save(model.state_dict(), model_load_path)
        pickle.dump(model,open(f"{model_save_path}.pkl", 'wb'))
        print(f"Model saved to {model_save_path}")

    # ------------------
    #   Start predicting
    # ------------------
    """
    _summary_
    tqdm:プログレスバーを表示するライブラリ
    `compute_epe_error`: 
        予測されたオプティカルフローと正解データのend point errorを計算する.
    `save_optical_flow_to_npy`: 
        オプティカルフローデータを `.npy` ファイルに保存する.
        97データがテストデータとして与えられる．
        - テストデータに対する回答は正解のオプティカルフローとし，訓練時には与えられない．
    batch_flow:
        バッチごとのオプティカルフローデータを格納するための変数?
    flow:
        オプティカルフローデータを格納するための変数?
    """
    if 'model_save_path' in locals():
        model_load = model_save_path
    else:
        model_load = model_load_path

    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device,non_blocking=True)
    with torch.no_grad():
        print(f"start test:{model_load}_model")
        for batch in tqdm(test_data):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device,non_blocking=True) # [1, 4, 480, 640]
            flow_dict,batch_flow = model(event_image) # [1, 2, 480, 640]
            flow = torch.cat((flow, batch_flow), dim=0)  # [N, 2, 480, 640]
        print("test done")

    # ------------------
    #  save submission
    # ------------------
    file_name = "submission"
    save_optical_flow_to_npy(flow, file_name)

if __name__ == "__main__":
    main()
