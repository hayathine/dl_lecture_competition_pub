# hydraの使い方

https://zenn.dev/gesonanko/articles/417d43669cf2af

### gettingstarted

https://hydra.cc/docs/intro/

#### 2024年6月25日12:19

lossが増えている→過学習？

dropoutを実装する→lossが増加する

#### 2024年6月25日13:28

trainデータの中身を確認する

#### 2024年6月27日
バッチの擬似的
https://kozodoi.me/blog/20210219/gradient-accumulation
    for x, _ in train_dl:
        step_count += 1
        model.train()
        x = x.to(device)

        rec_img, mask = model(x)
        train_loss = torch.mean((rec_img - x) ** 2 * mask) / config["mask_ratio"]
        train_loss.backward()

        if step_count % 8 == 0:  # 8イテレーションごとに更新することで，擬似的にバッチサイズを大きくしている
            optimizer.step()
            optimizer.zero_grad()

        total_train_loss += train_loss.item()

多分勾配消失を起こしているので修正する。
- 異なるスケールでのロスを足し合わせる．
  - ベースラインモデルはUNet構造なので，デコーダーの中間層の出力は最終的な出力サイズの0.5,0.25,...倍になっています．各中間層の出力を用いてロスを計算することで，勾配消失を防ぎ，性能向上が見込めます．

dataのpickle化による
https://hyper-pigeon.hatenablog.com/entry/2021/08/04/225814