# Variational AutoEncoder(VAE)を用いた異常検出

教師なし学習。  
生データと再構成した画像との差異の大きさで正常/異常を判定。

### 動作環境
- Python: 3.6.9
- TensorFlow: 2.3.0

# 実行例
### サンプル１
- 正常
![blush](samples/sample1-1.png "Title")
- 異常
![blush](samples/sample1-2.png "Title")

### サンプル２
- 正常
![blush](samples/sample2-1.png "Title")
- 異常
![blush](samples/sample2-2.png "Title")

# 検証例
- 正常：６の手書き画像（学習データ:5918枚/検証データ:958枚）
- 異常：９の手書き画像（検証データ:958枚）

### サンプル３
- 正常
![blush](samples/sample3-1.png "Title")
- 異常
![blush](samples/sample3-2.png "Title")

### 結果
![blush](samples/histgram.png "Title")

Score: 差異の大きさ
