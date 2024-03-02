# Chest CT RE
針對 Chest CT 影像文字報告，訓練模型識別指定實體的相關實體資訊。

## 文本前處理
根據要進行關係萃取的實體兩側添加特殊符號（$）。以下範例的實體為 `pleural effusion`。
```
4. Mild pleural effusion on the left side. => 4. Mild $ pleural effusion $ on the left side.
```

## 訓練模型
訓練模型的程式碼是使用自己開發的 [winlp](https://github.com/leechehao/MyMLOps) 套件，它主要結合了 PyTorch Lightning 和 Hydra 的強大功能。

## Inference
在此範例中，對於 `mass` 實體識別出和其相關的實體資訊。
```
python inference.py --tracking_uri http://192.168.1.76:9487 \
                    --run_id 707af7b9c32342db90bcd54836ecded9 \
                    --text "1. A 2 cm $ mass $ in the right upper lobe ( RUL ) , highly suspicious for primary lung cancer ."
```
+ **tracking_uri** *(str)* ─ 指定 MLflow 追蹤伺服器的 URI。
+ **run_id** *(str)* ─ MLflow 實驗運行的唯一標識符。
+ **text** *(str)* ─ 待分析的文本。文本中須先加入特殊符號（$）在要關係萃取的實體兩側。。

輸出結果：
```
[
    {
        'entity_group': 'Size',
        'score': 0.9670808,
        'word': '2 cm',
        'start': 4,
        'end': 9
    },
    {
        'entity_group': 'Observation',
        'score': 0.9985362,
        'word': 'mass',
        'start': 11,
        'end': 16
    },
    {
        'entity_group': 'Location',
        'score': 0.99949074,
        'word': 'right upper lobe',
        'start': 25,
        'end': 42
    },
    {
        'entity_group': 'Location',
        'score': 0.9979983,
        'word': 'RUL',
        'start': 44,
        'end': 48
    },
    {
        'entity_group': 'Diagnosis',
        'score': 0.99425644,
        'word': 'primary lung cancer',
        'start': 74,
        'end': 94
    }
]
```