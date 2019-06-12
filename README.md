
# Dual Encoder LSTM

Simple Dual Encoder LSTM implementation in PyTorch.

## Training

```
python train.py \
  --data_path data/ \
  --epochs 20 \
  --batch_size 512 \
  --embedding_size 320 \
  --hidden_size 512 \
  --num_layers 1 \
  --dropout 0.1 \
  --lr 0.001 \
  --log_every 50 \
  --save_path models
```

## Scoring

```
python score.py \
  --input data/test.tsv \
  --model models/model_best.pt \
  --vocab models/vocab.pt \
  --eval \
  --threshold 0.5 \
  --batch_size 64
```
