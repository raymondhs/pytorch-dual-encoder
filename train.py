import os
import time
import glob
import dill

import torch
import torch.optim as optim
import torch.nn as nn

from torchtext import data
from torchtext import datasets

from model import DualEncoder
from util import get_train_args, makedirs


args = get_train_args()
torch.cuda.set_device(args.gpu)
device = torch.device('cuda:{}'.format(args.gpu))

TEXT = data.Field(eos_token='<eos>', include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False, use_vocab=False)

train, dev = data.TabularDataset.splits(
        path=args.data_path, train='train.tsv',
        validation='valid.tsv', format='tsv', skip_header=True,
        csv_reader_params={"quotechar": None},
        fields=[('source', TEXT), ('target', TEXT), ('label', LABEL)])

TEXT.build_vocab(train)

train_iter, dev_iter = data.BucketIterator.splits(
        (train, dev), batch_size=args.batch_size, device=device,
        sort_key=lambda x: len(x.source), sort_within_batch=True)

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=device)
else:
    model = DualEncoder(len(TEXT.vocab),
                        embed_dim=args.embedding_size,
                        hidden_size=args.hidden_size,
                        num_layers=args.num_layers,
                        bidirectional=args.bidirectional,
                        dropout=args.dropout,
                        padding_idx=TEXT.vocab.stoi[TEXT.pad_token])
    model = model.to(device)
print(model)
print('{} parameters'.format(sum([p.numel() for p in model.parameters()])))

criterion = nn.BCELoss()
opt = optim.Adam(model.parameters(), lr=args.lr)

iterations = 0
start = time.time()
best_dev_acc = -1
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
makedirs(args.save_path)
torch.save(TEXT, os.path.join(args.save_path, 'vocab.pt'), pickle_module=dill)
print(header)

for epoch in range(args.epochs):
    train_iter.init_epoch()
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(train_iter):

        # switch model to training mode, clear gradient accumulators
        model.train(); opt.zero_grad()

        iterations += 1

        # forward pass
        (src, src_len), (trg, trg_len) = batch.source, batch.target
        label = batch.label.float()
        answer = model(src, src_len, trg, trg_len)

        # calculate accuracy of predictions in the current batch
        n_correct += ((answer > 0.5).float() == label).sum().item()
        n_total += batch.batch_size
        train_acc = 100. * n_correct/n_total

        # calculate loss of the network output with respect to training labels
        loss = criterion(answer, label)

        # backpropagate and update optimizer learning rate
        loss.backward(); opt.step()

        # checkpoint model periodically
        if iterations % args.save_every == 0:
            snapshot_prefix = os.path.join(args.save_path, 'snapshot')
            snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.item(), iterations)
            torch.save(model, snapshot_path)
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

        # evaluate performance on validation set periodically
        if iterations % args.dev_every == 0:

            # switch model to evaluation mode
            model.eval(); dev_iter.init_epoch()

            # calculate accuracy on validation set
            n_dev_correct, dev_loss = 0, 0
            with torch.no_grad():
                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                     (src, src_len), (trg, trg_len) = dev_batch.source, dev_batch.target
                     answer = model(src, src_len, trg, trg_len)
                     label = dev_batch.label.float()
                     n_dev_correct += ((answer > 0.5).float() == label).sum().item()
                     dev_loss = criterion(answer, label)
            dev_acc = 100. * n_dev_correct / len(dev)

            print(dev_log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.item(), dev_loss.item(), train_acc, dev_acc))

            # update best validation set accuracy
            if dev_acc > best_dev_acc:

                # found a model with better validation set accuracy

                best_dev_acc = dev_acc
                snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
                snapshot_path = snapshot_prefix + '_devacc_{:.4f}_devloss_{:.6f}__iter_{}_model.pt'.format(dev_acc, dev_loss.item(), iterations)

                # save model, delete previous 'best_snapshot' files
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

        elif iterations % args.log_every == 0:

            # print progress message
            print(log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.item(), ' '*8, n_correct/n_total*100, ' '*12))
