import torch
from torchtext import data, datasets

from util import get_scorer_args


args = get_scorer_args()
torch.cuda.set_device(args.gpu)
device = torch.device('cuda:{}'.format(args.gpu))

TEXT = torch.load(args.vocab)
LABEL = data.Field(sequential=False, use_vocab=False)

test = data.TabularDataset(
	path=args.input, format='tsv', skip_header=True,
	csv_reader_params={'quotechar': None},
	fields=[('source', TEXT), ('target', TEXT), ('label', LABEL)])

if args.eval:
	test_iter = data.BucketIterator(
		test, batch_size=args.batch_size, device=device,
		sort_key=lambda x: len(x.source),
		sort_within_batch=True, train=False)

if args.predict:
	test_iter = data.Iterator(
		test, batch_size=args.batch_size, device=device,
		sort=False, shuffle=False)

model = torch.load(args.model, map_location=device)
if args.predict:
	model.enforce_sorted_source = False

n_test_correct = 0
predictions = []
with torch.no_grad():
    for idx, batch in enumerate(test_iter):
        (src, src_len), (trg, trg_len) = batch.source, batch.target
        answer = model(src, src_len, trg, trg_len)
        if args.eval:
	        label = batch.label.float()
	        n_test_correct += ((answer > args.threshold).float() == label).sum().item()
        if args.predict:
            predictions.extend(answer.tolist())
 
if args.eval:
 	test_acc = 100. * n_test_correct / len(test)
 	print('Accuracy: {:.2f}%'.format(test_acc))

if args.predict:
 	with open(args.output, 'w') as f:
 		for p in predictions:
 			f.write(p)
 			f.write('\n')
