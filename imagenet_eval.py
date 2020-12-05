import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse
from nasnet import NASNetALarge, NASNetAMobile, ImageNet
import time

parser = argparse.ArgumentParser()
parser.add_argument('--nas-type', type=str, choices=['mobile', 'large'], metavar='NASNET_TYPE',
                    help='nasnet type: mobile | large')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--resume', type=str, metavar='NASNET_CHECKPOINT', help='path for nasnet checkpoint')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for eval (default: 256)')
parser.add_argument('--gpus', type=int, default=None, nargs='*', metavar='--gpus 0 1 2 ...',
                    help='gpu ids for CUDA training')
parser.add_argument('--data', type=str, default='datasets', metavar='data_root_path',
                    help="data root: /path/to/dataset (default: 'datasets')")

args = parser.parse_args()
print(args)

if not args.gpus or (len(args.gpus) > 0 and (args.gpus[0] < 0 or not torch.cuda.is_available())):
    args.gpus = []

torch.manual_seed(args.seed)
if len(args.gpus) > 0:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 8, 'pin_memory': True} if len(args.gpus) > 0 else {}
image_size = args.nas_type == 'mobile' and 224 or 331
test_loader = torch.utils.data.DataLoader(
    ImageNet(args.data, train=False, image_size=image_size),
    batch_size=args.batch_size, shuffle=False, **kwargs)

model = NASNetALarge(1001) if args.nas_type == 'large' else NASNetAMobile(1001)
model.eval()
checkpoint = torch.load(args.resume)
if 'state_dict' in checkpoint:
    checkpoint = checkpoint['state_dict']
model.load_state_dict(state_dict=checkpoint, strict=False)
# 1001 -> 1000
new_linear = nn.Linear(model.linear.in_features, 1000)
new_linear.weight.data = model.linear.weight.data[1:]
new_linear.bias.data = model.linear.bias.data[1:]
model.linear = new_linear
print(model)
print('num of params:', model.num_params)

if len(args.gpus) > 0:
    model.cuda()
    cudnn.benchmark = True
    cudnn.deterministic = True
    if len(args.gpus) > 1:
        model = nn.DataParallel(model, device_ids=args.gpus).cuda()

dataloader = test_loader
criterion = nn.CrossEntropyLoss()
# eval
loss = 0
top1 = 0
top5 = 0
timer = time.time()
for batch_idx, (data, target) in enumerate(dataloader):
    if len(args.gpus) > 0:
        data, target = data.cuda(), target.cuda()
    with torch.no_grad():
        output = model(data)[0]
    loss += criterion(output, target).data.item() * len(data)
    _, predictions = output.data.topk(5, 1, True, True)
    topk_correct = predictions.eq(target.data.contiguous().view(len(data), 1).expand_as(predictions)).cpu()
    top1 += len(data) - topk_correct.narrow(1, 0, 1).sum().item()
    top5 += len(data) - topk_correct.sum().item()
    if (batch_idx + 1) % 10 == 0:
        processed_data = len(data) * (batch_idx + 1)
        print('Test set[{}/{}]: Top1: {:.2f}%, Top5: {:.2f}%, Average loss: {:.4f}, Average time cost: {:.3f} s'.format(
            processed_data, len(dataloader.dataset), 100 * top1 / processed_data,
            100 * top5 / processed_data, loss / processed_data, (time.time() - timer) / processed_data))

loss /= len(dataloader.dataset)
print('Test set[{}]: Top1: {:.2f}%, Top5: {:.2f}%, Average loss: {:.4f}, Average time cost: {:.3f} s'.format(
    len(dataloader.dataset), 100 * top1 / len(dataloader.dataset),
    100 * top5 / len(dataloader.dataset), loss, (time.time() - timer) / len(dataloader.dataset)))
