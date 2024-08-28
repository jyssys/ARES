# -*- coding: utf-8 -*-
import numpy as np
import random
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
import wandb

from tqdm import tqdm
import math
from scipy.stats import entropy

from models.allconv import AllConvNet
from models.wrn_virtual import WideResNet

import pixmix_utils as p_utils

# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.validation_dataset import validation_split

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'],
                    default='cifar10',
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='wrn',
                    choices=['allconv', 'wrn'], help='Choose architecture.')
parser.add_argument('--calibration', '-c', action='store_true',
                    help='Train a model to be used for calibration. This holds out some data for validation.')
parser.add_argument('--mode', default='ARES', type=str, choices=['ARES', 'VOS'])
parser.add_argument('--seed', default=605, type=int)
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/baseline', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
# energy reg
parser.add_argument('--start_epoch', type=int, default=40)
parser.add_argument('--sample_number', type=int, default=1000)
parser.add_argument('--m_sample_number', type=int, default=10000)
parser.add_argument('--select', type=int, default=1)
parser.add_argument('--sample_from', type=int, default=10000)
parser.add_argument('--loss_weight', type=float, default=0.1)
# mixup
parser.add_argument('--alpha', default=2., type=float, help='mixup interpolation coefficient (default: 1) between 0~1')
# pixmix
parser.add_argument('--beta', default=3, type=int, help='Severity of mixing')
parser.add_argument('--k', default=4, type=int, help='Mixing iterations')
parser.add_argument('--aug-severity', default=3, type=int, help='Severity of base augmentation operators')
parser.add_argument('--all-ops', '-all', action='store_true', help='Turn on all augmentation operations (+brightness,contrast,color,sharpness).')

# other option
parser.add_argument('--project', default='ARES-classification')
parser.add_argument('--use_wandb', action="store_true")

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

# ----------SEED----------
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything(args.seed)
os.environ["WANDB__SERVICE_WAIT"] = "300"

# ---------PIXMIX----------
def pixmix_improve(orig, mixing_pic, preprocess):
    mixings = p_utils.mixings
    tensorize, normalize = preprocess['tensorize'], preprocess['normalize']
    
    if np.random.random() < 0.5:
        mixed = tensorize(augment_input(orig))
    else:
        mixed = tensorize(orig)

    for i in range(np.random.randint(args.k)+1): # 0~4 -> 1~4
        if i == 0:
            aug_image_copy = tensorize(mixing_pic)
        else:
            if np.random.random() < 0.5:
                aug_image_copy = tensorize(augment_input(orig))
            else:
                aug_image_copy = tensorize(mixing_pic)
                
        mixed_op = np.random.choice(mixings)
        mixed = mixed_op(mixed, aug_image_copy, args.beta)
        mixed = torch.clip(mixed, 0, 1)

    return normalize(mixed)

def augment_input(image):
    aug_list = p_utils.augmentations_all if args.all_ops else p_utils.augmentations
    op = np.random.choice(aug_list)
    return op(image.copy(), args.aug_severity)

class PixMixDataset_improve(torch.utils.data.Dataset):
    """Dataset wrapper to perform PixMix."""

    def __init__(self, dataset, mixing_set, preprocess):
        self.dataset = dataset
        self.mixing_set = mixing_set
        self.preprocess = preprocess

    def __getitem__(self, i):
        x, y = self.dataset[i]
        rnd_idx = np.random.choice(len(self.mixing_set))
        mixing_pic, _ = self.mixing_set[rnd_idx]
        return pixmix_improve(x, mixing_pic, self.preprocess), y

    def __len__(self):
        return len(self.dataset)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

if args.mode.startswith('ARES'):
    train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4)])
else:
    train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                trn.ToTensor(), trn.Normalize(mean, std)])

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
ood_transform = trn.Compose([
                trn.ToTensor(),
                trn.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if args.dataset == 'cifar10':
    train_data = dset.CIFAR10('./data/cifarpy', train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10('./data/cifarpy', train=False, transform=test_transform, download=True)
    ood_data = dset.SVHN('./data', split='test', transform=ood_transform, download=True)
    num_classes = 10
else:
    train_data = dset.CIFAR100('./data/cifarpy', train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR100('./data/cifarpy', train=False, transform=test_transform, download=True)
    ood_data = dset.SVHN('./data', split='test', transform=ood_transform, download=True)
    num_classes = 100

calib_indicator = ''
if args.calibration:
    train_data, val_data = validation_split(train_data, val_share=0.1)
    calib_indicator = '_calib'

if args.mode.startswith('ARES'):
    mixing_set_transform = trn.Compose(
        [trn.Resize(36), trn.RandomCrop(32)])
    
    to_tensor = trn.ToTensor()
    normalize = trn.Normalize([0.5] * 3, [0.5] * 3)
    mixing_path = './data/fractals_and_fvis/fractals'
    
    mixing_set = dset.ImageFolder(mixing_path, transform=mixing_set_transform)
    
    train_data = PixMixDataset_improve(train_data, mixing_set, {'normalize': normalize, 'tensorize': to_tensor})
    
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.prefetch,
        pin_memory=True)
else:
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)


test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

# -----------------------------------------------
# Create model
if args.model == 'allconv':
    net = AllConvNet(num_classes)
else:
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

start_epoch = 0

# Restore model if desired
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        model_name = os.path.join(args.load, args.dataset + calib_indicator + '_' + args.model +
                                  '_baseline_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume"

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()

# cudnn.benchmark = True  # fire on all cylinders

if args.dataset == 'cifar10':
    num_classes = 10
else:
    num_classes = 100

weight_energy = torch.nn.Linear(num_classes, 1).cuda()
torch.nn.init.uniform_(weight_energy.weight)

if args.mode == 'ARES':
    data_dict = torch.zeros(1, args.m_sample_number, 128).cuda()
    number_dict = 0
else:
    data_dict = torch.zeros(num_classes, args.sample_number, 128).cuda()
    
    number_dict = {}
    for i in range(num_classes):
        number_dict[i] = 0

eye_matrix = torch.eye(128, device='cuda')
logistic_regression = torch.nn.Linear(1, 2)
logistic_regression = logistic_regression.cuda()
optimizer = torch.optim.SGD(
    list(net.parameters()) + list(weight_energy.parameters()) + \
    list(logistic_regression.parameters()), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader), # args.epochs * len(train_loader)
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))

class JSDivergenceCalculator:
    def __init__(self, energy_score_for_fg, energy_score_for_bg):
        self.energy_score_for_fg = energy_score_for_fg
        self.energy_score_for_bg = energy_score_for_bg
    
    def kl_divergence(self, p, q):
        """Compute the KL divergence of two distributions."""
        
        mu_fg = p.mean().item()
        sigma_fg = math.sqrt(p.var().item())

        mu_bg = q.mean().item()
        sigma_bg = math.sqrt(q.var().item())
        
        kl_div = math.log(sigma_bg / sigma_fg, 2) + ((sigma_fg**2 + (mu_fg - mu_bg)**2) / (2 * sigma_bg**2)) - 0.5
        return kl_div

    def js_divergence(self, p, q):
        """Compute the Jensen-Shannon divergence between two distributions."""
        m = 0.5 * (p + q)
        return 0.5 * (self.kl_divergence(p, m) + self.kl_divergence(q, m))

    def calculate_js_divergence(self):
        """Calculate JS divergence between two distributions."""
        js_div = self.js_divergence(self.energy_score_for_fg, self.energy_score_for_bg)
        
        if not isinstance(js_div, torch.Tensor):
            js_div = torch.tensor(js_div)
            
        js_div_clamped = torch.where(js_div > 1, js_div, torch.clamp(js_div, min=1e-7))

        return js_div_clamped

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    import math
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(
            F.relu(weight_energy.weight) * torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        # if isinstance(sum_exp, Number):
        #     return m + math.log(sum_exp)
        # else:
        return m + torch.log(sum_exp)

# /////////////// Training VOS ///////////////

def train(epoch):    
    net.train()  # enter train mode
    loss_avg = 0.0
    
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()

        # forward
        x, output = net.forward_virtual(data)

        # energy regularization.
        sum_temp = 0
        for index in range(num_classes):
            sum_temp += number_dict[index]
        lr_reg_loss = torch.zeros(1).cuda()[0]
        if sum_temp == num_classes * args.sample_number and epoch < args.start_epoch:
            # maintaining an ID data queue for each class.
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:], output[index].detach().view(1, -1)), 0)
        elif sum_temp == num_classes * args.sample_number and epoch >= args.start_epoch:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:], output[index].detach().view(1, -1)), 0)
            # the covariance finder needs the data to be centered.
            for index in range(num_classes):
                if index == 0:
                    X = data_dict[index] - data_dict[index].mean(0)
                    mean_embed_id = data_dict[index].mean(0).view(1, -1)
                else:
                    X = torch.cat((X, data_dict[index] - data_dict[index].mean(0)), 0)
                    mean_embed_id = torch.cat((mean_embed_id, data_dict[index].mean(0).view(1, -1)), 0)

            ## add the variance.
            temp_precision = torch.mm(X.t(), X) / len(X)
            temp_precision += 0.0001 * eye_matrix


            for index in range(num_classes):
                new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                    mean_embed_id[index], covariance_matrix=temp_precision)
                negative_samples = new_dis.rsample((args.sample_from,))
                prob_density = new_dis.log_prob(negative_samples)
                # breakpoint()
                # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
                # keep the data in the low density area.
                cur_samples, index_prob = torch.topk(- prob_density, args.select)
                if index == 0:
                    ood_samples = negative_samples[index_prob]
                else:
                    ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
            
            if len(ood_samples) != 0:
                # add some gaussian noise
                # ood_samples = self.noise(ood_samples)
                # energy_score_for_fg = 1 * torch.logsumexp(predictions[0][selected_fg_samples][:, :-1] / 1, 1)
                energy_score_for_fg = log_sum_exp(x, 1)
                predictions_ood = net.fc(ood_samples)
                # energy_score_for_bg = 1 * torch.logsumexp(predictions_ood[0][:, :-1] / 1, 1)
                
                energy_score_for_bg = log_sum_exp(predictions_ood, 1)

                input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                labels_for_lr = torch.cat((torch.ones(len(output)).cuda(), torch.zeros(len(ood_samples)).cuda()), -1)

                criterion = torch.nn.CrossEntropyLoss()
                output1 = logistic_regression(input_for_lr.view(-1, 1))
                
                lr_reg_loss = criterion(output1, labels_for_lr.long())

                if epoch % 5 == 0:
                    print(f'epoch : {epoch}, lr_reg_loss : {lr_reg_loss}')
        else:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                if number_dict[dict_key] < args.sample_number:
                    data_dict[dict_key][number_dict[dict_key]] = output[index].detach()
                    number_dict[dict_key] += 1

        # backward

        optimizer.zero_grad()
        loss = F.cross_entropy(x, target)
        # breakpoint()
        loss += args.loss_weight * lr_reg_loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state['train_loss'] = loss_avg

# ////// ARES Training //////

def train_ares(epoch):    
    net.train()  # enter train mode
    loss_avg = 0.0
    global number_dict
    
    # vk_entropy = 0
    # id_entropy = 0
    
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()

        # ID data forward
        x, id_output = net.forward_virtual(data)
        
        # manifold mixup data to estimate OOD
        with torch.no_grad():
            _, output, _ = net.forward_spc_vm_mixup(data, target, mixup=True, mixup_alpha=args.alpha)

        # energy regularization.
        sum_temp = 0
        
        sum_temp += number_dict
            
        lr_reg_loss = torch.zeros(1).cuda()[0]
        
        if sum_temp == args.m_sample_number and epoch < args.start_epoch:
            # maintaining an ID data queue for each class.
            for index in range(output.size(0)):
                data_dict[0] = torch.cat((data_dict[0][1:], output[index].detach().view(1, -1)), 0)
        elif sum_temp == args.m_sample_number and epoch >= args.start_epoch:
            for index in range(output.size(0)):
                data_dict[0] = torch.cat((data_dict[0][1:], output[index].detach().view(1, -1)), 0)
            
            # the covariance finder needs the data to be centered.
            X = data_dict[0] - data_dict[0].mean(0)
            mean_embed_id = data_dict[0].mean(0).view(1, -1)

            ## add the variance.
            temp_precision = torch.mm(X.t(), X) / len(X)
            temp_precision += 0.0001 * eye_matrix

            # for index in range(num_classes):
            new_dis = torch.distributions.multivariate_normal.MultivariateNormal(mean_embed_id, covariance_matrix=temp_precision)
            negative_samples = new_dis.rsample((args.sample_from,)).squeeze(1)
            prob_density = new_dis.log_prob(negative_samples)
            # breakpoint()
            # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
            # keep the data in the low density area.
            _, js_index_prob = torch.topk(- prob_density, x.size(0))
            cur_samples, index_prob = torch.topk(- prob_density, args.select)
            
            ood_samples = negative_samples[index_prob]
            ood_dis = negative_samples[js_index_prob]
            
            # add some gaussian noise
            # ood_samples = self.noise(ood_samples)
            # energy_score_for_fg = 1 * torch.logsumexp(predictions[0][selected_fg_samples][:, :-1] / 1, 1)
            energy_score_for_fg = log_sum_exp(x, 1)
            predictions_ood = net.fc(ood_samples)
            predictions_ood_dis = net.fc(ood_dis)
        
            energy_score_for_bg_dis = log_sum_exp(predictions_ood_dis, dim=1)
            
            div_calc = JSDivergenceCalculator(energy_score_for_fg, energy_score_for_bg_dis)
            lr_reg_loss = div_calc.calculate_js_divergence()
            
            if epoch % 5 == 0:
                print(f'epoch : {epoch}, lr_reg_loss : {lr_reg_loss}')
        else:
            if number_dict < args.m_sample_number:
                for index in range(output.size(0)):
                    if number_dict < args.m_sample_number:
                        data_dict[0][number_dict] = output[index].detach()
                        number_dict += 1
                    else:
                        break
                    
        # backward
        optimizer.zero_grad()
        loss = F.cross_entropy(x, target)
        # breakpoint()
        if epoch >= args.start_epoch:
            loss += args.loss_weight * (1/lr_reg_loss)
        
        loss.backward()
        
        optimizer.step()
        scheduler.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state['train_loss'] = loss_avg

# test function

def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)

# -------------------------

if args.test:
    test()
    print(state)
    exit()

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
'_' + str(args.loss_weight) + \
                             '_' + str(args.sample_number)+ '_' + str(args.start_epoch) + '_' +\
                            str(args.select) + '_' + str(args.sample_from) +
                                  '_baseline_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')

# Main loop
for epoch in range(start_epoch, args.epochs):
    if args.use_wandb and args.mode == 'VOS':
        print('set wandb complete!')
        wandb.init(project=args.project, name='WideResNet-task1-virtual train', notes=str(torch.cuda.get_device_name())+' x '+str(1))
    elif args.use_wandb and args.mode.startswith('ARES'):
        print('set wandb complete!')
        wandb.init(project=args.project, name=f'WideResNet-task1-{args.mode} train-{args.epochs}epochs seed-{args.seed}', notes=str(torch.cuda.get_device_name())+' x '+str(1))
    
    
    state['epoch'] = epoch

    begin_epoch = time.time()

    if args.mode == 'VOS':
        train(epoch)
    elif args.mode == 'ARES':
        train_ares(epoch)

    test()

    # Save model
    torch.save(net.state_dict(),
               os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
                            '_baseline'  + '_' + str(args.loss_weight) + \
                             '_' + str(args.m_sample_number)+ '_' + str(args.start_epoch) + '_' +\
                            str(args.select) + '_' + str(args.sample_from) + '_' + 'epoch_'  + str(epoch) + \
                                '_' + args.mode + '_' + str(args.epochs) + '_seed' + str(args.seed) + '.pt'))
    # Let us not waste space and delete the previous model
    prev_path = os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
                             '_baseline' + '_' + str(args.loss_weight) + \
                             '_' + str(args.m_sample_number)+ '_' + str(args.start_epoch) + '_' +\
                            str(args.select) + '_' + str(args.sample_from)  + '_' + 'epoch_' + str(epoch - 1) + \
                                '_' + args.mode + '_' + str(args.epochs) + '_seed' + str(args.seed) + '.pt')
    if os.path.exists(prev_path): os.remove(prev_path)

    # Show results

    with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
                                      '_' + str(args.loss_weight) + \
                                      '_' + str(args.m_sample_number) + '_' + str(args.start_epoch) + '_' + \
                                      str(args.select) + '_' + str(args.sample_from) +
                                      '_baseline_training_results_' + args.mode + '_' + str(args.epochs) + '_seed' + str(args.seed) + '.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'],
        ))

    # print state with rounded decimals
    # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100 - 100. * state['test_accuracy'])
    )
    
    if args.use_wandb:
        wandb.log({
            "Epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train loss": state['train_loss'],
            "test loss": state['test_loss'],
            "test_acc": state['test_accuracy']
            })