from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
#from testers import *
import random
import pandas as pd
import numpy as np

# from resnet32_cifar10_grasp import resnet32
# from vgg_grasp import vgg19, vgg16, vgg11, vgg13
# from resnet20_cifar import resnet20
# from resnet50_cifar100 import ResNet50, ResNet18, ResNet34

# from models.vanilla_resnet_slow import *
# resnet34S = resnet34
# resnet18S = resnet18
# from models.vanilla_resnet import *
# from models.small_model import Small_Net, Small_Net_2
# from models.resnet20_imagenet import resnet20
# from models.vgg_grasp import vgg11, vgg13, vgg16, vgg19

from models.vanilla_resnet_slow import resnet18S, resnet34S
from models.vanilla_resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.vgg_grasp import *
from models.small_model import *
from models.resnet20_imagenet import resnet20
from models.builder import build_model
from set_dataset import *
from models.cifar.resnet20_cifar10 import resnet20 as resnet20_cifar10
from models.cifar.resnet32_cifar10_grasp import wide_resnet32_cifar10 as wide_resnet32_cifar10
from models.cifar.resnet50_cifar10 import ResNet50 as resnet50_cifar10
from models.cifar.vgg_grasp_cifar10 import vgg11_cifar10, vgg13_cifar10, vgg16_cifar10, vgg19_cifar10

from models.mobilenetV2_imagenet import MobileNetV2 as mobilenet_v2
from models.efficientnet_pytorch import EfficientNet
from models.googlenet_pytorch import GoogLeNet

parser = argparse.ArgumentParser(description='PyTorch CIFAR training')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')
parser.add_argument('--sparsity_type', type=str, default='column',
                    help="define sparsity_type: [irregular,column,filter]")
parser.add_argument('--config_file', type=str, default='config_vgg16',
                    help="define sparsity_type: [irregular,column,filter]")

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

from time import time
print(time())
random.seed(int(time()))
torch.manual_seed(int(time()))
torch.cuda.manual_seed(int(time()))

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, download=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * float(correct) / float(len(test_loader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy


def test_model_size(model):

    # --------------------- total sparsity --------------------
    total_FC = 0
    total_zeros = 0
    total_nonzeros = 0


    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4):# and "shortcut" not in name):  # only consider conv layers
            zeros = np.sum(weight.cpu().detach().numpy() == 0)
            total_zeros += zeros
            non_zeros = np.sum(weight.cpu().detach().numpy() != 0)
            total_nonzeros += non_zeros

        if (len(weight.size()) == 2):
            weight = weight.reshape(-1)
            total_FC += weight.shape[0]

    total = total_zeros + total_nonzeros + total_FC
    total_nonzeros_w_fc = total_nonzeros + total_FC

    comp_ratio = float((total_zeros + total_nonzeros)) / float(total_nonzeros)

    print("---------------------------------------------------------------------------")
    print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
        total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
    print("only consider conv layers, compression rate is: {:.4f}".format(
        (total_zeros + total_nonzeros) / total_nonzeros))
    print("total_zero:", total_zeros)
    print("total conv:", total_zeros+total_nonzeros)
    print("total FC:", total_FC)
    print("original_total_with FC:", total)
    print("AfterPrune_total_with_FC:", total_nonzeros_w_fc)
    print("===========================================================================\n\n")



def main():

    seed = 914
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print("Using manual seed:", seed)

    # model = resnet20()
    # model.load_state_dict(torch.load("./checkpoints/resnet20/prune_iter/cifar10/round_11_sp0.914/seed914_64_lr_0.01_resnet20_cifar10_acc_88.240_sgd_lr0.01_default_sp0.913_epoch156.pt"))

    # model = resnet32(dataset="cifar10")
    # model.load_state_dict(
    #     torch.load("./checkpoints/resnet32/irr/seed914_irr_0.8_uneven_cifar10_resnet32_acc_92.520_sgd_lr0.1_cosine_epoch184.pt"))
    # model.cuda()

    # model = vgg19()

    print("\n------------------------------\n")

    # comp_ratio = test_sparsity(model, column=True, channel=True, filter=True, kernel=False)



    from measure_model import measure_model

    measure_df = pd.DataFrame(columns=["model", "params", "flops"])
    efficient_models = [ "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3", "efficientnet-b4",\
                        "efficientnet-b5", "efficientnet-b6", "efficientnet-b7"]
    model_names = ["resnet18", "resnet19", "resnet20", "resnet34", "resnet35", "resnet50", "resnet101", "resnet152", \
                    "small2", "small3" ,"small4", "small5", "small6" ,"small7", "small8", "small9" ,"small10", \
                        "vgg11", "vgg13", "vgg16", "vgg19", "vgg5",  "vgg6",  "vgg7",  "vgg51",  "vgg61",  "vgg71",  "vgg52",  "vgg62",  "vgg72", 
                        "mobilenet_v2", "googlenet1"]
    models = [resnet18(), resnet18S(), resnet20(), resnet34(), resnet34S(), resnet50(), resnet101(), resnet152(), \
            small2(), small3(), small4(), small5(), small6(), small7(), small8(), small9(), small10(),\
            vgg11(), vgg13(), vgg16(), vgg19(), vgg5(), vgg6(), vgg7(), vgg51(), vgg61(), vgg71(), vgg52(), vgg62(), vgg72(),\
            mobilenet_v2(), GoogLeNet.from_name("googlenet")]
    
    for model_name in ["resnet", "smresnet", "mdresnet"]:
        for i in range(10, 36, 2):
            model_names.append("{}{}".format(model_name, i))
            models.append(build_model(model_name, i))

    for model_name in ["cifar10-resnet"]:
        for i in [20, 32, 50]:
            model_names.append("{}{}".format(model_name, i))
    
    for model_name in ["cifar10-vgg"]:
        for i in [11, 13, 16, 19]:
            model_names.append("{}{}".format(model_name, i))
    
    model_names.extend(efficient_models)

    models.extend([resnet20_cifar10(dataset="cifar10"), wide_resnet32_cifar10(depth=32, dataset='cifar10'), resnet50_cifar10(), \
                    vgg11_cifar10(), vgg13_cifar10(), vgg16_cifar10(), vgg19_cifar10()])
    models.extend([EfficientNet.from_name(m) for m in efficient_models])

    for model_name, model in zip(model_names, models) :
        print("------------------------------")
        print(model_name)
        try:
            if "cifar10" in model_name:
                count_ops, count_params, count_conv = measure_model(model, inp_shape=(3, 32, 32))
            else:
                count_ops, count_params, count_conv = measure_model(model, inp_shape=(3, 224, 224))

            print("Flops = %.2f M" % (count_ops / 1e6))
            print("Params = %.2f M" % (count_params / 1e6))

            measure_df.loc[len(measure_df),:] = np.array([model_name, count_params, count_ops])
        except Exception as e:
            print("error, skipped")
            print(e)

    return measure_df
    # model = resnet20()
    # model = resnet32()
    # model = vgg11()
    # model = vgg13()
    # model = vgg16()
    # model = vgg19()
    # model = ResNet18()
    # model = ResNet34()
    # model = ResNet50()

    # model = model.cuda()

    # count_ops, count_params, count_conv = measure_model(model, inp_shape=(3, 224, 224))

    # print("Flops = %.2f M" % (count_ops / 1e6))
    # print("Params = %.2f M" % (count_params / 1e6))



if __name__ == '__main__':
    df = main()
    df.to_csv("models.csv", index=False)
