import argparse

def parse_opt():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
    parser.add_argument('--dataset', type=str, default='cifar100', help='training dataset (default: cifar100)')
    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true', help='train with channel sparsity regularization')
    parser.add_argument('--s', type=float, default=0.0001, help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--refine', default='', type=str, metavar='PATH', help='path to the pruned model to be fine tuned')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N', help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=160, metavar='N', help='number of epochs to train (default: 160)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save', default='./logs', type=str, metavar='PATH', help='path to save prune model (default: current directory)')
    parser.add_argument('--arch', default='vgg', type=str,  help='architecture to use')
    parser.add_argument('--depth', default=19, type=int, help='depth of the neural network')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_opt()
    print(args)
