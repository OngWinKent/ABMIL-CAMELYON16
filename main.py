from __future__ import print_function
import argparse
import utils

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--root', type=str, default= "./datasets",
                    help='dataset root directory')
parser.add_argument('--epoch_num', type=int, default=10,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate (default: 0.0005)')
parser.add_argument('--weight_decay', type=float, default=10e-5,
                    help='weight decay')
parser.add_argument('--in_features', type=int, default=1024, 
                    help='Attenton model input feature size')
parser.add_argument('--patch_emb_size', type=int, default=500, 
                    help='Patch embedding size')
parser.add_argument('--attn_hid_size', type=int, default=128, 
                    help='Attention hidden size')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--model_name', type=str, default='attention', help='Attention model selection', 
                    choices= ["attention", "gated_attention"])
args = parser.parse_args()


if __name__ == "__main__":
    # Init device
    is_cuda = utils.init_device(seed= args.seed)

    # Load dataset as loader
    train_loader, test_loader = utils.load_dataset(root= args.root)

    # Attention model initialization
    model = utils.init_model(model_name= args.model_name, 
                            in_features= args.in_features,
                            patch_emb_size= args.patch_emb_size,
                            attn_hid_size= args.attn_hid_size)

    # Model training
    train_params = {'epoch_num': args.epoch_num, 'lr': args.lr, 'weight_decay': args.weight_decay}
    utils.train(model= model, train_loader= train_loader, is_cuda= is_cuda, train_params= train_params)
    # Running inference on test dataset
    utils.inference(model= model, test_loader= test_loader, is_cuda= is_cuda, show_plot= True)
    