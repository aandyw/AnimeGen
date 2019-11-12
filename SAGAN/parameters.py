import argparse


def get_parameters():

    parser = argparse.ArgumentParser()

    # TRAINING SETTINGS & HYPERPARAMETERS

    # number of workers for dataloader
    parser.add_argument('--num_workers', type=int, default=2)
    # batch size for training
    parser.add_argument('--batch_size', type=int, default=32)
    # spatial size of training images
    parser.add_argument('--imsize', type=int, default=64)
    # channels of training images
    parser.add_argument('--nc', type=int, default=3)
    # size of z latent vector (gen input)
    parser.add_argument('--nz', type=int, default=100)
    # size of feature maps in gen
    parser.add_argument('--ngf', type=int, default=64)
    # size of feature maps in disc
    parser.add_argument('--ndf', type=int, default=64)
    # number of training epochs
    parser.add_argument('--total_steps', type=int, default=150000)
    # disc iterations
    parser.add_argument('--d_iters', type=int, default=1)
    # gen iterations
    parser.add_argument('--g_iters', type=int, default=1)
    # gen learning rate
    parser.add_argument('--g_lr', type=float, default=0.0001)
    # disc learning rate
    parser.add_argument('--d_lr', type=float, default=0.0004)
    # learning rate decay
    parser.add_argument('--lr_decay', type=float, default=0.95)
    # beta hyperparams for optimizers
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.9)

    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--plot', type=bool, default=False)

    # PATHS
    parser.add_argument('--image_path', type=str, default="../data")
    parser.add_argument('--model_path', type=str, default="./models")
    parser.add_argument('--sample_path', type=str, default="./samples")

    # DATA
    parser.add_argument('--visualize', type=bool, default=False)

    # STEPS
    parser.add_argument('--log_step', type=int, default=20)
    parser.add_argument('--save_epoch', type=int, default=10)

    # PRETRAIN
    parser.add_argument('--pretrained_model', type=int, default=222)

    return parser.parse_args()
