
import os
import torch
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=72,
                        help="number of different places in a batch, each place contain 4 images, total batch size = train_batch_size*4")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001, help="_")
    parser.add_argument("--optim", type=str, default="adam", help="_", choices=["adam", "sgd"])
    parser.add_argument("--epochs_num", type=int, default=50,
                        help="number of epochs to train for")
    parser.add_argument('--crica_path', type=str, default='./logs/default/dinob_[8,9,10,11]_768_crica/last_model.pth',
                        help="加载crica weight的path")
    # Inference parameters
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (caching and testing)")
    # Model parameters
    parser.add_argument('--backbone_arch', type=str, default='dinov2_vitl14',
                        help="backbone_arch type, base(b) or large(l)")
    parser.add_argument('--layer1', type=int, default=23,
                        help="backbone不用训练的层数，默认是large的23层全部固定，或者base的11层全部固定")
    parser.add_argument('--out_indices', type=int, default=[23],nargs="+",
                        help="输出backbone的第23层即最后一层对应linear cls结果，或者[20, 21, 22, 23]对应ms cls结果；或者base的[11],[8,9,10,11]")
    parser.add_argument('--backbone_out_dim', type=int, default=1024,
                        help="输出backbone的维度，23对应1024，[20, 21, 22, 23]对应4096；base对应768或者3072")
    parser.add_argument('--mix_in_dim', type=int, default=1024,
                        help="输入mixer的维度，我自己设定的1024哈")
    parser.add_argument('--token_num', type=int, default=1,
                        help="token mixer个数")
    parser.add_argument('--token_ratio', type=int, default=1,
                        help="token mixer内部维度比例")
    parser.add_argument('--token_mix_ratio', type=int, default=1,
                        help="mixer中token mix内部维度比例")
    parser.add_argument('--mix_hidden_channels', type=int, default=768,
                        help="mixer中channel mix部分内部维度")
    parser.add_argument('--distill_num', type=int, default=1,
                        help="mixer个数")
    parser.add_argument('--encoder_in_dim', type=int, default=1024,
                        help="输入encoder的维度，我自己设定的1024哈")
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--foundation_model_path", type=str, default=None,
                        help="Path to load foundation model checkpoint.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    # Other parameters   
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=16, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[224, 224], nargs=2, help="Resizing shape for images (HxW).")
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize"],
                        help="This includes pre/post-processing methods and prediction refinement")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="_")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10, help="_")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 100], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    # Paths parameters
    parser.add_argument("--eval_datasets_folder", type=str, default=None, help="Path with all datasets")
    parser.add_argument("--eval_dataset_name", type=str, default="pitts30k", help="Relative path of the dataset")
    parser.add_argument('--pca_dim', type=int, default=None, help="PCA dimension (number of principal components). If None, PCA is not used.")
    parser.add_argument("--pca_dataset_folder", type=str, default=None,
                        help="Path with images to be used to compute PCA (ie: pitts30k/images/train")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="Folder name of the current run (saved in ./logs/)")
    args = parser.parse_args()
    
    if args.eval_datasets_folder == None:
        try:
            args.eval_datasets_folder = os.environ['DATASETS_FOLDER']
        except KeyError:
            raise Exception("You should set the parameter --datasets_folder or export " +
                            "the DATASETS_FOLDER environment variable as such \n" +
                            "export DATASETS_FOLDER=../datasets_vg/datasets")
    
    if args.pca_dim != None and args.pca_dataset_folder == None:
        raise ValueError("Please specify --pca_dataset_folder when using pca")
    
    return args

