import argparse

import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import DINOHead, Hyperbolic_DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups

def test(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits, _ = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args, print_output=True)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)

    parser.add_argument('--exp_id', default=None, type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--wandb_mode', type=str, default="online")
    parser.add_argument('--epochs_warmup', default=2, type=int)
    parser.add_argument('--norm_last_layer', action='store_true', default=False)
    parser.add_argument('--hyperbolic', action='store_true', default=False)
    parser.add_argument('--poincare', action='store_true', default=False)
    parser.add_argument('--original_poincare_layer', action='store_true', default=False)
    parser.add_argument('--euclidean_clipping', type=float, default=None)
    parser.add_argument('--curvature', type=float, default=1.0)
    parser.add_argument('--freeze_curvature', type=str, default="false")
    parser.add_argument('--proj_alpha', type=float, default=1.7035**-1)
    parser.add_argument('--freeze_proj_alpha', type=str, default="false")
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--angle_loss', action='store_true', default=False)
    parser.add_argument('--max_angle_loss_weight', type=float, default=0.5)
    parser.add_argument('--decay_angle_loss_weight', action='store_true', default=False)
    parser.add_argument('--use_adam', action='store_true', default=False)
    #parser.add_argument('--mlp_out_dim', type=int, default=768)
    parser.add_argument('--use_dinov2', action='store_true', default=False)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--avg_grad_norm', type=float, default=2.5)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['simgcd'], exp_id=args.exp_id)
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    DEBUG_DIR = args.debug_dir

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vitb14') if args.use_dinov2 else torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    _, test_dataset, _, datasets = get_datasets(args.dataset_name,
                                                train_transform,
                                                test_transform,
                                                args)

    # --------------------
    # DATALOADERS
    # --------------------
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    if args.hyperbolic:
        projector = Hyperbolic_DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, norm_last_layer=args.norm_last_layer,
                                                               nlayers=args.num_mlp_layers, curv_init=args.curvature,
                                                               learn_curv=not args.freeze_curvature.lower() == "full",
                                                               alpha_init=args.proj_alpha,
                                                               learn_alpha=not args.freeze_proj_alpha.lower() == "full",
                                                               poincare=args.poincare, euclidean_clip_value=args.euclidean_clipping,
                                                               original_poincare_layer=args.original_poincare_layer)
    else:
        projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model = nn.Sequential(backbone, projector).to(device)

    # ----------------------
    # LOAD CHECKPOINT
    # ----------------------
    checkpoint = {}
    start_epoch = 0
    best_test_acc = 0
    best_loss = 1e10
    if not args.checkpoint_path is None:
            print("Loading checkpoint from {}".format(args.checkpoint_path))
            checkpoint = torch.load(args.checkpoint_path)#["model_state_dict"]
            if checkpoint.get('model_state_dict', None) is not None:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)

    # ----------------------
    # TEST
    # ----------------------
    print("Testing...")
    all_acc, old_acc, new_acc = test(model, test_loader_labelled, epoch=0, save_name='Test ACC', args=args)
    print(f"Test ACC: {all_acc:.4f}, Old ACC: {old_acc:.4f}, New ACC: {new_acc:.4f}")