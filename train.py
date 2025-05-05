import argparse

import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import DINOHead, Hyperbolic_DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups

import wandb

def train(student, train_loader, test_loader, unlabelled_train_loader, args, optimizer, scheduler,
          best_test_acc = 0, start_epoch = 0):
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.GradScaler("cuda")

    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )
    
    best_test_acc_lab = best_test_acc
    freeze_curv_for_warmup = args.freeze_curvature.lower() == "warmup" and args.hyperbolic
    if freeze_curv_for_warmup:
        student[1].train_curvature(False)

    # # inductive
    # best_test_acc_lab = 0
    # # transductive
    # best_train_acc_lab = 0
    # best_train_acc_ubl = 0 
    # best_train_acc_all = 0

    for epoch in range(start_epoch, args.epochs):
        loss_record = AverageMeter()
        con_loss_record = AverageMeter()
        sup_con_loss_record = AverageMeter()
        cls_loss_record = AverageMeter()
        cluster_loss_record = AverageMeter()

        if epoch >= args.epochs_warmup and freeze_curv_for_warmup:
            student[1].train_curvature(True)
            freeze_curv_for_warmup = False
            print("Unfreezing curvature at epoch {}".format(epoch))

        # TODO: Check about checkpointing the teacher model too
        student.train()
        for batch_idx, batch in enumerate(train_loader):
            with torch.autograd.detect_anomaly(check_nan=True):
                step_log_dict = {}
                #print(batch_idx)

                images, class_labels, uq_idxs, mask_lab = batch
                mask_lab = mask_lab[:, 0]

                class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
                images = torch.cat(images, dim=0).cuda(non_blocking=True)

                with torch.cuda.amp.autocast(fp16_scaler is not None):
                    student_proj, student_out, output_log_stats = student(images)
                    teacher_out = student_out.detach()

                    step_log_dict["step/train/embed_mean"] = output_log_stats[0][0]
                    step_log_dict["step/train/embed_stddiv"] = output_log_stats[0][1]
                    step_log_dict["step/train/embed_max"] = output_log_stats[0][2]
                    step_log_dict["step/train/embed_min"] = output_log_stats[0][3]
                    if args.hyperbolic:
                        step_log_dict["step/train/curvature"] = student[1].get_curvature()
                        step_log_dict["step/train/proj_alpha"] = student[1].get_proj_alpha()
                        step_log_dict["step/train/hyp_lorentz_mean"] = output_log_stats[1][0]
                        step_log_dict["step/train/hyp_lorentz_stddiv"] = output_log_stats[1][1]
                        step_log_dict["step/train/hyp_lorentz_max"] = output_log_stats[1][2]
                        step_log_dict["step/train/hyp_lorentz_min"] = output_log_stats[1][3]
                        if args.poincare:
                            step_log_dict["step/train/hyp_poincare_mean"] = output_log_stats[2][0]
                            step_log_dict["step/train/hyp_poincare_stddiv"] = output_log_stats[2][1]
                            step_log_dict["step/train/hyp_poincare_max"] = output_log_stats[2][2]
                            step_log_dict["step/train/hyp_poincare_min"] = output_log_stats[2][3]
                        step_log_dict["step/train/hyp_logits_mean"] = output_log_stats[3-1*args.poincare][0]
                        step_log_dict["step/train/hyp_logits_stddiv"] = output_log_stats[3-1*args.poincare][1]
                        step_log_dict["step/train/hyp_logits_max"] = output_log_stats[3-1*args.poincare][2]
                        step_log_dict["step/train/hyp_logits_min"] = output_log_stats[3-1*args.poincare][3]
                    else:
                        step_log_dict["step/train/logits_mean"] = output_log_stats[1][0]
                        step_log_dict["step/train/logits_stddiv"] = output_log_stats[1][1]
                        step_log_dict["step/train/logits_max"] = output_log_stats[1][2]
                        step_log_dict["step/train/logits_min"] = output_log_stats[1][3]

                    # clustering, sup
                    sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                    sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                    cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                    # clustering, unsup
                    cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                    avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                    me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                    cluster_loss += args.memax_weight * me_max_loss

                    # represent learning, unsup

                    if args.hyperbolic:
                        contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj, args=args, curv=student[1].get_curvature(), DEBUG_DIR=DEBUG_DIR)
                    else:
                        contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj, args=args, DEBUG_DIR=DEBUG_DIR)
                    if contrastive_logits is None:
                        wandb.log(step_log_dict)
                        ValueError('Hyperbolic distance has NaN')
                    
                    # TODO: Do we need to use a hyperbolic cross entropy loss? I forgot to consider that.
                    contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                    # representation learning, sup
                    student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                    student_proj = student_proj if args.hyperbolic else torch.nn.functional.normalize(student_proj, dim=-1)
                    sup_con_labels = class_labels[mask_lab]
                    sup_con_loss, SCL_log_stats = SupConLoss(hyperbolic=args.hyperbolic)(student_proj, labels=sup_con_labels,
                                                            curv = student[1].get_curvature() if args.hyperbolic else None,
                                                            DEBUG_DIR = DEBUG_DIR)

                    pstr = ''
                    pstr += f'cls_loss: {cls_loss.item():.4f} '
                    pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                    pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                    pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

                    loss = 0
                    loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                    if loss.isnan():
                        print(f"Loss is NaN. cluster_loss is: {cluster_loss}, cls_loss is: {cls_loss}")
                    loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                    if loss.isnan():
                        print(f"Loss is NaN. con_loss is: {contrastive_loss}, sup_con_loss is: {sup_con_loss}")
                    
                # Train acc
                loss_record.update(loss.item(), class_labels.size(0))
                con_loss_record.update(contrastive_loss.item(), class_labels.size(0))
                sup_con_loss_record.update(sup_con_loss.item(), class_labels.size(0))
                cls_loss_record.update(cls_loss.item(), class_labels.size(0))
                cluster_loss_record.update(cluster_loss.item(), class_labels.size(0))
                step_log_dict["step/train/contrastive_loss"] = contrastive_loss.item()
                step_log_dict["step/train/sup_con_loss"] = sup_con_loss.item()
                step_log_dict["step/train/cls_loss"] = cls_loss.item()
                step_log_dict["step/train/cluster_loss"] = cluster_loss.item()
                step_log_dict["step/train/full_loss"] = loss.item()
                step_log_dict["debug/step/train/SCL_logits_mean"] = SCL_log_stats[0][0]
                step_log_dict["debug/step/train/SCL_logits_stddiv"] = SCL_log_stats[0][1]
                step_log_dict["debug/step/train/SCL_logits_max"] = SCL_log_stats[0][2]
                step_log_dict["debug/step/train/SCL_logits_min"] = SCL_log_stats[0][3]
                step_log_dict["debug/step/train/SCL_exp_logits_mean"] = SCL_log_stats[1][0]
                step_log_dict["debug/step/train/SCL_exp_logits_stddiv"] = SCL_log_stats[1][1]
                step_log_dict["debug/step/train/SCL_exp_logits_max"] = SCL_log_stats[1][2]
                step_log_dict["debug/step/train/SCL_exp_logits_min"] = SCL_log_stats[1][3]
                step_log_dict["debug/step/train/SCL_exp_logits_masked_mean"] = SCL_log_stats[2][0]
                step_log_dict["debug/step/train/SCL_exp_logits_masked_stddiv"] = SCL_log_stats[2][1]
                step_log_dict["debug/step/train/SCL_exp_logits_masked_max"] = SCL_log_stats[2][2]
                step_log_dict["debug/step/train/SCL_exp_logits_masked_min"] = SCL_log_stats[2][3]
                step_log_dict["debug/step/train/SCL_log_prob_mean"] = SCL_log_stats[3][0]
                step_log_dict["debug/step/train/SCL_log_prob_stddiv"] = SCL_log_stats[3][1]
                step_log_dict["debug/step/train/SCL_log_prob_max"] = SCL_log_stats[3][2]
                step_log_dict["debug/step/train/SCL_log_prob_min"] = SCL_log_stats[3][3]
                step_log_dict["debug/step/train/SCL_log_prob_masked_mean"] = SCL_log_stats[4][0]
                step_log_dict["debug/step/train/SCL_log_prob_masked_stddiv"] = SCL_log_stats[4][1]
                step_log_dict["debug/step/train/SCL_log_prob_masked_max"] = SCL_log_stats[4][2]
                step_log_dict["debug/step/train/SCL_log_prob_masked_min"] = SCL_log_stats[4][3]
                
                optimizer.zero_grad()
                if fp16_scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    fp16_scaler.scale(loss).backward()
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()
                wandb.log(step_log_dict)

                if batch_idx % args.print_freq == 0:
                    args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                                .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        epoch_log_dict = {"epoch": epoch+1, "epoch/train/loss": loss_record.avg,  "epoch/train/learning_rate": scheduler.get_last_lr()[0],
                          "epoch/train/contrstive_loss": con_loss_record.avg, "epoch/train/sup_con_loss": sup_con_loss_record.avg,
                          "epoch/train/cls_loss": cls_loss_record.avg, "epoch/train/cluster_loss": cluster_loss_record.avg}

        if args.hyperbolic:
            print(f"Current curvature: {student[1].get_curvature()}")
            print(f"Current projection weight: {student[1].get_proj_alpha()}")
            epoch_log_dict["epoch/train/curvature"] = student[1].get_curvature()
            epoch_log_dict["epoch/train/proj_alpha"] = student[1].get_proj_alpha()

        if loss.isnan():
            break

        args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc = test(student, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
        # args.logger.info('Testing on disjoint test set...')
        # all_acc_test, old_acc_test, new_acc_test = test(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)

        epoch_log_dict["epoch/all_acc"] = all_acc
        epoch_log_dict["epoch/old_acc"] = old_acc
        epoch_log_dict["epoch/new_acc"] = new_acc

        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        # args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

        # Step schedule
        scheduler.step()

        if old_acc > best_test_acc_lab:
            print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                    new_acc))

            torch.save(student.state_dict(), args.model_path[:-3] + f'_best.pt')
            print("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))
            wandb.save(args.model_path[:-3] + f'_best.pt')

            best_test_acc_lab = old_acc

        #args_copy = Namespace(**vars(args))
        #args_copy.writer = None
        torch.save({
            "epoch": epoch+1,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            #"arguments": args,
            "wandb_run_id": wandb.run.id,
            "best_test_acc": best_test_acc_lab,
        }, args.model_path)
        print("model saved to {}.".format(args.model_path))

        wandb.log(epoch_log_dict)
        wandb.save(args.model_path)

        args.logger.info("model saved to {}.".format(args.model_path))

        # if old_acc_test > best_test_acc_lab:
        #     
        #     args.logger.info(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
        #     args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        #     
        #     torch.save(save_dict, args.model_path[:-3] + f'_best.pt')
        #     args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))
        #     
        #     # inductive
        #     best_test_acc_lab = old_acc_test
        #     # transductive            
        #     best_train_acc_lab = old_acc
        #     best_train_acc_ubl = new_acc
        #     best_train_acc_all = all_acc
        # 
        # args.logger.info(f'Exp Name: {args.exp_name}')
        # args.logger.info(f'Metrics with best model on test set: All: {best_train_acc_all:.4f} Old: {best_train_acc_lab:.4f} New: {best_train_acc_ubl:.4f}')


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
                                                    args=args)

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
    parser.add_argument('--hyperbolic', action='store_true', default=False)
    parser.add_argument('--poincare', action='store_true', default=False)
    parser.add_argument('--curvature', type=float, default=1.0)
    parser.add_argument('--freeze_curvature', type=str, default="false")
    parser.add_argument('--proj_alpha', type=float, default=1.7035**-1)
    parser.add_argument('--freeze_proj_alpha', type=str, default="false")
    parser.add_argument('--checkpoint_path', type=str)

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

    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    
    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    # test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
    #                                   batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    if args.hyperbolic:
        projector = Hyperbolic_DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim,
                                                               nlayers=args.num_mlp_layers, curv_init=args.curvature,
                                                               learn_curv=not args.freeze_curvature.lower() == "full",
                                                               alpha_init=args.proj_alpha,
                                                               learn_alpha=not args.freeze_proj_alpha.lower() == "full",
                                                               poincare=args.poincare)
    else:
        projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model = nn.Sequential(backbone, projector).to(device)

    # ----------------------
    # OPTIMIZER AND SCHEDULER
    # ----------------------
    params_groups = get_params_groups(model)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )

    # ----------------------
    # LOAD CHECKPOINT
    # ----------------------
    checkpoint = {}
    start_epoch = 0
    best_test_acc = 0
    if not args.checkpoint_path is None:
            checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
            if not "model_state_dict" in checkpoint.keys():
                model.load_state_dict(checkpoint)
            else:
                #checkpoint = torch.load(args.checkpoint_path)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                start_epoch = checkpoint["epoch"]
                best_test_acc = checkpoint["best_test_acc"]

    # ----------------------
    # INITIALIZE WANDB
    # ----------------------
    wandb.login()
    if checkpoint.get("wandb_run_id", None) is None or args.wandb_mode != "online":# or args.wandb_new_id:
        wandb.init(config = args,
                #dir = args.save_dir + '/wandb_logs',
                dir = "wandb_logs/",
                project = 'Hyperbolic_SimGCD',
                name = args.exp_id + '-' + str(args.seed),
                mode = args.wandb_mode)
    else:
        wandb.init(config = args,
                #dir = args.save_dir + '/wandb_logs',
                dir = "wandb_logs/",
                project = 'Hyperbolic_SimGCD',
                name = args.exp_id + '-' + str(args.seed),
                id = checkpoint["wandb_run_id"],
                resume = 'must',
                mode = args.wandb_mode)
    wandb.watch(model, log="all", log_graph=True, log_freq=1)

    # ----------------------
    # TRAIN
    # ----------------------
    # train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)
    train(model, train_loader, None, test_loader_unlabelled, args, optimizer, scheduler,
          best_test_acc=best_test_acc, start_epoch=start_epoch)
