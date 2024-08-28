import os
import shutil

import argparse
import numpy as np
import pandas as pd
import torch
import torchvision

from sklearn.utils import class_weight

from datasets import *
from utils import *
from losses import *
from torch.optim.lr_scheduler import CosineAnnealingLR
def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda:0')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
force_cudnn_initialization()

def main(args):
    # Set model/output directory name
    MODEL_NAME = args.dataset
    MODEL_NAME += f'_{args.model_name}'
    MODEL_NAME += f'_rand' if args.rand_init else ''
    MODEL_NAME += f'_mixup-{args.mixup_alpha}' if args.mixup else ''
    MODEL_NAME += f'_decoupling-{args.decoupling_method}' if args.decoupling_method != '' else ''
    MODEL_NAME += f'_rw-{args.rw_method}' if args.rw_method != '' else ''
    MODEL_NAME += f'_{args.loss}'
    MODEL_NAME += '-drw' if args.drw else ''
    MODEL_NAME += f'_cb-beta-{args.cb_beta}' if args.rw_method == 'cb' else ''
    MODEL_NAME += f'_fl-gamma-{args.fl_gamma}' if args.loss == 'focal' else ''
    MODEL_NAME += f'_lr-{args.lr}'
    MODEL_NAME += f'_bs-{args.batch_size}'
    if args.decoupling_method != '':
        dec_weights = args.decoupling_weights.split('/chkpt')[0].split('/')[-1]
        MODEL_NAME += f"_from---{dec_weights}---"
    if args.use_hol_new:
        MODEL_NAME+=f"lam-holnew-{args.lambda_hol}-{args.lambda_hol_syn}"
    if args.seed!=0:
        MODEL_NAME+=f'seed_{args.seed}'
    if args.no_es:
        MODEL_NAME+='_noES'
    if args.lrScheduler is not None:
        MODEL_NAME+=f'_{args.lrScheduler}'
    if args.lr!=1e-4:
        MODEL_NAME+=f'_lr{args.lr}'
    if args.minLR!=0:
        MODEL_NAME+=f'_minLR{args.minLR}'
    if args.optim!='adam':
        MODEL_NAME+=f'_{args.optim}'
        
        
    # Create output directory for model (and delete if already exists)
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    model_dir = os.path.join(args.out_dir, MODEL_NAME)
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    print(model_dir)
    # Set all seeds for reproducibility
    set_seed(args.seed)

    # Create datasets + loaders
    if args.dataset=="Skin_dataset_similar_PIL":
        dataset=Skin_dataset_similar_PIL
        N_CLASSES=16
        TAIL_CLASSES=['nontumor_skin_dermis_dermis', 'nontumor_skin_epidermis_epidermis', 'tumor_skin_naevus_naevus', 'nontumor_skin_subcutis_subcutis', 'nontumor_skin_sebaceousglands_sebaceousglands', 'tumor_skin_epithelial_bcc', 'nontumor_skin_muscle_skeletal', 'nontumor_skin_chondraltissue_chondraltissue', 'nontumor_skin_sweatglands_sweatglands', 'nontumor_skin_necrosis_necrosis', 'nontumor_skin_vessel_vessel', 'nontumor_skin_elastosis_elastosis']
    elif args.dataset=="Skin_dataset_PIL":
        dataset=Skin_dataset_PIL
        N_CLASSES=16
        TAIL_CLASSES=['nontumor_skin_dermis_dermis', 'nontumor_skin_epidermis_epidermis', 'tumor_skin_naevus_naevus', 'nontumor_skin_subcutis_subcutis', 'nontumor_skin_sebaceousglands_sebaceousglands', 'tumor_skin_epithelial_bcc', 'nontumor_skin_muscle_skeletal', 'nontumor_skin_chondraltissue_chondraltissue', 'nontumor_skin_sweatglands_sweatglands', 'nontumor_skin_necrosis_necrosis', 'nontumor_skin_vessel_vessel', 'nontumor_skin_elastosis_elastosis']
    elif args.dataset=="Skin_dataset_similar_PIL_real_and_syn_targeted_maxconf_100":
        dataset=Skin_dataset_similar_PIL_real_and_syn_targeted_maxconf_100
        N_CLASSES=16
        TAIL_CLASSES=['nontumor_skin_dermis_dermis', 'nontumor_skin_epidermis_epidermis', 'tumor_skin_naevus_naevus', 'nontumor_skin_subcutis_subcutis', 'nontumor_skin_sebaceousglands_sebaceousglands', 'tumor_skin_epithelial_bcc', 'nontumor_skin_muscle_skeletal', 'nontumor_skin_chondraltissue_chondraltissue', 'nontumor_skin_sweatglands_sweatglands', 'nontumor_skin_necrosis_necrosis', 'nontumor_skin_vessel_vessel', 'nontumor_skin_elastosis_elastosis']
    else:
        raise NotImplementedError

    train_dataset = dataset(data_dir=args.data_dir, label_dir=args.label_dir, split='train')
    val_dataset = dataset(data_dir=args.data_dir, label_dir=args.label_dir, split='val-10-100-noaug') 
    bal_test_dataset = dataset(data_dir=args.data_dir, label_dir=args.label_dir, split='balanced-test')
    test_dataset = dataset(data_dir=args.data_dir, label_dir=args.label_dir, split='test')


    if args.decoupling_method == 'cRT':
        cls_weights = [len(train_dataset) / cls_count for cls_count in train_dataset.cls_num_list]
        instance_weights = [cls_weights[label] for label in train_dataset.labels]
        sampler = torch.utils.data.WeightedRandomSampler(torch.Tensor(instance_weights), len(train_dataset))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn, sampler=sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=val_worker_init_fn)
    bal_test_loader = torch.utils.data.DataLoader(bal_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=val_worker_init_fn)
    
    ood_loader = None

    ood_classes_train = [i for i, c in enumerate(train_dataset.CLASSES) if c in TAIL_CLASSES] 
    id_classes_train = [i for i, c in enumerate(train_dataset.CLASSES) if c not in TAIL_CLASSES]
    print('OOD classes train: ',ood_classes_train)
    print('ID classes train: ',id_classes_train)
    ood_classes = [i for i, c in enumerate(val_dataset.CLASSES) if c in TAIL_CLASSES] 
    id_classes = [i for i, c in enumerate(val_dataset.CLASSES) if c not in TAIL_CLASSES]
    print('OOD classes val: ',ood_classes)
    print('ID classes val: ',id_classes)    
    history = pd.DataFrame(columns=['epoch', 'phase', 'loss', 'balanced_acc', 'mcc', 'auroc', 'auroc_ood', 'fpr_ood','balanced_acc_head','balanced_acc_tail', 'loss_head', 'loss_tail', 'loss_extra'])
    history.to_csv(os.path.join(model_dir, 'history.csv'), index=False)
    # Set device
    device = torch.device('cuda:0')

    # Instantiate model
    model = torchvision.models.resnet50(pretrained=(not args.rand_init))
    model.fc = torch.nn.Linear(model.fc.in_features, N_CLASSES)
    
    if args.decoupling_method == 'tau_norm': # currently only usable for 20-class training to 20-class decoupling
        msg = model.load_state_dict(torch.load(args.decoupling_weights, map_location='cpu')['weights'])
        print(f'Loaded weights from {args.decoupling_weights} with message: {msg}')
        
        model.fc.bias.data = torch.zeros_like(model.fc.bias.data)
        fc_weights = model.fc.weight.data.clone()

        weight_norms = torch.norm(fc_weights, 2, 1)

        model.fc.weight.data = torch.stack([fc_weights[i] / torch.pow(weight_norms[i], -4) for i in range(N_CLASSES)], dim=0)
    elif args.decoupling_method == 'cRT': # decoupling usable for all classes irrespective of pretraining

        msg = model.load_state_dict(torch.load(args.decoupling_weights, map_location='cpu')['weights'])
        print(f'Loaded weights from {args.decoupling_weights} with message: {msg}')

        model.fc = torch.nn.Linear(model.fc.in_features, N_CLASSES)  # re-initialize classifier head

    model = model.to(device)        

    # Set loss and weighting method
    if args.rw_method == 'sklearn':
        weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_dataset.labels), y=np.array(train_dataset.labels))
        weights = torch.Tensor(weights).to(device)
    elif args.rw_method == 'cb':
        weights = get_CB_weights(samples_per_cls=train_dataset.cls_num_list, beta=args.cb_beta)
        weights = torch.Tensor(weights).to(device)
    else:
        weights = None

    if weights is None:
        print('No class reweighting')
    else:
        print(f'Class weights with rw_method {args.rw_method}:')
        for i, c in enumerate(train_dataset.CLASSES):
            print(f'\t{c}: {weights[i]}')

    loss_fxn = get_loss(args, None if args.drw else weights, train_dataset)

    # Set optimizer
    optim = torch.optim.AdamW if args.optim=='adamw' else torch.optim.Adam
    if args.decoupling_method != '':
        optimizer = optim(model.fc.parameters(), lr=args.lr)    
    else:
        optimizer = optim(model.parameters(), lr=args.lr)
    optimizer.sam_opt=False # for logging
   
    if args.lrScheduler=='cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.minLR) # eta_min is the minimum LR.
    else: 
        scheduler=None # ConstantLR(optimizer, factor=1., total_iters=args.max_epochs)

    # Train with early stopping
    if args.decoupling_method != 'tau_norm':
        epoch = 1
        early_stopping_dict = {'best_acc': 0., 'epochs_no_improve': 0, 'best_auroc_ood':0., 'epochs_no_improve_auroc':0, 'best_acc_head': 0.,'best_acc_tail': 0., 'best_fpr_ood':1.,'best_acc_head_fpr':-2,'epochs_no_improve_fpr':0,'epochs_no_improve_head':0, 'epochs_no_improve_head_fpr':0.}
        # best_model_wts_auroc, best_model_wts_balacc, best_model_wts_balacc_head, best_model_wts_balacc_head_fpr, best_model_wts_fpr = None, None, None, None, None
        best_model_wts_balacc_head_fpr = None
        while epoch <= args.max_epochs and (args.no_es or (early_stopping_dict['epochs_no_improve_auroc' if False else 'epochs_no_improve'] <= args.patience)):
            if args.use_hol_new:
                history = train_hol_new(model=model, device=device, loss_fxn=loss_fxn, optimizer=optimizer, data_loader=train_loader, history=history, epoch=epoch, model_dir=model_dir, classes=train_dataset.CLASSES, mixup=args.mixup, mixup_alpha=args.mixup_alpha, ood_loader=ood_loader, id_classes=id_classes_train, ood_classes=ood_classes_train, lambda_hol=args.lambda_hol, lambda_hol_syn=args.lambda_hol_syn)
            else:
                history = train(model=model, device=device, loss_fxn=loss_fxn, optimizer=optimizer, data_loader=train_loader, history=history, epoch=epoch, model_dir=model_dir, classes=train_dataset.CLASSES, mixup=args.mixup, mixup_alpha=args.mixup_alpha, ood_loader=ood_loader, id_classes=id_classes_train, ood_classes=ood_classes_train, dummy_oe=False)
            # validation
            history, early_stopping_dict, best_model_wts_balacc_head_fpr= validate(model=model, device=device, loss_fxn=loss_fxn, optimizer=optimizer, data_loader=val_loader, history=history, epoch=epoch, model_dir=model_dir, early_stopping_dict=early_stopping_dict, best_model_wts_balacc_head_fpr=best_model_wts_balacc_head_fpr, classes=val_dataset.CLASSES, id_classes=id_classes, ood_classes=ood_classes, stop_auroc=False, dreamood=False, convert_head=False)
            # history, early_stopping_dict, best_model_wts_balacc_head_fpr= validate(model=model, device=device, loss_fxn=loss_fxn, optimizer=optimizer, data_loader=val_loader, history=history, epoch=epoch, model_dir=model_dir, early_stopping_dict=early_stopping_dict, best_model_wts_auroc=best_model_wts_auroc, best_model_wts_balacc=best_model_wts_balacc, best_model_wts_balacc_head=best_model_wts_balacc_head, best_model_wts_balacc_head_fpr=best_model_wts_balacc_head_fpr, best_model_wts_fpr=best_model_wts_fpr, classes=val_dataset.CLASSES, id_classes=id_classes, ood_classes=ood_classes, stop_auroc=False, dreamood=False, convert_head=False)
            # log test performance 
            history= validate(model=model, device=device, loss_fxn=loss_fxn, optimizer=optimizer, data_loader=bal_test_loader, history=history, epoch=epoch, model_dir=model_dir, early_stopping_dict=early_stopping_dict, best_model_wts_balacc_head_fpr=best_model_wts_balacc_head_fpr, classes=val_dataset.CLASSES, id_classes=id_classes, ood_classes=ood_classes, stop_auroc=False, only_log_test=True, dreamood=False, convert_head=False)
            # history= validate(model=model, device=device, loss_fxn=loss_fxn, optimizer=optimizer, data_loader=bal_test_loader, history=history, epoch=epoch, model_dir=model_dir, early_stopping_dict=early_stopping_dict, best_model_wts_auroc=best_model_wts_auroc, best_model_wts_balacc=best_model_wts_balacc, best_model_wts_balacc_head=best_model_wts_balacc_head, best_model_wts_balacc_head_fpr=best_model_wts_balacc_head_fpr, best_model_wts_fpr=best_model_wts_fpr, classes=val_dataset.CLASSES, id_classes=id_classes, ood_classes=ood_classes, stop_auroc=False, only_log_test=True, dreamood=False, convert_head=False)
            if args.drw and epoch == 10:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.1  # anneal LR
                loss_fxn = get_loss(args, weights, train_dataset)  # get class-weighted loss
                early_stopping_dict['epochs_no_improve'] = 0  # reset patience
                # potentially set args.use_hol, args.use_hol_new, args.use_oe = False and ood_loader=None 
                # such that only during pretraining OE+hol is used
            if scheduler is not None and not args.drw:
                scheduler.step()

            epoch += 1
    else:
        # best_model_wts_auroc= best_model_wts_balacc= best_model_wts_balacc_head= best_model_wts_balacc_head_fpr= best_model_wts_fpr = model.state_dict()
        best_model_wts_balacc_head_fpr= model.state_dict()
    # save final weights
    final_model_wts = model.state_dict()
    torch.save({'weights': final_model_wts, 'optimizer': optimizer.optimizer.state_dict() if optimizer.sam_opt else optimizer.state_dict()}, os.path.join(model_dir, f'final-wts-{epoch}.pt'))
    
    # if best_model_wts_auroc is not None:
    #     # auroc weights
    #     evaluate(model=model, device=device, loss_fxn=loss_fxn, dataset=bal_test_dataset, split='balanced-test', batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts_auroc, id_classes=id_classes, ood_classes=ood_classes,save_postfix='valSelectedAuroc', dreamood=False, convert_head=False)
    #     # Evaluate on imbalanced test set
    #     evaluate(model=model, device=device, loss_fxn=loss_fxn, dataset=test_dataset, split='test', batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts_auroc, id_classes=id_classes, ood_classes=ood_classes,save_postfix='valSelectedAuroc', dreamood=False, convert_head=False)
    #     # Evaluate on balanced val set
    #     evaluate(model=model, device=device, loss_fxn=loss_fxn, dataset=val_dataset, split='val-10-100-noaug', batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts_auroc, id_classes=id_classes, ood_classes=ood_classes,save_postfix='valSelectedAuroc', dreamood=False, convert_head=False)
    
    # if best_model_wts_fpr is not None:
    #     # fpr weights
    #     evaluate(model=model, device=device, loss_fxn=loss_fxn, dataset=bal_test_dataset, split='balanced-test', batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts_fpr, id_classes=id_classes, ood_classes=ood_classes,save_postfix='valSelectedfpr', dreamood=False, convert_head=False)
    #     # Evaluate on imbalanced test set
    #     evaluate(model=model, device=device, loss_fxn=loss_fxn, dataset=test_dataset, split='test', batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts_fpr, id_classes=id_classes, ood_classes=ood_classes,save_postfix='valSelectedfpr', dreamood=False, convert_head=False)
    #     # Evaluate on balanced val set
    #     evaluate(model=model, device=device, loss_fxn=loss_fxn, dataset=val_dataset, split='val-10-100-noaug', batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts_fpr, id_classes=id_classes, ood_classes=ood_classes,save_postfix='valSelectedfpr', dreamood=False, convert_head=False)
    
    # if best_model_wts_balacc is not None:
    #     # balacc weights
    #     evaluate(model=model, device=device, loss_fxn=loss_fxn, dataset=bal_test_dataset, split='balanced-test', batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts_balacc, id_classes=id_classes, ood_classes=ood_classes,save_postfix='valSelectedBalacc', dreamood=False, convert_head=False)
    #     # Evaluate on imbalanced test set
    #     evaluate(model=model, device=device, loss_fxn=loss_fxn, dataset=test_dataset, split='test', batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts_balacc, id_classes=id_classes, ood_classes=ood_classes,save_postfix='valSelectedBalacc', dreamood=False, convert_head=False)
    #     # Evaluate on balanced val set
    #     evaluate(model=model, device=device, loss_fxn=loss_fxn, dataset=val_dataset, split='val-10-100-noaug', batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts_balacc, id_classes=id_classes, ood_classes=ood_classes,save_postfix='valSelectedBalacc', dreamood=False, convert_head=False)
    
    # if best_model_wts_balacc_head is not None:
    #     # balacc weights head
    #     evaluate(model=model, device=device, loss_fxn=loss_fxn, dataset=bal_test_dataset, split='balanced-test', batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts_balacc_head, id_classes=id_classes, ood_classes=ood_classes,save_postfix='valSelectedBalaccHead', dreamood=False, convert_head=False)
    #     # Evaluate on imbalanced test set
    #     evaluate(model=model, device=device, loss_fxn=loss_fxn, dataset=test_dataset, split='test', batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts_balacc_head, id_classes=id_classes, ood_classes=ood_classes,save_postfix='valSelectedBalaccHead', dreamood=False, convert_head=False)
    #     # Evaluate on balanced val set
    #     evaluate(model=model, device=device, loss_fxn=loss_fxn, dataset=val_dataset, split='val-10-100-noaug', batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts_balacc_head, id_classes=id_classes, ood_classes=ood_classes,save_postfix='valSelectedBalaccHead', dreamood=False, convert_head=False)
    
    if best_model_wts_balacc_head_fpr is not None:
        # balacc weights tail
        evaluate(model=model, device=device, loss_fxn=loss_fxn, dataset=bal_test_dataset, split='balanced-test', batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts_balacc_head_fpr, id_classes=id_classes, ood_classes=ood_classes,save_postfix='valSelectedBalaccheadFPR', dreamood=False, convert_head=False)
        # Evaluate on imbalanced test set
        evaluate(model=model, device=device, loss_fxn=loss_fxn, dataset=test_dataset, split='test', batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts_balacc_head_fpr, id_classes=id_classes, ood_classes=ood_classes,save_postfix='valSelectedBalaccheadFPR', dreamood=False, convert_head=False)
        # Evaluate on balanced val set
        evaluate(model=model, device=device, loss_fxn=loss_fxn, dataset=val_dataset, split='val-10-100-noaug', batch_size=args.batch_size, history=history, model_dir=model_dir, weights=best_model_wts_balacc_head_fpr, id_classes=id_classes, ood_classes=ood_classes,save_postfix='valSelectedBalaccheadFPR', dreamood=False, convert_head=False)


if __name__ == '__main__':
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/ssd1/greg/NIH_CXR/images', type=str)
    parser.add_argument('--label_dir', default='labels/', type=str)
    parser.add_argument('--out_dir', default='results/', type=str, help="path to directory where results and model weights will be saved")
    parser.add_argument('--dataset', required=True, type=str, )
                        # choices=['nih-cxr-lt','nih-cxr-lt-head-med','nih-cxr-lt-tail', 'mimic-cxr-lt', 'nih-cxr-lt-tail-syn','nih-cxr-lt-head','syn_roentgen_CXR_Dataset','syn_roentgen_CXR_Dataset_tail','NIH_CXR_Dataset_tail_real_and_syn','nih-cxr-lt-tail-syn-bal-short', 'NIH_CXR_Dataset_tail_real_and_syn_bal_short', "mimic-cxr-lt-head-med", "MIMIC_CXR_Dataset_tail_real_and_syn","NIH_CXR_Dataset_tail_real_and_syn_targeted_maxconf", "NIH_CXR_Dataset_tail_real_and_syn_targeted_maxconf_extended", "Skin_dataset", "Skin_dataset_lt", "Skin_dataset_very_lt", "isic2019", "Skin_dataset_very_very_lt", "isic2019_lt", "MIMIC_CXR_Dataset_tail_real_and_syn_targeted_maxconf_extended", "Skin_dataset_tumor","Skin_dataset_similar", "Skin_dataset_similar_PIL", "Skin_dataset_PIL", "Skin_dataset_similar_PIL_real_and_syn_targeted_maxconf", "Skin_dataset_similar_PIL_real_and_syn_targeted_maxconf_extended", "Skin_dataset_similar_PIL_full_real_and_syn_targeted_maxconf"])
    parser.add_argument('--loss', default='ce', type=str, choices=['ce', 'focal', 'ldam'])
    parser.add_argument('--drw', action='store_true', default=False)
    parser.add_argument('--rw_method', default='', choices=['', 'sklearn', 'cb'])
    parser.add_argument('--cb_beta', default=0.9999, type=float)
    parser.add_argument('--fl_gamma', default=2., type=float)
    parser.add_argument('--mixup', action='store_true', default=False)
    parser.add_argument('--mixup_alpha', default=0.2, type=float)
    parser.add_argument('--decoupling_method', default='', choices=['', 'cRT', 'tau_norm'], type=str)
    parser.add_argument('--decoupling_weights', type=str)
    parser.add_argument('--model_name', default='resnet50', type=str, help="CNN backbone to use")
    parser.add_argument('--max_epochs', default=60, type=int, help="maximum number of epochs to train")
    parser.add_argument('--batch_size', default=256, type=int, help="batch size for training, validation, and testing (will be lowered if TTA used)")
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--patience', default=15, type=int, help="early stopping 'patience' during training")
    parser.add_argument('--no_es', action='store_true', default=False, help="no early stopping")
    parser.add_argument('--rand_init', action='store_true', default=False)
    parser.add_argument('--n_TTA', default=0, type=int, help="number of augmented copies to use during test-time augmentation (TTA), default 0")
    parser.add_argument('--seed', default=0, type=int, help="set random seed")
# lr decay
    parser.add_argument('--lrScheduler', default=None, choices=['cosine'])
    parser.add_argument('--minLR', default=0, type=float)
    parser.add_argument('--optim', default='adam', choices=['adam', 'adamw'])
# for hol
    parser.add_argument('--use_hol_new', action='store_true', default=False)
    parser.add_argument('--lambda_hol', default=0.1, type=float)
    parser.add_argument('--lambda_hol_syn', default=0.1, type=float)

    args = parser.parse_args()

    print(args)

    main(args)
