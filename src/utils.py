import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm

from copy import deepcopy

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, classification_report, confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import LabelBinarizer


def auroc_ood(values_in: np.ndarray, values_out: np.ndarray) -> float:
    if len(values_in) * len(values_out) == 0:
        return np.NAN
    y_true = len(values_in) * [1] + len(values_out) * [0]
    y_score = np.nan_to_num(np.concatenate([values_in, values_out]).flatten())
    return roc_auc_score(y_true, y_score)

def fpr_at_tpr(values_in: np.ndarray, values_out: np.ndarray, tpr: float) -> float:
    if len(values_in) * len(values_out) == 0:
        return np.NAN
    t = np.quantile(values_in, (1 - tpr))
    fpr = (values_out >= t).mean()
    return fpr

def evaluate_hol(softmax_id_val, softmax_ood, tail_idxs):
    score_id = -softmax_id_val[:,tail_idxs].sum(axis=-1)
    score_ood = -softmax_ood[:,tail_idxs].sum(axis=-1)
    return score_id, score_ood

def set_seed(seed):
    """Set all random seeds and settings for reproducibility (deterministic behavior)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def val_worker_init_fn(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(model, device, loss_fxn, optimizer, data_loader, history, epoch, model_dir, classes, mixup, mixup_alpha, ood_loader, id_classes, ood_classes, dummy_oe):
    """Train PyTorch model for one epoch on NIH ChestXRay14 dataset.
    Parameters
    ----------
        model : PyTorch model
        device : PyTorch device
        loss_fxn : PyTorch loss function
        optimizer : PyTorch optimizer
        data_loader : PyTorch data loader
        history : pandas DataFrame
            Data frame containing history of training metrics
        epoch : int
            Current epoch number (1-K)
        model_dir : str
            Path to output directory where metrics, model weights, etc. will be stored
        classes : list[str]
            Ordered list of names of output classes
    Returns
    -------
        history : pandas DataFrame
            Updated history data frame with metrics from completed training epoch
    """
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch}')

    running_loss = 0.
    running_loss_extra = 0.
    y_true, y_hat = [], []
    losses=torch.tensor([]).to(device)
    for i, (x, y) in pbar:
        x = x.to(device)
        y = y.to(device)

        if mixup:
            x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha, True)

        out = model(x)

        if mixup:
            loss = mixup_criterion(loss_fxn, out[0:len(y)], y_a, y_b, lam)
        else:
            loss = loss_fxn(out[0:len(y)], y)
            losses = torch.cat([losses,loss.detach()])
            loss=loss.mean()

        for param in model.parameters():
            param.grad = None
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        y_hat.append(out[0:len(y)].softmax(dim=1).detach().cpu().numpy())
        y_true.append(y.detach().cpu().numpy())

        pbar.set_postfix({'loss': running_loss / (i + 1)})

    # Collect true and predicted labels into flat numpy arrays
    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)

    tail_bool = [i in ood_classes for i in y_true]
    head_med_bool = [i in id_classes for i in y_true]
    score_id_tail, score_ood_tail = evaluate_hol(y_hat[head_med_bool],y_hat[tail_bool], ood_classes)
    auroc = auroc_ood(score_id_tail,score_ood_tail)
    fpr = fpr_at_tpr(score_id_tail, score_ood_tail, 0.95)

    # Compute metrics
    auc = roc_auc_score(y_true, y_hat, average='macro', multi_class='ovr')
    b_acc = balanced_accuracy_score(y_true, y_hat.argmax(axis=1))
    mcc = matthews_corrcoef(y_true, y_hat.argmax(axis=1))
    b_acc_head = balanced_accuracy_score(y_true[head_med_bool], y_hat[head_med_bool,:].argmax(axis=1))
    b_acc_tail = balanced_accuracy_score(y_true[tail_bool], y_hat[tail_bool,:].argmax(axis=1))

    loss_head = (torch.sum(losses[head_med_bool])/(len(losses))).item()
    loss_tail = (torch.sum(losses[tail_bool])/len(losses)).item()

    print('Balanced Accuracy:', round(b_acc, 3), '|', 'MCC:', round(mcc, 3), '|', 'AUC:', round(auc, 3), '|', 'AUC-OOD:',round(auroc,3), '|', 'FPR-OOD:', round(fpr,3), '|','Balanced Accuracy head:', round(b_acc_head, 3), '|','Balanced Accuracy tail:', round(b_acc_tail, 3))

    current_metrics = pd.DataFrame([[epoch, 'train', running_loss / (i + 1), b_acc, mcc, auc,auroc,fpr, b_acc_head, b_acc_tail, loss_head, loss_tail, running_loss_extra / (i + 1)]], columns=history.columns)
    current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    return history.append(current_metrics)

def train_hol_new(model, device, loss_fxn, optimizer, data_loader, history, epoch, model_dir, classes, mixup, mixup_alpha, ood_loader, ood_classes, id_classes, lambda_hol=0.5,lambda_hol_syn=0.5):
    """Train PyTorch model for one epoch on NIH ChestXRay14 dataset.
    Parameters
    ----------
        model : PyTorch model
        device : PyTorch device
        loss_fxn : PyTorch loss function
        optimizer : PyTorch optimizer
        data_loader : PyTorch data loader
        history : pandas DataFrame
            Data frame containing history of training metrics
        epoch : int
            Current epoch number (1-K)
        model_dir : str
            Path to output directory where metrics, model weights, etc. will be stored
        classes : list[str]
            Ordered list of names of output classes
    Returns
    -------
        history : pandas DataFrame
            Updated history data frame with metrics from completed training epoch
    """
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch}')

    running_loss = 0.
    running_loss_extra = 0.
    losses = torch.tensor([]).to(device)
    y_true, y_hat = [], []
    for i, (x, y) in pbar:
        x = x.to(device)
        y = y.to(device)

        if mixup:
            raise NotImplementedError("Mixup canot be combined with hol loss")

        out = model(x)
        loss = loss_fxn(out[0:len(y)], y)
        losses = torch.cat([losses,loss.detach()])
        loss=loss.mean()

        # add L-coarse
        out = torch.softmax(out,-1)
        if lambda_hol>0.:
            samples_ood_bool = y.cpu().apply_(lambda x: x in ood_classes).bool().to(device)
            loss_coarse = (-out[0:len(y)][samples_ood_bool][:,ood_classes].sum(-1).log().sum() - out[0:len(y)][~samples_ood_bool][:,id_classes].sum(-1).log().sum()) / len(y)
            loss+=(lambda_hol*loss_coarse)
        else:
            loss_coarse=torch.tensor(0)

        for param in model.parameters():
            param.grad = None
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss_extra = (lambda_hol*loss_coarse).item()
        y_hat.append(out[0:len(y)].softmax(dim=1).detach().cpu().numpy())
        y_true.append(y.detach().cpu().numpy())

        pbar.set_postfix({'loss': running_loss / (i + 1)})

    # Collect true and predicted labels into flat numpy arrays
    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)

    tail_bool = [i in ood_classes for i in y_true]
    head_med_bool = [i in id_classes for i in y_true]
    score_id_tail, score_ood_tail = evaluate_hol(y_hat[head_med_bool],y_hat[tail_bool], ood_classes)
    auroc = auroc_ood(score_id_tail,score_ood_tail)
    fpr = fpr_at_tpr(score_id_tail, score_ood_tail, 0.95)

    # Compute metrics
    auc = roc_auc_score(y_true, y_hat, average='macro', multi_class='ovr')
    b_acc = balanced_accuracy_score(y_true, y_hat.argmax(axis=1))
    mcc = matthews_corrcoef(y_true, y_hat.argmax(axis=1))
    b_acc_head = balanced_accuracy_score(y_true[head_med_bool], y_hat[head_med_bool,:].argmax(axis=1))
    b_acc_tail = balanced_accuracy_score(y_true[tail_bool], y_hat[tail_bool,:].argmax(axis=1))

    loss_head = (torch.sum(losses[head_med_bool])/(len(losses))).item()
    loss_tail = (torch.sum(losses[tail_bool])/len(losses)).item()

    print('Balanced Accuracy:', round(b_acc, 3), '|', 'MCC:', round(mcc, 3), '|', 'AUC:', round(auc, 3), '|', 'AUC-OOD:',round(auroc,3), '|', 'FPR-OOD:', round(fpr,3), '|','Balanced Accuracy head:', round(b_acc_head, 3), '|','Balanced Accuracy tail:', round(b_acc_tail, 3) )

    current_metrics = pd.DataFrame([[epoch, 'train', running_loss / (i + 1), b_acc, mcc, auc,auroc,fpr, b_acc_head, b_acc_tail, loss_head, loss_tail, running_loss_extra / (i + 1)]], columns=history.columns)
    current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    return history.append(current_metrics)

# def validate(model, device, loss_fxn, optimizer, data_loader, history, epoch, model_dir, early_stopping_dict, best_model_wts_auroc, best_model_wts_balacc, best_model_wts_balacc_head, best_model_wts_balacc_head_fpr, best_model_wts_fpr, classes, id_classes, ood_classes,stop_auroc=False, only_log_test=False, convert_head=False, dreamood=False):
def validate(model, device, loss_fxn, optimizer, data_loader, history, epoch, model_dir, early_stopping_dict,  best_model_wts_balacc_head_fpr, classes, id_classes, ood_classes,stop_auroc=False, only_log_test=False, convert_head=False, dreamood=False):
    """Evaluate PyTorch model on validation set of NIH ChestXRay14 dataset.
    Parameters
    ----------
        model : PyTorch model
        device : PyTorch device
        loss_fxn : PyTorch loss function
        ls : int
            Ratio of label smoothing to apply during loss computation
        optimizer : PyTorch optimizer
        data_loader : PyTorch data loader
        history : pandas DataFrame
            Data frame containing history of training metrics
        epoch : int
            Current epoch number (1-K)
        model_dir : str
            Path to output directory where metrics, model weights, etc. will be stored
        early_stopping_dict : dict
            Dictionary of form {'epochs_no_improve': <int>, 'best_loss': <float>} for early stopping
        best_model_wts : PyTorch state_dict
            Model weights from best epoch
        classes : list[str]
            Ordered list of names of output classes
        fusion : bool
            Whether or not fusion is being performed (image + metadata inputs)
        meta_only : bool
            Whether or not to train on *only* metadata as input
    Returns
    -------
        history : pandas DataFrame
            Updated history data frame with metrics from completed training epoch
        early_stopping_dict : dict
            Updated early stopping metrics
        best_model_wts : PyTorch state_dict
            (Potentially) updated model weights (if best validation loss achieved)
    """
    model.eval()
    
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[VAL] Epoch {epoch}')

    running_loss = 0.
    running_loss_extra = 0.
    losses=torch.tensor([]).to(device)
    y_true, y_hat, Ecs = [], [], []
    with torch.no_grad():
        for i, (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)

            out = model(x)

            if not dreamood:
                loss = loss_fxn(out, y)
                losses = torch.cat([losses,loss])
                running_loss += loss.mean().item()

            y_hat.append(out.softmax(dim=1).detach().cpu().numpy())
            y_true.append(y.detach().cpu().numpy())
            if dreamood:
                Ec = torch.logsumexp(out,dim=1)
                Ecs.append(Ec)
            pbar.set_postfix({'loss': running_loss / (i + 1)})

    # Collect true and predicted labels into flat numpy arrays
    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)
    preds = y_hat.argmax(axis=1)

    if convert_head: # convert only-head predictions with few classes to correct indices with more classes
        preds = np.array([id_classes[pred] for pred in preds])

    tail_bool = [i in ood_classes for i in y_true]
    head_med_bool = [i in id_classes for i in y_true]

    if dreamood: # use logistic score as ood score
        with torch.no_grad():
            output1 = model.logistic_regression(torch.cat(Ecs).reshape(-1, 1))
            score_id_tail = output1.softmax(-1)[head_med_bool,1].detach().cpu().numpy()
            score_ood_tail = output1.softmax(-1)[tail_bool,1].detach().cpu().numpy()
    else: # use sum of tail probs
        score_id_tail, score_ood_tail = evaluate_hol(y_hat[head_med_bool],y_hat[tail_bool], ood_classes)

    auroc = auroc_ood(score_id_tail,score_ood_tail)
    fpr = fpr_at_tpr(score_id_tail, score_ood_tail, 0.95)

    loss_head = (torch.sum(losses[head_med_bool])/(len(losses))).item() if not dreamood else 0
    loss_tail = (torch.sum(losses[tail_bool])/len(losses)).item() if not dreamood else 0

    print('Loss mean: ', torch.mean(losses).item(), running_loss / (i+1))
    
    # Compute metrics
    auc = -99#roc_auc_score(y_true, y_hat, average='macro', multi_class='ovr')
    b_acc = balanced_accuracy_score(y_true, preds)
    mcc = matthews_corrcoef(y_true, preds)
    b_acc_head = balanced_accuracy_score(y_true[head_med_bool], preds[head_med_bool])
    b_acc_tail = balanced_accuracy_score(y_true[tail_bool], preds[tail_bool])


    if only_log_test:
        current_metrics = pd.DataFrame([[epoch, 'test', running_loss / (i + 1), b_acc, mcc, auc,auroc,fpr, b_acc_head, b_acc_tail, loss_head, loss_tail, running_loss_extra / (i + 1) ]], columns=history.columns)
        current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)
        return history.append(current_metrics)

    print('[VAL] Balanced Accuracy:', round(b_acc, 3), '|', 'MCC:', round(mcc, 3), '|', 'AUC:', round(auc, 3), '|', 'AUC-OOD:',round(auroc,3), '|', 'FPR-OOD:', round(fpr,3), '|','Balanced Accuracy head:', round(b_acc_head, 3), '|','Balanced Accuracy tail:', round(b_acc_tail, 3) )

    current_metrics = pd.DataFrame([[epoch, 'val', running_loss / (i + 1), b_acc, mcc, auc,auroc,fpr, b_acc_head, b_acc_tail, loss_head, loss_tail, running_loss_extra / (i + 1)]], columns=history.columns)
    current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    if b_acc_head-fpr > early_stopping_dict['best_acc_head_fpr']:
        print(f'--- EARLY STOPPING: Accuracy-tail has improved from {round(early_stopping_dict["best_acc_head_fpr"], 3)} to {round(b_acc_head-fpr, 3)}!')
        early_stopping_dict['epochs_no_improve_head_fpr'] = 0
        early_stopping_dict['best_acc_head_fpr'] = b_acc_head-fpr
        best_model_wts_balacc_head_fpr = deepcopy(model.state_dict())
        torch.save({'weights': best_model_wts_balacc_head_fpr, 'optimizer': optimizer.optimizer.state_dict() if optimizer.sam_opt else optimizer.state_dict()}, os.path.join(model_dir, f'chkpt_epoch-{epoch}-balacc-head-fpr.pt'))
    else:
        print(f'--- EARLY STOPPING: Accuracy-tail has not improved from {round(early_stopping_dict["best_acc_head_fpr"], 3)} ---')
        early_stopping_dict['epochs_no_improve_head_fpr'] += 1


    

    return history.append(current_metrics), early_stopping_dict, best_model_wts_balacc_head_fpr #best_model_wts_auroc, best_model_wts_balacc, best_model_wts_balacc_head, best_model_wts_balacc_head_fpr, best_model_wts_fpr


def evaluate(model, device, loss_fxn, dataset, split, batch_size, history, model_dir, weights, id_classes, ood_classes, save_postfix="", convert_head=False, dreamood=False):
    """Evaluate PyTorch model on test set of NIH ChestXRay14 dataset. Saves training history csv, summary text file, training curves, etc.
    Parameters
    ----------
        model : PyTorch model
        device : PyTorch device
        loss_fxn : PyTorch loss function
        ls : int
            Ratio of label smoothing to apply during loss computation
        batch_size : int
        history : pandas DataFrame
            Data frame containing history of training metrics
        model_dir : str
            Path to output directory where metrics, model weights, etc. will be stored
        weights : PyTorch state_dict
            Model weights from best epoch
        n_TTA : int
            Number of augmented copies to use for test-time augmentation (0-K)
        fusion : bool
            Whether or not fusion is being performed (image + metadata inputs)
        meta_only : bool
            Whether or not to train on *only* metadata as input
    """
    model.load_state_dict(weights)  # load best weights
    model.eval()

    ## INFERENCE
    data_loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8 if split == 'test' else 2, pin_memory=True, worker_init_fn=val_worker_init_fn)

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[{split.upper()}] EVALUATION')

    running_loss = 0.
    y_true, y_hat, Ecs = [], [], []
    with torch.no_grad():
        for i, (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            if not dreamood:
                loss = loss_fxn(out, y)
                running_loss += loss.mean().item()

            y_hat.append(out.softmax(dim=1).detach().cpu().numpy())
            y_true.append(y.detach().cpu().numpy())
            if dreamood:
                Ec = torch.logsumexp(out,dim=1)
                Ecs.append(Ec)
            pbar.set_postfix({'loss': running_loss / (i + 1)})

    # Collect true and predicted labels into flat numpy arrays
    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)
    preds = y_hat.argmax(axis=1)

    if convert_head: # convert only-head predictions with few classes to correct indices with more classes
        preds = np.array([id_classes[pred] for pred in preds])

    tail_bool = [i in ood_classes for i in y_true]
    head_med_bool = [i in id_classes for i in y_true]
    if dreamood: # use logistic score as ood score
        with torch.no_grad():
            output1 = model.logistic_regression(torch.cat(Ecs).reshape(-1, 1))
            score_id_tail = output1.softmax(-1)[head_med_bool,1].detach().cpu().numpy()
            score_ood_tail = output1.softmax(-1)[tail_bool,1].detach().cpu().numpy()
    else: # use sum of tail probs
        score_id_tail, score_ood_tail = evaluate_hol(y_hat[head_med_bool],y_hat[tail_bool], ood_classes)    

    auroc = auroc_ood(score_id_tail,score_ood_tail)
    fpr = fpr_at_tpr(score_id_tail, score_ood_tail, 0.95)

    # Compute metrics
    auc = -99#roc_auc_score(y_true, y_hat, average='macro', multi_class='ovr')
    b_acc = balanced_accuracy_score(y_true, y_hat.argmax(axis=1))
    conf_mat = confusion_matrix(y_true, y_hat.argmax(axis=1),labels=[i for i in range(len(dataset.CLASSES))])
    accuracies = conf_mat.diagonal() / conf_mat.sum(axis=1)
    mcc = matthews_corrcoef(y_true, y_hat.argmax(axis=1))
    b_acc_head = balanced_accuracy_score(y_true[head_med_bool], y_hat[head_med_bool,:].argmax(axis=1))
    b_acc_tail = balanced_accuracy_score(y_true[tail_bool], y_hat[tail_bool,:].argmax(axis=1))
    cls_report = classification_report(y_true, y_hat.argmax(axis=1), target_names=dataset.CLASSES, labels = [i for i in range(len(dataset.CLASSES))], digits=3)

    print(f'[{split.upper()}] Balanced Accuracy: {round(b_acc, 3)} | MCC: {round(mcc, 3)} | AUC: {round(auc, 3)}', '|', 'AUC-OOD:',round(auroc,3), '|', 'FPR-OOD:', round(fpr,3), '|','Balanced Accuracy head:', round(b_acc_head, 3), '|','Balanced Accuracy tail:', round(b_acc_tail, 3))

    # Collect and save true and predicted disease labels for test set
    pred_df = pd.DataFrame(y_hat, columns=dataset.CLASSES)
    true_df = pd.DataFrame(LabelBinarizer().fit(range(len(dataset.CLASSES))).transform(y_true), columns=dataset.CLASSES)

    pred_df.to_csv(os.path.join(model_dir, f'{split}_pred{save_postfix}.csv'), index=False)
    true_df.to_csv(os.path.join(model_dir, f'{split}_true{save_postfix}.csv'), index=False)

    # Plot confusion matrix
    fig, ax = plot_confusion_matrix(conf_mat, figsize=(24, 24), colorbar=True, show_absolute=True, show_normed=True, class_names=dataset.CLASSES)
    fig.savefig(os.path.join(model_dir, f'{split}_cm{save_postfix}.png'), dpi=300, bbox_inches='tight')

    # Configure the figure and subplots
    fig, axs = plt.subplots(4, 1, figsize=(20, 24))  # 4 rows, 1 column

    # Overall Loss with Secondary y-axis for Train
    ax1_twin = axs[0].twinx()
    axs[0].plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'loss'], label='Val', color='blue')
    axs[0].plot(history.loc[history['phase'] == 'test', 'epoch'], history.loc[history['phase'] == 'test', 'loss'], label='Test', color='green')
    ax1_twin.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'loss'], label='Train', color='red', linestyle='--')
    axs[0].set_title('Overall Loss')
    axs[0].set_ylabel('Val/Test Loss')
    ax1_twin.set_ylabel('Train Loss', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    axs[0].grid(True)
    # Handling legends for dual-axis
    lines, labels = axs[0].get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    axs[0].legend(lines + lines2, labels + labels2)
    # Head Loss with Secondary y-axis for Train
    ax2_twin = axs[1].twinx()
    axs[1].plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'loss_head'], label='Val Head', color='blue')
    axs[1].plot(history.loc[history['phase'] == 'test', 'epoch'], history.loc[history['phase'] == 'test', 'loss_head'], label='Test Head', color='green')
    ax2_twin.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'loss_head'], label='Train Head', color='red', linestyle='--')
    axs[1].set_title('Head Loss')
    axs[1].set_ylabel('Val/Test Head Loss')
    ax2_twin.set_ylabel('Train Head Loss', color='red')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    axs[1].grid(True)
    # Handling legends for dual-axis
    lines, labels = axs[1].get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    axs[1].legend(lines + lines2, labels + labels2)
    # Tail Loss with Secondary y-axis for Train
    ax3_twin = axs[2].twinx()
    axs[2].plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'loss_tail'], label='Val Tail', color='blue')
    axs[2].plot(history.loc[history['phase'] == 'test', 'epoch'], history.loc[history['phase'] == 'test', 'loss_tail'], label='Test Tail', color='green')
    ax3_twin.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'loss_tail'], label='Train Tail', color='red', linestyle='--')
    axs[2].set_title('Tail Loss')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Val/Test Tail Loss')
    ax3_twin.set_ylabel('Train Tail Loss', color='red')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    axs[2].grid(True)
    # Handling legends for dual-axis
    lines, labels = axs[2].get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    axs[2].legend(lines + lines2, labels + labels2)
    # Extra Loss
    axs[3].plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'loss_extra'], label='Train Extra', color='red')
    axs[3].set_title('Extra Loss')
    axs[3].set_xlabel('Epoch')
    axs[3].set_ylabel('Loss')
    axs[3].grid(True)
    axs[3].legend()
    # Adjust the subplot layout for a clear look
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(model_dir, f'loss.png'), dpi=300, bbox_inches='tight')

    # Plot accuracy curves
    fig, axs = plt.subplots(2, 1, figsize=(15, 12))  # 2 rows for separate metrics
    # First row: Balanced Accuracy
    ax = axs[0]
    # Train
    ax.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'balanced_acc'],
            label='Train Total', color='blue', linestyle='--')
    ax.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'balanced_acc_head'],
            label='Train Head', color='red', linestyle='--')
    ax.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'balanced_acc_tail'],
            label='Train Tail', color='green', linestyle='--')
    # Validation
    ax.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'balanced_acc'],
            label='Val Total', color='blue')
    ax.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'balanced_acc_head'],
            label='Val Head', color='red')
    ax.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'balanced_acc_tail'],
            label='Val Tail', color='green')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.grid(True)
    ax.legend()
    # Second row: FPR and AUROC
    ax1 = axs[1]
    # fpr_ood on primary y-axis
    ax1.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'fpr_ood'],
            label='Train FPR', color='blue', linestyle='--')
    ax1.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'fpr_ood'],
            label='Val FPR', color='blue')
    ax1.plot(history.loc[history['phase'] == 'test', 'epoch'], history.loc[history['phase'] == 'test', 'fpr_ood'],
            label='Test FPR', color='blue', linestyle=':')
    ax1.set_ylabel('FPR ood hol', fontsize=12,color='blue')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.grid(True)
    # auroc_ood on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'auroc_ood'],
            label='Train AUROC', color='green', linestyle='--')
    ax2.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'auroc_ood'],
            label='Val AUROC', color='green')
    ax2.plot(history.loc[history['phase'] == 'test', 'epoch'], history.loc[history['phase'] == 'test', 'auroc_ood'],
            label='Test AUROC', color='green', linestyle=':')
    ax2.set_ylabel('AUROC ood hol', fontsize=12,color='green')
    # Creating a combined legend for dual-axis
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(model_dir, f'balanced_acc.png'), dpi=300, bbox_inches='tight')
    
    
    # Create summary text file describing final performance
    summary = f'Balanced Accuracy: {round(b_acc, 3)}\n'
    summary += f'Matthews Correlation Coefficient: {round(mcc, 3)}\n'
    summary += f'Mean AUC: {round(auc, 3)}\n\n'
    summary += f'AUC-OOD-hol: {round(auroc, 3)}\n'
    summary += f'FPR-OOD-hol: {round(fpr, 3)}\n\n'
    summary += f'Balanced Accuracy head: {round(b_acc_head, 3)}\n'
    summary += f'Balanced Accuracy tail: {round(b_acc_tail, 3)}\n\n'

    summary += 'Class:| Accuracy\n'
    for i, c in enumerate(dataset.CLASSES):
        summary += f'{c}:| {round(accuracies[i], 3)}\n'
    summary += '\n'
    
    summary += cls_report

    f = open(os.path.join(model_dir, f'{split}_summary{save_postfix}.txt'), 'w')
    f.write(summary)
    f.close()

