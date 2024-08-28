import torch
import torch.nn.functional as F
from lpips import LPIPS

def get_loss_function(loss, classifier, classifier2=None):
    if classifier2 is None:
        if loss == 'log_conf':
            def loss_function(image, targets, augment=True):
                class_out = classifier(image, augment=augment)
                target_ = torch.zeros(class_out.shape[0], dtype=torch.long, device=image.device).fill_(targets)
                loss = F.cross_entropy(class_out, target_, reduction='mean')
                return loss
        elif loss == 'conf':
            def loss_function(image, targets, augment=True):
                # print(f'Targets: {targets}')
                class_out = classifier(image, augment=augment)
                # print(f'class-out shape: {class_out.shape}')
                probs = torch.softmax(class_out, dim=1)
                # print(f'probs shape: {class_out.shape}')
                log_conf = probs[:, targets]
                loss = -log_conf.mean()
                return loss
        elif loss == 'neg_conf':
            def loss_function(image, targets, augment=True):
                # print(f'Targets: {targets}')
                class_out = classifier(image, augment=augment)
                # print(f'class-out shape: {class_out.shape}')
                probs = torch.softmax(class_out, dim=1)
                # print(f'probs shape: {class_out.shape}')
                log_conf = probs[:, targets]
                loss = log_conf.mean()
                return loss
        elif loss == 'neg_log_conf':
            def loss_function(image, targets, augment=True):
                class_out = classifier(image, augment=augment)
                target_ = torch.zeros(class_out.shape[0], dtype=torch.long, device=image.device).fill_(targets)
                loss = F.cross_entropy(class_out, target_, reduction='mean')
                return -loss        
        elif loss == 'neg_entropy':
            def loss_function(image, targets, augment=True):
                class_out = classifier(image, augment=augment)
                targets = torch.full_like(class_out, fill_value=1.0 / class_out.shape[1]) # uniform
                log_probs = F.log_softmax(class_out, dim=1)
                cross_entropy = -torch.sum(targets * log_probs, dim=1)
                return cross_entropy.mean()
        elif loss == 'conf_boundary':
          def loss_function(image, targets, augment=True):
                assert len(targets)>1
                val=1/len(targets)
                class_out = classifier(image, augment=augment)
                probs = torch.softmax(class_out, dim=1)
                log_conf = probs[:, targets]
                border_probs = torch.tensor([[0.7,0.1]],device=image.device)#torch.full(log_conf.shape, val, device=image.device)
                print('Borderprobs: ',border_probs)
                # loss = (torch.norm(border_probs-log_conf))
                # loss = 1/(log_conf)
                loss = log_conf**(-0.5)
                print('loss wo red: ',loss)
                loss = loss.mean()
                print('loss after red: ', loss)
                # loss = -log_conf.mean()
                return loss
        else:
            raise NotImplementedError(f'{loss} is not implemented.')
    else:
        if loss == 'conf':
            def loss_function(image, targets, augment=True):
                class_out1 = classifier(image, augment=augment)
                probs1 = torch.softmax(class_out1, dim=1)
                conf1 = probs1[:, targets]

                class_out2 = classifier2(image, augment=augment)
                probs2 = torch.softmax(class_out2, dim=1)
                conf2 = probs2[:, targets]

                loss = -conf1.mean() + conf2.mean()
                return loss
        elif loss == 'log_conf':
            def loss_function(image, targets, augment=True):
                class_out1 = classifier(image, augment=augment)
                log_probs1 = torch.log_softmax(class_out1, dim=1)
                log_conf1 = log_probs1[:, targets]

                class_out2 = classifier2(image, augment=augment)
                log_probs2 = torch.log_softmax(class_out2, dim=1)
                log_conf2 = log_probs2[:, targets]

                loss = (-log_conf1 + log_conf2).mean()
                return loss
        elif loss == 'logits':
            def loss_function(image, targets, augment=True):

                class_out1 = classifier(image, augment=augment)

                target_mask = torch.zeros_like(class_out1)
                target_mask[:, targets] = -1e12

                target_logits1 = class_out1[:, targets]
                max_logits1 = torch.max(class_out1 - target_mask, dim=1)[0]

                class_out2 = classifier2(image, augment=augment)
                target_logits2 = class_out2[:, targets]
                max_logits2 = torch.max(class_out2 - target_mask, dim=1)[0]

                loss = - (target_logits1 - max_logits1).mean() + (target_logits2 - max_logits2).mean()
                return loss
        else:
            raise NotImplementedError()

    return loss_function

def get_feature_loss_function(loss, classifier, layer_activations, classifier2=None, layer_activations2=None, mean1=None, prec1=None, mean2=None, prec2=None):
    if loss == 'neuron_activation':
        def loss_function(image, target_class_target_neuron, augment=True):
            target_class, target_neuron = target_class_target_neuron
            if len(image.shape) == 3:
                image = image[None, :]
            bs = image.shape[0]
            _ = layer_activations(image, augment=augment)
            act = layer_activations.activations[0][0]
            neuron = act[:, target_neuron].mean()
            loss = -neuron

            return loss
    elif loss == 'neuron_activation_plus_log_confidence':
        def loss_function(image, target_class_target_neuron, augment=True):
            target_class, target_neuron = target_class_target_neuron
            if len(image.shape) == 3:
                image = image[None, :]
            bs = image.shape[0]
            _ = layer_activations(image, augment=augment)
            act = layer_activations.activations[0][0]
            
            neuron = act[:, target_neuron].mean()
            loss = 10 * -neuron

            class_out = classifier(image, augment=augment)
            log_probs = torch.log_softmax(class_out, dim=1)
            log_conf = log_probs[:, target_class]
            loss = loss - 0.5 * log_conf.mean()

            return loss
    elif loss == 'neg_neuron_activation_plus_log_confidence':
        def loss_function(image, target_class_target_neuron, augment=True):
            target_class, target_neuron = target_class_target_neuron
            if len(image.shape) == 3:
                image = image[None, :]
            bs = image.shape[0]
            _ = layer_activations(image, augment=augment)
            act = layer_activations.activations[0][0]
            
            neuron = act[:, target_neuron].mean()
            loss = 10 * neuron

            class_out = classifier(image, augment=augment)
            log_probs = torch.log_softmax(class_out, dim=1)
            log_conf = log_probs[:, target_class]
            loss = loss - 0.5 * log_conf.mean()

            return loss
    elif loss == 'activation_target':
        def loss_function(image, target_activation, augment=True):
            if len(image.shape) == 3:
                image = image[None, :]
            act = layer_activations.attribute(image, attribute_to_layer_input=True)
            bs = act.shape[0]
            neurons_spatial_avg = act.reshape((bs, act.shape[1], -1)).mean(dim=2)
            loss = F.mse_loss(neurons_spatial_avg, target_activation[None, :].expand(bs, -1), reduction='mean')
            return loss
    elif loss =='diff_maha':
        assert classifier2 is not None and layer_activations2 is not None
        def loss_function(image,target_activation,augment=True):
            if len(image.shape) == 3:
                image = image[None, :]
            _ = layer_activations(image, augment=augment)
            act_1 = layer_activations.activations[0][0]
            score_ood_1 = - torch.stack([(((f - mean1) @ prec1) * (f - mean1)).sum(axis=-1).min() for f in act_1.double()]).sum()
            #(((act_1.double() - mean1) @ prec1) * (act_1.double() - mean1)).sum(axis=-1).min()
            _ = layer_activations2(image, augment=augment)
            act_2 = layer_activations2.activations[0][0]
            score_ood_2 = - torch.stack([(((f - mean2) @ prec2) * (f - mean2)).sum(axis=-1).min() for f in act_2.double()]).sum()
            #(((act_2.double() - mean2) @ prec2) * (act_2.double() - mean2)).sum(axis=-1).min()
            print(score_ood_1, score_ood_2)
            loss = score_ood_1 - score_ood_2
            return loss
    elif loss =='diff_maha_squared2':
        assert classifier2 is not None and layer_activations2 is not None
        def loss_function(image,target_activation,augment=True):
            if len(image.shape) == 3:
                image = image[None, :]
            _ = layer_activations(image, augment=augment)
            act_1 = layer_activations.activations[0][0]
            score_ood_1 = - torch.stack([(((f - mean1) @ prec1) * (f - mean1)).sum(axis=-1).min() for f in act_1.double()]).sum()
            #(((act_1.double() - mean1) @ prec1) * (act_1.double() - mean1)).sum(axis=-1).min()
            _ = layer_activations2(image, augment=augment)
            act_2 = layer_activations2.activations[0][0]
            score_ood_2 = - torch.stack([(((f - mean2) @ prec2) * (f - mean2)).sum(axis=-1).min() for f in act_2.double()]).sum()
            #(((act_2.double() - mean2) @ prec2) * (act_2.double() - mean2)).sum(axis=-1).min()
            print(score_ood_1, score_ood_2)
            loss = score_ood_2**2
            return loss
    elif loss =='max_single_maha_score':
        def loss_function(image,target_activation,augment=True):
            if len(image.shape) == 3:
                image = image[None, :]
            _ = layer_activations(image, augment=augment)
            act_1 = layer_activations.activations[0][0]
            score_ood_1 = - torch.stack([(((f - mean1) @ prec1) * (f - mean1)).sum(axis=-1).min() for f in act_1.double()]).sum()
            loss = - score_ood_1
            return loss
    elif loss =='max_single_maha_score_targeted':
        def loss_function(image,target_class_target_neuron,augment=True):
            target_class, _ = target_class_target_neuron
            if len(image.shape) == 3:
                image = image[None, :]
            _ = layer_activations(image, augment=augment)
            act_1 = layer_activations.activations[0][0]
            score_ood_1 = - torch.stack([(((f - mean1) @ prec1) * (f - mean1)).sum(axis=-1)[target_class] for f in act_1.double()]).sum()
            loss = -score_ood_1
            return loss
    elif loss =='min_single_maha_score':
        def loss_function(image,target_activation,augment=True):
            if len(image.shape) == 3:
                image = image[None, :]
            _ = layer_activations(image, augment=augment)
            act_1 = layer_activations.activations[0][0]
            score_ood_1 = - torch.stack([(((f - mean1) @ prec1) * (f - mean1)).sum(axis=-1).min() for f in act_1.double()]).sum()
            loss = score_ood_1
            return loss
    elif loss =='min_single_maha_score_targeted':
        def loss_function(image,target_class_target_neuron,augment=True):
            target_class, _ = target_class_target_neuron
            if len(image.shape) == 3:
                image = image[None, :]
            _ = layer_activations(image, augment=augment)
            act_1 = layer_activations.activations[0][0]
            score_ood_1 = - torch.stack([(((f - mean1) @ prec1) * (f - mean1)).sum(axis=-1)[target_class] for f in act_1.double()]).sum()
            loss = score_ood_1
            return loss
    elif loss=='max_maha_min_msp':
        def loss_function(image,target_activation,augment=True):
            if len(image.shape) == 3:
                image = image[None, :]
            _ = layer_activations(image, augment=augment)
            act_1 = layer_activations.activations[0][0]
            class_out1 = classifier.model.head(act_1)
            probs1 = torch.softmax(class_out1, dim=1)
            conf1s = probs1.max(1)[0].sum()
            score_ood_1 = - torch.stack([(((f - mean1) @ prec1) * (f - mean1)).sum(axis=-1).min() for f in act_1.double()]).sum()
            loss = -score_ood_1+500*conf1s
            return loss
    elif loss=='max_maha_min_msp_targeted':
        def loss_function(image,target_class_target_neuron,augment=True):
            target_class, _ = target_class_target_neuron
            if len(image.shape) == 3:
                image = image[None, :]
            _ = layer_activations(image, augment=augment)
            act_1 = layer_activations.activations[0][0]
            class_out1 = classifier.model.head(act_1)
            probs1 = torch.softmax(class_out1, dim=1)
            conf1s = probs1[:,target_class].sum()
            score_ood_1 = - torch.stack([(((f - mean1) @ prec1) * (f - mean1)).sum(axis=-1)[target_class] for f in act_1.double()]).sum()
            loss = -score_ood_1+500*conf1s
            return loss
    elif loss=='min_maha_max_msp': # needs to be finished
        def loss_function(image,target_activation,augment=True):
            if len(image.shape) == 3:
                image = image[None, :]
            _ = layer_activations(image, augment=augment)
            act_1 = layer_activations.activations[0][0]
            class_out1 = classifier.model.head(act_1)
            probs1 = torch.softmax(class_out1, dim=1)
            conf1s = probs1.max(1)[0].sum()
            score_ood_1 = - torch.stack([(((f - mean1) @ prec1) * (f - mean1)).sum(axis=-1).min() for f in act_1.double()]).sum()
            loss = score_ood_1-500*conf1s
            return loss
    elif loss=='min_maha_max_msp_targeted':
        def loss_function(image,target_class_target_neuron,augment=True):
            target_class, _ = target_class_target_neuron
            if len(image.shape) == 3:
                image = image[None, :]
            _ = layer_activations(image, augment=augment)
            act_1 = layer_activations.activations[0][0]
            class_out1 = classifier.model.head(act_1)
            probs1 = torch.softmax(class_out1, dim=1)
            conf1s = probs1[:,target_class].sum()
            score_ood_1 = - torch.stack([(((f - mean1) @ prec1) * (f - mean1)).sum(axis=-1)[target_class] for f in act_1.double()]).sum()
            loss = score_ood_1-500*conf1s
            return loss        
    elif loss == 'max_conf_target':
        def loss_function(image, target_class_target_neuron, augment=True):
            target_class, _ = target_class_target_neuron
            class_out = classifier(image, augment=augment)
            probs = torch.softmax(class_out, dim=1)
            log_conf = probs[:, target_class]
            loss = -log_conf.mean()
            return loss
    elif loss == 'min_conf_target':
        def loss_function(image, target_class_target_neuron, augment=True):
            target_class, _ = target_class_target_neuron
            class_out = classifier(image, augment=augment)
            probs = torch.softmax(class_out, dim=1)
            log_conf = probs[:, target_class]
            loss = log_conf.mean()
            return loss 
    else:
        raise NotImplementedError()

    return loss_function

def calculate_confs(classifier, imgs, device, target_class=None, return_predictions=False, return_entropy=False):
    confs = torch.zeros(imgs.shape[0])
    preds = torch.zeros(imgs.shape[0], dtype=torch.long)
    entropies = torch.zeros(imgs.shape[0])
    with torch.no_grad():
        for i in range(imgs.shape[0]):
            img = torch.clamp(imgs[i,:].to(device)[None, :], 0, 1)
            out = classifier(img, augment=False)
            probs = torch.softmax(out, dim=1)
            _, pred = torch.max(probs, dim=1)
            if target_class is None:
                conf, _ = torch.max(probs, dim=1)
            elif type(target_class) is list:
                conf, _ = torch.max(probs[:,target_class],dim=1)
            else:
                conf = probs[:, target_class]
            targets_uniform = torch.full_like(out, fill_value=1.0 / out.shape[1]) # uniform
            log_probs = F.log_softmax(out, dim=1)
            entropy = -torch.sum(targets_uniform * log_probs, dim=1)
            entropies[i] = entropy
            confs[i] = conf
            preds[i] = pred
    if return_predictions:
        if return_entropy:
            return confs, preds, entropies
        return confs, preds
    else:
        if return_entropy:
            confs, entropies
        return confs

def calculate_conf_diff(classifier, classifier_2, imgs, device, target_class=None):
    conf_diff = torch.zeros(imgs.shape[0])
    with torch.no_grad():
        for i in range(imgs.shape[0]):
            img = imgs[i,:].to(device)[None, :]
            out = classifier(img, augment=False)
            probs = torch.softmax(out, dim=1)
            if target_class is None:
                conf, target_class = torch.max(probs, dim=1)
            else:
                conf = probs[:, target_class]

            out_2 = classifier_2(img, augment=False)
            probs_2 = torch.softmax(out_2, dim=1)
            conf2 = probs_2[:, target_class]

            conf_diff[i]= conf - conf2

    return conf_diff


def calculate_neuron_activations(classifier, layer_activations, imgs, device, target, loss):
    losses = torch.zeros(imgs.shape[0])
    loss_function = get_feature_loss_function(loss, classifier, layer_activations)
    with torch.no_grad():
        for i in range(imgs.shape[0]):
            image = imgs[i,:].to(device)[None, :]
            neg_loss = loss_function(image, [None, target], augment=False)
            losses[i] = -neg_loss
    return losses

def calculate_maha_dists(layer_activations, imgs, device, mean1, prec1, transform, layer_activations2=None, mean2=None, prec2=None, classifier1=None, classifier2=None, target_class = None):
    neg_dists_maha = torch.zeros(imgs.shape[0],1 if layer_activations2 is None else 3)
    neg_dists_maha_raw = torch.zeros(imgs.shape[0],1 if layer_activations2 is None else 3)
    neg_dists_maha_target = torch.zeros(imgs.shape[0],1 if layer_activations2 is None else 3)
    neg_dists_maha_target_raw = torch.zeros(imgs.shape[0],1 if layer_activations2 is None else 3)   
    preds_maha = torch.zeros(imgs.shape[0],2)
    preds = torch.zeros(imgs.shape[0],2)
    max_confs = torch.zeros(imgs.shape[0],2)
    target_confs = torch.zeros(imgs.shape[0],2)
    # neg_dists_maha = torch.zeros(imgs.shape[0],1 if layer_activations2 is None else 3)
    with torch.no_grad():
        for i in range(imgs.shape[0]):
            image_raw = imgs[i,:]
            image = torch.clamp(imgs[i,:],0,1) #[None, :]
            image = transform(image)
            if len(image.shape) == 3:
                image = image[None, :]
                image_raw = image_raw[None, :]
            image=image.to(device)
            image_raw=image_raw.to(device)
            _ = layer_activations(image, augment=False)
            act_1 = layer_activations.activations[0][0]
            score_ood_1 = - torch.stack([(((f - mean1) @ prec1) * (f - mean1)).sum(axis=-1).min() for f in act_1.double()]).sum()
            pred_maha_1 = (((act_1.double() - mean1) @ prec1) * (act_1.double() - mean1)).sum(axis=-1).argmin().item()
            #torch.stack([(((f - mean1) @ prec1) * (f - mean1)).sum(axis=-1).argmax(-1) for f in act_1.double()])
            score_ood_1_target = - torch.stack([(((f - mean1) @ prec1) * (f - mean1)).sum(axis=-1)[target_class] for f in act_1.double()]).sum()
            _ = layer_activations(image_raw, augment=False)
            act_1_raw = layer_activations.activations[0][0]
            score_ood_1_raw = - torch.stack([(((f - mean1) @ prec1) * (f - mean1)).sum(axis=-1).min() for f in act_1_raw.double()]).sum()
            score_ood_1_target_raw = - torch.stack([(((f - mean1) @ prec1) * (f - mean1)).sum(axis=-1)[target_class] for f in act_1_raw.double()]).sum()
            neg_dists_maha[i,0]=score_ood_1
            neg_dists_maha_raw[i,0]=score_ood_1_raw
            preds_maha[i,0]=pred_maha_1
            neg_dists_maha_target[i,0]=score_ood_1_target
            neg_dists_maha_target_raw[i,0]=score_ood_1_target_raw           
            if layer_activations2 is not None:
                _ = layer_activations2(image, augment=False)
                act_2 = layer_activations2.activations[0][0]
                score_ood_2 = - torch.stack([(((f - mean2) @ prec2) * (f - mean2)).sum(axis=-1).min() for f in act_2.double()]).sum()
                # pred_maha_2 =  torch.stack([(((f - mean2) @ prec2) * (f - mean2)).sum(axis=-1).argmax(-1) for f in act_2.double()])
                pred_maha_2 = (((act_2.double() - mean2) @ prec2) * (act_2.double() - mean2)).sum(axis=-1).argmin().item()
                score_ood_2_target = - torch.stack([(((f - mean2) @ prec2) * (f - mean2)).sum(axis=-1)[target_class] for f in act_2.double()]).sum()

                _ = layer_activations2(image_raw, augment=False)
                act_2_raw = layer_activations2.activations[0][0]
                score_ood_2_raw = - torch.stack([(((f - mean2) @ prec2) * (f - mean2)).sum(axis=-1).min() for f in act_2_raw.double()]).sum()
                score_ood_2_target_raw = - torch.stack([(((f - mean2) @ prec2) * (f - mean2)).sum(axis=-1)[target_class] for f in act_2_raw.double()]).sum()
                neg_dists_maha[i,1]=score_ood_2
                neg_dists_maha_raw[i,1]=score_ood_2_raw
                neg_dists_maha[i,2]=score_ood_1-score_ood_2            
                neg_dists_maha_raw[i,2]=score_ood_1_raw-score_ood_2_raw         
                preds_maha[i,1]=pred_maha_2
                neg_dists_maha_target[i,1]=score_ood_2_target
                neg_dists_maha_target_raw[i,1]=score_ood_2_target_raw
                neg_dists_maha_target[i,2]=score_ood_1_target-score_ood_2_target            
                neg_dists_maha_target_raw[i,2]=score_ood_1_target_raw-score_ood_2_target_raw    
            for j, classifier in enumerate([classifier1,classifier2]):
                if classifier is not None:
                    out = classifier(image, augment=False)
                    out = torch.softmax(out,-1)
                    assert out.shape[-1]==1000
                    pred = torch.argmax(out,axis=-1).item()
                    conf = out[:,pred].item() 
                    preds[i,j]=pred
                    max_confs[i,j]=conf
                    target_confs[i,j]=out[:,target_class].item()
    return neg_dists_maha, preds_maha, max_confs, preds, neg_dists_maha_raw, neg_dists_maha_target, neg_dists_maha_target_raw, target_confs

def calculate_lp_distances(imgs, starting_imgs, ps = (1., 2.)):
    assert len(imgs) == len(starting_imgs)
    distances = {p: [] for p in ps}
    for i in range(len(imgs)):
        img = imgs[i].view(-1)
        start_img = starting_imgs[i].view(-1).to(img.device)

        for p in ps:
            d_i = torch.norm(img - start_img, p=p).item()
            distances[p].append(d_i)
    return distances

def calculate_lpips_distances(imgs, starting_imgs):
    assert len(imgs) == len(starting_imgs)
    distances = []

    loss_fn_alex = None
    device = None

    for i in range(len(imgs)):
        img = imgs[i]
        if img.dim() == 3:
            img = img[None, :, :, :]

        if loss_fn_alex is None:
            device = img.device
            loss_fn_alex = LPIPS(net='alex').to(device)
        else:
            img = img.to(device)

        start_img = starting_imgs[i]
        if start_img.dim() == 3:
            start_img = start_img[None, :, :, :].to(device)

        d = loss_fn_alex(img, start_img, normalize=True).mean().item()
        distances.append(d)
    return distances

def compute_losses(loss, neg_dists_maha, neg_dists_maha_target, confs, target_confs):
    if loss=='min_maha_max_msp_targeted':
        losses=neg_dists_maha_target[:,0]-500*target_confs[:,0]
    elif loss=='min_maha_max_msp':
        losses=neg_dists_maha[:,0]-500*confs[:,0]
    elif loss=='max_maha_min_msp_targeted':
        losses=-neg_dists_maha_target[:,0]+500*target_confs[:,0]
    elif loss=='max_maha_min_msp':
        losses=-neg_dists_maha[:,0]+500*confs[:,0]
    elif loss=='max_single_maha_score_targeted':
        losses = -neg_dists_maha_target[:,0]
    elif loss=='max_single_maha_score':
        losses = -neg_dists_maha[:,0]
    elif loss=='min_single_maha_score_targeted':
        losses = neg_dists_maha_target[:,0]
    elif loss=='min_single_maha_score':
        losses = neg_dists_maha[:,0]
    elif loss=='diff_maha':
        losses = neg_dists_maha[:,2]
    elif loss=='min_conf_target':
        losses = target_confs[:,0]
    elif loss=='max_conf_target':
        losses = -target_confs[:,0]
    else:
        raise NotImplementedError
    return losses