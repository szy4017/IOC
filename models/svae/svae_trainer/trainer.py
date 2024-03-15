import os
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from utils.metrics import *
from sklearn.metrics import roc_auc_score, average_precision_score


def Trainer(model, model_optimizer, train_dl, val_dl, test_dl, device, logger, config, xp_path):
    # Start training
    logger.debug("Training started ....")

    # save_path = "./best_network/" + config.dataset + config.scene
    save_path = os.path.join(xp_path,  str(config.scene).zfill(2) + '_best_network.pkl')
    # os.makedirs(save_path, exist_ok=True)
    # early_stopping = EarlyStopping(save_path, config.scene)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    scheduler = optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=config.lr_milestones, gamma=0.1)
    all_epoch_train_loss, all_epoch_test_loss = [], []
    
    if model.center is None:
        logger.info('Initializing center c...')
        model.center = init_center_c(train_dl, model, device, config.project_channels, config.center_eps)
        logger.info('Center c initialized.')
    # model.center = center_c(train_dl, model, device, model.center, config, eps=config.center_eps)
        
    logger.info('Start training')
    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        scheduler.step()
        train_loss, _, _, _  = model_train(model, model_optimizer, train_dl, config, device, epoch)
        # val_target, val_score_origin, val_loss, all_projection = model_evaluate(model, val_dl, center, length, config,
        #                                                                     device, epoch)
        test_loss, test_indices, test_labels, test_scores = model_evaluate(model, test_dl, config, device)
        # if epoch < config.change_center_epoch:
        #     model.center = center_c(train_dl, model, device, model.center, config, eps=config.center_eps)
        # scheduler.step(train_loss)
        # logger.debug(f'\nEpoch : {epoch}\n'
        #              f'Train Loss     : {train_loss:.4f}\t | \n'
        #              f'Test Loss     : {test_loss:.4f}\t  | \n'
        #              )
        all_epoch_train_loss.append(train_loss)
        all_epoch_test_loss.append(test_loss)

        # auc = compute_auc(test_score, test_target)
        # bf1, thre = compute_bestf1(test_score, test_target, return_threshold=True)
        # wbestpred = (test_score >= thre)#.type(torch.cuda.FloatTensor)
        # wbestresult = compute_dacc(wbestpred, test_target)
        # p, r, f1, iou = compute_precision_recall(wbestresult)
        # print(f"epoch: {epoch}: auc: {auc:.3f}; bf1: {bf1:.3f}, precision: {p:.3f}; recall: {r:.3f}; f1: {f1:.3f}")

        auc = 100. * roc_auc_score(test_labels, test_scores)
        ap  = 100. * average_precision_score(test_labels, test_scores)
        logger.info(f'Epoch {epoch}: train loss: {train_loss} ; AUC :{auc:.2f}%; AP: {ap:.2f}%')

        torch.save(model, save_path)


    logger.debug("\n################## Training is Done! #########################")
    # according to scores to create predicting labels
    
    model = torch.load(save_path)
    test_loss, test_indices, test_labels, test_scores = model_evaluate(model, test_dl, config, device)

    # auc = compute_auc(test_score, test_target)
    # bf1, thre = compute_bestf1(test_score, test_target, return_threshold=True)
    # wbestpred = (test_score >= thre)#.type(torch.cuda.FloatTensor)
    # wbestresult = compute_dacc(wbestpred, test_target)
    # p, r, f1, iou = compute_precision_recall(wbestresult)
    # logger.debug(f"Final results: auc: {auc:.3f}; bf1: {bf1:.3f}, precision: {p:.3f}; recall: {r:.3f}; f1: {f1:.3f}")
    
    auc = 100. * roc_auc_score(test_labels, test_scores)
    ap  = 100. * average_precision_score(test_labels, test_scores)
    logger.info(f'Final results:  AUC :{auc:.2f}%; AP: {ap:.2f}%')

    return test_indices, test_labels, test_scores

def Tester(test_dl, device, logger, config, xp_path):
    logger.debug("Testinging started ....")
    save_path = os.path.join(xp_path,  str(config.scene).zfill(2) + 'best_network.pkl')
    
    model = torch.load(save_path)
    test_target, test_score, test_indices, test_loss, all_projection = model_evaluate(model, test_dl, config, device)
    
    
    auc = compute_auc(test_score, test_target)
    bf1, thre = compute_bestf1(test_score, test_target, return_threshold=True)
    wbestpred = (test_score >= thre)#.type(torch.cuda.FloatTensor)
    wbestresult = compute_dacc(wbestpred, test_target)
    p, r, f1, iou = compute_precision_recall(wbestresult)
    logger.debuge(f"auc: {auc:3f}; bf1: {bf1:3f}, precision: {p:3f}; recall: {r:3f}; f1: {f1:3f}; iou: {iou:3f}")


    return test_indices, test_target, test_score



def model_train(model, optimizer, train_loader,  config, device, epoch):

    loss_epoch = 0.0
    n_batches = 0
    idx_label_score = []

    model.train()
    # torch.autograd.set_detect_anomaly(True)
    for batch_idx, data in enumerate(train_loader):
        # send to device
        inputs, labels, idx = data
        inputs = inputs.float().to(device)
        # optimizer
        optimizer.zero_grad()
        outputs, feature = model(inputs) # ()
        loss, score = train(inputs, outputs, feature, model.center, model.length, config, device)
        # loss, score = train2(feature1, center, target, length, epoch, config, device)
        # Update hypersphere radius R on mini-batch distances
        if (config.objective == 'soft-boundary') and (epoch >= config.freeze_length_epoch):
            model.length = torch.tensor(get_radius(score, config.nu), device=device)
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        n_batches += 1
        idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                labels.cpu().data.numpy().tolist(),
                                score.cpu().data.numpy().tolist()))


    train_loss = loss_epoch / n_batches
    indices, labels, scores = zip(*idx_label_score)
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)

    return train_loss, indices, labels, scores


def model_evaluate(model, test_dl,  config, device):
    model.eval()
    loss_epoch = 0.0
    n_batches = 0
    idx_label_score = []
    with torch.no_grad():
        for data in test_dl:
            inputs, labels, idx = data 
            inputs = inputs.to(device)
            outputs, feature = model(inputs)
            loss, score = train(inputs, outputs, feature,  model.center, model.length, config, device)
            loss_epoch += loss.item()
            n_batches += 1

            idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                        labels.cpu().data.numpy().tolist(),
                                        score.cpu().data.numpy().tolist()))

    test_loss = loss_epoch / n_batches  # average loss
    indices, labels, scores = zip(*idx_label_score)
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)

    return test_loss, indices, labels, scores



def train(inputs, outputs, feature, center, length, config, device):
    # normalize feature vectors
    mseloss = nn.MSELoss(reduction='none')
    dist = torch.sum((feature - center) ** 2, dim=1)
    if config.objective == 'soft-boundary':
        scores_oc = dist - length ** 2 
        loss_oc = length ** 2 + (1 / config.nu) * torch.mean(torch.max(torch.zeros_like(scores_oc), scores_oc))
    else:
        scores_oc = dist 
        loss_oc = torch.mean(scores_oc)
    
    scores_rec = torch.mean(mseloss(outputs, inputs).view(inputs.size(0), -1), dim=1)
    loss_rec = torch.mean(scores_rec)

    scores = scores_rec + scores_oc
    loss = loss_oc + loss_rec

    return loss, scores


def init_center_c(train_loader, model, device, rep_dim, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = torch.zeros(rep_dim, device=device)

    model.eval()
    with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            inputs, _,_ = data
            inputs = inputs.to(device)
            _, latent = model(inputs)
            n_samples += latent.shape[0]
            c += torch.sum(latent, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def center_c(train_loader, model, device, center, config, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = center
    model.eval()
    with torch.no_grad():
        for data, target, idx in train_loader:
            # get the inputs of the batch
            data = data.float().to(device)
            # aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
            # all_data = torch.cat((data, aug1, aug2), dim=0)
            _, outputs = model(data)
            n_samples += outputs.shape[0]
            # all_feature = torch.cat((outputs, dec), dim=0)
            # all_feature = outputs
            c += torch.sum(outputs, dim=0)
    # c /= (2 * n_samples)
    c /= (n_samples)

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    # return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
    dist = dist.reshape(-1)
    return np.quantile(dist.clone().data.cpu().numpy(), 1 - nu)