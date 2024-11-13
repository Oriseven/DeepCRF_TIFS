from model.models import *
from utiles.dataset import *
import torch.nn as nn
import model.utils as model_utils
from torch.optim import Adam
from tqdm import tqdm
import time
import os
from utiles.losses import SupConLoss
import logging

def worker_init_fn(worker_id): # This is apparently needed to ensure workers have different random seeds and draw different examples!
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def training(train_dataloader, val_dataloder, test_dataloder, model, optimizer, criterion, args):
 
    val_losses = []
    train_losses = []
    val_accs = []
    train_accs = []
    if args.cuda:
        model = model_utils.DataParallel(model)
        print("move model to gpu")
        model.cuda()

    print('model: ', model)
    print('parameter count: ', str(sum(p.numel() for p in model.parameters())))

    state = {"step" : 0,
            "worse_epochs" : 0,
            "epochs" : 0,
            "best_loss" : np.Inf,
            "best_acc": 0}

    print('TRAINING START')
    training_start_time = time.time()
    while (state["worse_epochs"] < args.patience) & (state["epochs"] < args.numepoch):
        print("Training the " + str(state["epochs"])+" epoch from iteration " + str(state["step"]))
        avg_time = 0.
        model.train()
        with tqdm(total=len(train_dataloader.sampler) // args.batch_size) as pbar: 
            np.random.seed()
            for example_num, (x, targets) in enumerate(train_dataloader):
                if args.cuda:
                    x = x.cuda()
                    targets = (targets - 1).type(torch.LongTensor).cuda()

                t = time.time()

                optimizer.zero_grad()
                _, _, loss, acc = model_utils.compute_loss(model, x, targets, criterion)
                
                loss.backward()
                
                optimizer.step()

                state["step"] += 1

                t = time.time() - t
                avg_time += (1. / float(example_num + 1)) * (t - avg_time)

                train_losses.append(loss.item())
                train_accs.append(acc.item())
                pbar.update(1)
        # VALIDATE
        val_loss, val_acc, _, _ = validate(args, model, criterion, val_dataloder)
        print(f"VALIDATION FINISHED: LOSS: " + str(val_loss)+ "Acc " + str(val_acc))

        val_losses.append(val_loss.item())
        val_accs.append(val_acc.item())
        # EARLY STOPPING CHECK
        checkpoint_path = os.path.join(args.checkpoint_dir, f"best_checkpoint")
        if val_loss >= state["best_loss"]:
            state["worse_epochs"] += 1
        else:
            print("MODEL IMPROVED ON VALIDATION SET!")
            state["worse_epochs"] = 0
            state["best_loss"] = val_loss
            state["best_checkpoint"] = checkpoint_path
            state["best_acc"] = val_acc
            # CHECKPOINT
            print("Saving model...")
            model_utils.save_model(model, optimizer, state, checkpoint_path)

        logging.info('epoch: {}, train_loss: {:.4f}, train_accuracy:{:.4f}, val_loss: {:.4f}, val_accuracy:{:.4f}'.format(
                        state["epochs"],sum(train_losses) / len(train_losses), sum(train_accs) / len(train_accs),
                        val_loss.item(),val_acc.item())
                    )
        state["epochs"] += 1
    print('Training finished, took {:.2f}s'.format(time.time() - training_start_time))

    #### TESTING ####
    print("TESTING")

    # Load best model based on validation loss
    state = model_utils.load_model(model, None, state["best_checkpoint"], args.cuda)
    test_loss,test_acc, _, _= validate(args, model,criterion, test_dataloder)
    print(f"TEST FINISHED: LOSS: " + str(test_loss) + "Acc " + str(test_acc))
    logging.info('epoch: {}, test_loss: {:.4f},test_accuracy:{:.4f}'.format(
                    state["epochs"],
                    test_loss.item(),
                    test_acc.item())
                )
    return state

def get_model_loss_optimizer(args):

    if args.Model == 'ss':
        model = SSModule(args.class_num)
    elif args.Model == 'deepcrf-con':
        model = DeepCRFConNet(head='linear', feat_dim=52, in_channels=1, d=args.d, act_fn_name=args.af)
    elif args.Model == 'deepcrf':   
        model = DeepCRF(in_channels=1, out_channels=args.class_num, d=args.d, act_fn_name=args.af)
    elif args.Model == 'complex-deepcrf':
        model = complex_DeepCRF(1,args.class_num,args.d, args.af)
    elif args.Model == 'self-acc':
        model = Self_ACC(args.class_num)
    elif args.Model == 'att_network':
        input_shape = [512,1,52,2]
        model = att_network(args.class_num, input_shape)

    # Set up the loss function
    if args.loss =='cross':
        criterion = nn.CrossEntropyLoss()
    elif args.loss =='SVM':
        criterion = nn.MultiMarginLoss()
    elif args.loss =='contrastive':
        criterion = SupConLoss(temperature=0.07)
    else:
        raise NotImplementedError("Couldn't find this loss!")
    
    # Set up optimiser
    if args.loss =='contrastive':
        optimizer = Adam(params=model.parameters(), lr=args.optimizer.lr_contrastive, weight_decay=args.optimizer.weight_decay)
    else:
        optimizer = Adam(params=model.parameters(), lr=args.optimizer.lr, weight_decay=args.optimizer.weight_decay)

    return model, criterion, optimizer

def validate(args, model, criterion, test_data):

    # VALIDATE
    model.eval()
    total_loss = 0.
    total_acc = 0.
    y_pred = []
    y_true = []
    with tqdm(total=len(test_data.sampler)// args.batch_size) as pbar, torch.no_grad():
        for example_num, (x, targets) in enumerate(test_data):
            if args.cuda:
                x = x.cuda()
                targets = (targets - 1).type(torch.LongTensor).cuda()

            predic_label, true_label, avg_loss, acc = model_utils.compute_loss(model, x, targets, criterion)

            total_loss += avg_loss
            total_acc += acc
            y_pred.extend(predic_label.cpu().numpy())
            y_true.extend(true_label.cpu().numpy())

            pbar.set_description("Current loss: {:.4f}, Current acc: {:.4f}".format(total_loss/(example_num+1),total_acc/(example_num+1)))
            pbar.update(1)

    return total_loss/(example_num+1), total_acc/(example_num+1), y_pred, y_true