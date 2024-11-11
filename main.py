from utiles.dataset import *
from utiles.trainer import *
import model.utils as model_utils
import torch
import os
import shutil
import datetime
import hydra
from omegaconf import OmegaConf
import logging

def set_logger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:'
                                                    ' %(message)s'))
        logger.addHandler(file_handler)
        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


@hydra.main(version_base=None, config_path="config", config_name="config_train")
def main(args: OmegaConf):
    
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H_%M_%S')

    args.checkpoint_dir = args.checkpoint_dir + '{}'.format(nowtime) +'/'
    if not os.path.exists(os.path.abspath(args.checkpoint_dir)):
        os.makedirs(os.path.abspath(args.checkpoint_dir))
    else:
        shutil.rmtree(args.checkpoint_dir, ignore_errors=True)
        shutil.rmtree(args.log_dir, ignore_errors=True)

    set_logger(os.path.join(args.checkpoint_dir, 'process.log'))
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    
    OmegaConf.save(config=args, f=os.path.join(args.checkpoint_dir, 'conf.yaml'))

    traindataset, validate_dataset, testdataset = get_datasets(args)

    train_dataloader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=worker_init_fn)
    val_dataloder = torch.utils.data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=worker_init_fn)
    test_dataloder = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size)

    model,criterion,optimizer = get_model_loss_optimizer(args)

    if args.Model == 'deepcrf-con':
        if args.load_model is not None:
            print("Continuing training full model from checkpoint " + str(args.load_model))
            state = model_utils.load_model(model, optimizer, args.load_model, args.cuda)
        else:
            state = training(train_dataloader, val_dataloder, test_dataloder, model, optimizer, criterion, args)
        args.Model = 'deepcrf'
        args.loss = 'cross'
        model,criterion,optimizer = get_model_loss_optimizer(args)
        state = model_utils.load_model(model, None, state["best_checkpoint"], args.cuda)
        state = training(train_dataloader, val_dataloder, test_dataloder, model, optimizer, criterion, args)
    else:
        if args.load_model is not None:
            print("Continuing training full model from checkpoint " + str(args.load_model))
            state = model_utils.load_model(model, None, args.load_model, args.cuda)
            if args.cuda:
                model = model.cuda()
            state = training(train_dataloader, val_dataloder, test_dataloder, model, optimizer, criterion, args)
        else:
            state = training(train_dataloader, val_dataloder, test_dataloder, model, optimizer, criterion, args)

    



if __name__ == '__main__':
    main()