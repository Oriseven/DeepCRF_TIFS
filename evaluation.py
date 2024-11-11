from model.models import *
from utiles.dataset import *
from utiles.trainer import *
from utiles.dataset_create import *
import torch
import model.utils as model_utils
from sklearn.metrics import confusion_matrix
import hydra
from omegaconf import OmegaConf
import scipy.io as sio
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []

def majority_voting(nic_feature_reshape):
    # Apply mode across the first dimension (axis=0), i.e., for each sample across all classifiers
    predictions = np.argmax(nic_feature_reshape,axis=2)
    majority_vote = np.squeeze(mode(predictions, axis=1).mode)
    return majority_vote

def average_probabilities(nic_feature_reshape):
    sum_pred = np.sum(nic_feature_reshape, axis=1)
    lab_merge_max = np.argmax(sum_pred,axis=1)
    return lab_merge_max

def product_probabilities(nic_feature_reshape):
    product_probs = np.prod(nic_feature_reshape, axis=1)
    return np.argmax(product_probs, axis=1)

def borda_count(nic_feature_reshape):
    # Number of samples, classifiers, and classes
    num_samples = nic_feature_reshape.shape[0]
    num_classifiers = nic_feature_reshape.shape[1]
    num_classes = nic_feature_reshape.shape[2]

    # Convert the class probabilities to rankings for each sample and classifier
    rankings = (-nic_feature_reshape).argsort(axis=2).argsort(axis=2)

    # Initialize the Borda count scores
    borda_scores = np.zeros((num_samples, num_classes))

    # Calculate Borda count scores for each sample
    for i in range(num_samples):
        for j in range(num_classifiers):
            borda_scores[i] += num_classes - 1 - rankings[i, j]

    # Determine the winner for each sample based on the Borda scores
    return np.argmax(borda_scores, axis=1)
    
def plot_confusion(y_true,y_predict,figure_name,args):
    cf_matrix = confusion_matrix(y_true,y_predict,normalize="true")                            
    per_cls_acc = cf_matrix.diagonal()/cf_matrix.sum(axis=1)
    class_names = np.arange(1, args.class_num+1)
    print('max: {}, ave: {}, min: {}'.format(np.round(np.max(per_cls_acc*100),2),np.round(np.mean(per_cls_acc*100),2),np.round(np.min(per_cls_acc*100),2)))
                                                    
    print("Plot confusion matrix")
    figure_name =  os.path.dirname(args.checkpoint_dir) + figure_name 
    df_cm = pd.DataFrame(cf_matrix, class_names, class_names)    
    fig = plt.figure(figsize = (args.class_num-5,args.class_num-5))
    sns.set(font_scale=1.2)
    sns.heatmap(df_cm, annot=True,fmt='.2f',cmap='Blues',cbar=False)
    plt.xlabel("Prediction", fontsize=30)
    plt.ylabel("Ground Truth", fontsize=30)
    fig.tight_layout()
    plt.savefig(figure_name)

def decision_fusion(y_output, y_true, args, NCSI):
    predict = np.empty([0])
    true = np.empty([0])

    for nic in np.arange(args.class_num):
        index = [i for i, val in enumerate(y_true) if val == nic]
        nic_feature = y_output[index[:(len(index) // NCSI)*NCSI],:]
        nic_feature_reshape = np.reshape(nic_feature,(-1,NCSI,args.class_num))
    
        # lab_predict = majority_voting(nic_feature_reshape)
        lab_predict = average_probabilities(nic_feature_reshape)
        # lab_predict = product_probabilities(nic_feature_reshape)
        # lab_predict = borda_count(nic_feature_reshape)

        predict = np.append(predict, lab_predict, axis=0)
        true = np.append(true,nic*np.ones(len(lab_predict)),axis=0)
    figure_name = args.figure_name + '_' + str(NCSI)
    plot_confusion(true,predict,figure_name,args)

@hydra.main(version_base=None, config_path="config", config_name="config_evaluation")
def main(args: OmegaConf):
    if args.test_with_practical_data:
        if args.Model == 'ss':
            data = practical_fingerprint(1, [0,1,2,3], args.test_positions,1,args.class_num)
        else:
            data = practical_data(1,[0,1,2,3], args.test_positions,0,args.Model,args.class_num)
        print('channel: {}'.format(args.test_positions))
    else:
        channel=['B_L','B_NL','C_L','C_NL','D_L','D_NL','F']
        data = sio.loadmat('./data/syn_testing_'+ channel[args.channel_type]+'.mat') #using the same testing data for fair comparison
        
        # data = synthesis_data(da_type= args.da, N_csi=40, used_rx=[0,1,2,3],
        #                       data_position= args.test_positions_syn,
        #                       channel_type = [args.channel_type], 
        #                       channel_num_per_channeltype= args.test_channel_num_per_channeltype, 
        #                       model=args.Model, nic_num=args.class_num)
        # sio.savemat('./data/syn_testing_'+ channel[args.channel_type]+'.mat',data)

        if args.Model == 'ss':
            Xdata = np.array(SS_estimation(data['Xdata']))
            data['Xdata']=np.stack((np.real(Xdata), np.imag(Xdata)), axis=1)

        '''uncomment below two lines if input type should be amplitude and phase'''
        # Xdata=data['Xdata'][:,:,0] + 1j*data['Xdata'][:,:,1]
        # data['Xdata']=np.stack((np.abs(Xdata), np.angle(Xdata)), axis=2)

        if args.Model == 'self-acc':
            subcarriers_index = list(np.arange(-26,0)) + list(np.arange(1,27))
            Xdata = np.unwrap(np.angle(data['Xdata'][:,:,0] + 1j*data['Xdata'][:,:,1]),axis=1)
            z = (Xdata[:,-1]+ Xdata[:,0]) /2
            k = (Xdata[:,-1]+ Xdata[:,0]) / (112 * mt.pi)
            for j in range(52):
                Xdata[:,j] = Xdata[:,j] - (2 * mt.pi * k *subcarriers_index[j]) - z
            Xdata = Xdata[:,:,None]
            data['Xdata']= Xdata
        print('channel: {}, SNR: {}'.format(args.channel_type,args.snr))
    dataset = MyDataset(data, args.data_fusion, args.Ncsi, args.snr)
    dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.num_workers)
    
    ## load model
    model, criterion, _ = get_model_loss_optimizer(args)
    _ = model_utils.load_model(model, None, args.checkpoint_dir, args.cuda)
    model.eval()
    print('parameter count: ', str(sum(p.numel() for p in model.parameters())))

    save_output = SaveOutput()
    hook_handles = []
    # get all the model children as list
    model_children = list(model.children())
    # output layer
    if args.Model == 'ss':
        handle = model_children[0][0].register_forward_hook(save_output)
    elif args.Model == 'deepcrf-con':
        raise NotImplementedError("Only for training!")
    elif args.Model == 'deepcrf':   
        handle = model_children[1][1].register_forward_hook(save_output)
    elif args.Model == 'complex-deepcrf':
        handle = model_children[3][2].register_forward_hook(save_output)
    elif args.Model == 'self-acc':
        handle = model_children[24].register_forward_hook(save_output)
    elif args.Model == 'att_network':
        handle = model_children[15].register_forward_hook(save_output)
    hook_handles.append(handle)
    
    model = model_utils.DataParallel(model)
    model.cuda()

    _, _, y_predict, y_true = validate(args, model,criterion, dataloader)
    outputs = save_output.outputs
    figure_name = args.figure_name + '_1'
    plot_confusion(y_true, y_predict, figure_name, args)

    # Visualize the feature maps
    feature_map = np.empty([0, outputs[0].shape[1]])
    for _, t in enumerate(outputs):
        if args.Model == 'complex-deepcrf':
            feature_map = np.append(feature_map, np.abs( t.cpu().detach().numpy()),axis=0)
        else:
            feature_map = np.append(feature_map,  t.cpu().detach().numpy(),axis=0)
        
    
    # decision_fusion(feature_map,y_true,args,4)
    # decision_fusion(feature_map,y_true,args,8)
    # decision_fusion(feature_map,y_true,args,12)
    decision_fusion(feature_map,y_true,args,16)
    # decision_fusion(feature_map,y_true,args,20)

if __name__ == '__main__':
    main()