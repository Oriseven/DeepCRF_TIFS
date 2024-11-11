import torch
import numpy as np
from utiles.dataset_create import *

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, data_fusion, Ncsi, SNR=None):
        """
        data_fusion: enable data fusion (1) or not (0)
        Ncsi: number of CSI group together of data fusion
        SNR: filter CSI of certain snr values
        """
        
        self.csi = torch.from_numpy(data['Xdata']).float()
        w = self.csi.shape[2]
        if data_fusion == 0:
            
            self.csi = self.csi[:,None,:,:]
            self.label = torch.from_numpy(np.squeeze(np.array(data['Ydata']))).float()
            self.snr = torch.from_numpy(np.squeeze(np.array(data['SNRdata']))).float()

            if not SNR is None:
                # filter snr values
                indices=[i for i,j in enumerate( np.array(self.snr)) if j in np.array(SNR)]
                self.label=self.label[indices]
                self.csi=self.csi[indices]
                self.snr=self.snr[indices]
        else:
            self.label = torch.from_numpy(np.array(data['Ydata'])).float()
            if not SNR is None:
                self.snr = torch.from_numpy(np.array(data['SNRdata'])).float()
                indices = np.where(self.snr==SNR)
                self.label=self.label[indices]
                self.csi=self.csi[indices]
                self.snr=self.snr[indices]

            indices = np.argsort(self.label,kind='mergesort')
            self.label=self.label[indices]
            self.csi=self.csi[indices]
            for nic in np.unique(self.label):
                index=np.where(self.label==nic)
                num=len(index[0])
                index=index[0][int(np.floor(num//Ncsi) *Ncsi):num]
                self.label = np.delete(self.label, index)
                self.csi = np.delete(self.csi, index, axis=0)

            self.label = self.label[0::Ncsi]
            self.csi = np.reshape(self.csi,(self.csi.shape[0]//Ncsi,Ncsi,52,w))
   
            complex_csi = self.csi[:,:,:,0] + 1j* self.csi[:,:,:,1]
            self.csi = torch.zeros(self.csi.shape[0],1,self.csi.shape[2],self.csi.shape[3]).float()
            complex_csi_amp = torch.mean(np.abs(complex_csi),axis=1)
            complex_csi_phase = np.mean(np.unwrap(np.angle(complex_csi),axis=2),axis=1)
            complex_csi = complex_csi_amp * np.exp(1j*complex_csi_phase)
            self.csi[:,0,:,0] = np.real(complex_csi)
            self.csi[:,0,:,1] = np.imag(complex_csi)
        

    def __getitem__(self, index):
        x = self.csi[index]
        y = self.label[index]
        return x, y

    def __len__(self):
        return len(self.csi)


def get_datasets(args):
    # training with augmented CSI data
    data = synthesis_data(da_type= args.da, N_csi=40, used_rx=[0,1,2,3],data_position= args.train_positions_syn,channel_type=args.channel_type, 
                            channel_num_per_channeltype= args.train_channel_num_per_channeltype, model=args.Model, nic_num=args.class_num)
    if args.Model == 'ss':
            Xdata = np.array(SS_estimation(data['Xdata']))
            data['Xdata']=np.stack((np.real(Xdata), np.imag(Xdata)), axis=1)
    traindataset = MyDataset(data,args.data_fusion,args.Ncsi, args.snr)
    data = synthesis_data(da_type= args.da, N_csi=40, used_rx=[0,1,2,3],data_position= args.train_positions_syn,channel_type=args.channel_type,
                            channel_num_per_channeltype= args.val_channel_num_per_channeltype, model=args.Model, nic_num=args.class_num)
    if args.Model == 'ss':
            Xdata = np.array(SS_estimation(data['Xdata']))
            data['Xdata']=np.stack((np.real(Xdata), np.imag(Xdata)), axis=1)
    validatedataset = MyDataset(data,args.data_fusion,args.Ncsi, args.snr)

    if args.Model == 'ss':
        # Prepare estimated fingerprints from collected CSI files as SS baseline input
        data = practical_fingerprint(1, [0,1,2,3], args.test_positions, 1, args.class_num)
    else:
        data = practical_data(1, [0,1,2,3], args.test_positions, 1, model=args.Model,nic_num=args.class_num)
    testdataset = MyDataset(data,args.data_fusion,args.Ncsi, args.snr)
    
    if args.train_with_practical_data :
        if args.Model == 'ss':
            data = practical_fingerprint(1, [0,1,2,3], args.train_positions, 0, args.class_num)
        else:
            data = practical_data(1, [0,1,2,3], args.train_positions, 0, model=args.Model,nic_num=args.class_num)
        practicaldataset = MyDataset(data,args.data_fusion,args.Ncsi, args.snr)
        dataset_size = len(practicaldataset)
        train_size = int(0.8 * dataset_size)
        val_size = int(0.1 * dataset_size)
        test_size = dataset_size - train_size - val_size
        train_practicaldataset, validate_practicaldataset, _ = torch.utils.data.random_split(practicaldataset, [train_size, val_size, test_size])
        traindataset = torch.utils.data.ConcatDataset([traindataset, train_practicaldataset])
        validatedataset = torch.utils.data.ConcatDataset([validatedataset, validate_practicaldataset])
        
    return traindataset, validatedataset, testdataset