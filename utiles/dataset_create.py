import os
import mat73
import numpy as np
from utiles.fingerprint import * 
import math as mt
import json
import random

def preprocessing(N_csi, used_rx, data_position, nic_num):
    """
    N_csi: number of CSI measurements that used for denoise, if N_csi = 0 means using all CSI for denoising
    used_rx: number of using rx chains
    data_position: filter data from specific positions
    nic_num: filter nics
    """
    data_path = "./data/"
    Files = os.listdir(data_path+ "CSI/")
    all_NICs = ["mt7601c1", "mt7601c2", "mt7601c3", "mt7601c4", "mt7601c5", "mt7601c6", "mt7612c1", "rt3070c1", "8811cuc1", "8821cuc1", "rt8822c1",
                  "ar9271c1", "ar9271c2", "ar9271c3", "ar9271c4", "ar9271c5", "ar9271c6", "ar9271c7", "ax200c1","ux3h","xdn6000h","ESP32C5"]
    fdata = Fingerprint(N_csi, used_rx)
    SNRdata = []
    for nic in range(nic_num):
        CSI = []
        for i in range(len(Files)):
            if all_NICs[nic] in Files[i] and any(s in Files[i] for s in data_position):
                mat = mat73.loadmat(data_path + "CSI/" + str(Files[i]))
                CSI.append(mat["CSI"])
                name = Files[i]
                name = "MAC" + name[3:-4] +".json"
                f = open(data_path + "MAC/" + str(name))
                macinfo = np.asarray(json.load(f))
                SNRdata = np.concatenate((SNRdata, np.reshape(macinfo[:,:,6], -1))) # order: firt rx index then frame index
        fdata.get_fingerprint(CSI)
        
    data = list(fdata.devices.values())
    return data, SNRdata

def awgn(sig, reqSNR):
    """
    Apply Additive white Gaussian noise to sig
    """
    sigPower = np.sum(np.abs(sig)**2,axis=1) / sig.shape[1]
    reqSNR = 10**(reqSNR/10)
    noisePower = sigPower / reqSNR
    
    if np.iscomplexobj(sig):
        noise = np.sqrt(noisePower/2).T * (np.random.randn(*sig.shape) + 1j*np.random.randn(*sig.shape)).T
    else:
        noise = np.sqrt(noisePower) * np.random.randn(*sig.shape)
    
    y = sig + noise.T
    return y

def synthesis_data(da_type, N_csi, used_rx, data_position, channel_type, channel_num_per_channeltype, model, nic_num):
    """
    da_type: data augmentation type, 0 for fingerprint, 1 for denoised csi
    N_csi: number of CSI measurements that used for denoise, if N_csi = 0 means using all CSI for denoising
    used_rx: number of using rx chains
    data_position: filter data from specific positions
    channel_type: channel model types (model B-los B-nlos C-los C-nlos D-los D-nlos F) [0, 1, 2, 3, 4, 5, 6]
    channel_num_per_channeltype: number of used channels per channel type
    model: deep learning model in use
    nic_num: filter nics
    """
    scsi_clear = np.array(mat73.loadmat('./data/channel_BCDF.mat')["scsi_clear"])

    data, _ = preprocessing(N_csi, used_rx, data_position, nic_num)
    SNR = np.arange(5, 41, 5) + 3 # In 802.11 protocol, after measuring SNR using two LTS symbols, it average estimated CSI from these two symbols. 
                                    # In this case, the SNR of reported CSI is about 3db higher than groundtruth channel SNR.
    num_per_nic = 6 
    index = np.arange(int(channel_num_per_channeltype))
    
    frame_num = num_per_nic * len(SNR) * len(index) * len(channel_type) * nic_num
    Xdata = np.zeros((frame_num, 52),dtype=complex)
    Ydata = np.zeros((frame_num))
    SNRdata = np.zeros((frame_num))
    frame_start = 0

    for i in range(nic_num):
        csi = []
        for sc in channel_type:
            p_csi = np.empty((0,52)) 
            # random pick num_per_nic csi from practical denoised data for data synthesis
            for c in range(len(data[i][0])):
                used=random.sample(range(0, data[i][da_type][c].shape[1]), (num_per_nic// len(data[i][0])))
                p_csi = np.append(p_csi, data[i][da_type][c][:, used ].T,axis=0 )

            for n in index:
                used = random.randrange(5000)
                syn_channel = scsi_clear[sc][0][2:54, used] 
                csi = syn_channel * p_csi
                num_new_frame = csi.shape[0]
                
                for snr in SNR:
                    csi_f_noise = awgn(csi, snr) 
                    Xdata[frame_start + np.arange(num_new_frame), :] = csi_f_noise #np.divide(csi_f_noise.T , A  @ csi_f_noise.T).T #csi_f_noise
                    Ydata[frame_start + np.arange(num_new_frame)] = np.repeat(i + 1, num_new_frame)
                    SNRdata[frame_start + np.arange(num_new_frame)] = np.repeat(snr - 3, num_new_frame)
                    frame_start = frame_start + num_new_frame


    if model =="self-acc":
        '''
        Nonlinear phase error in [Eliminating rogue access point attacks in iot: 
        A deep learning approach with physical-layer feature purification and device identification]
        '''
        subcarriers_index = list(np.arange(-26,0)) + list(np.arange(1,27))
        Xdata= np.unwrap(np.angle(Xdata),axis=1)
        z = (Xdata[:,-1]+ Xdata[:,0]) /2
        k = (Xdata[:,-1]+ Xdata[:,0]) / (112 * mt.pi)
        for j in range(52):
            Xdata[:,j] = Xdata[:,j] - (2 * mt.pi * k *subcarriers_index[j]) - z
        Xdata = Xdata[:,:,None]
    else:
        Xdata = np.stack((np.real(Xdata), np.imag(Xdata)), axis=2)
    
    data = dict(Xdata = Xdata, Ydata = Ydata, SNRdata = np.double(SNRdata))
    return data

def practical_data(N_csi, used_rx, data_position, testing, model, nic_num):
    """
    Prepare practical CSI as deep learning model input
    """
    Xdata = np.empty((0,52))
    Ydata = np.empty((0,))
    for position in data_position:
        if position in ['d3','d10'] and testing: # 10% data from ['d3','d10'] for testing
            data, SNRdata = preprocessing(N_csi, used_rx, [position], nic_num)
            for i in range(len(data)):
                for c in range(len(data[i][0])):
                    csi = np.squeeze(data[i][1][c]).T
                    csi = csi[-int(csi.shape[0]*0.1):,:]
                    Xdata = np.append(Xdata,csi,axis=0)
                    Ydata = np.append(Ydata, np.repeat(i+1, len(csi)))
        else: 
            data, SNRdata = preprocessing(N_csi, used_rx, [position], nic_num) # 100% data from unseen positions for testing
            for i in range(len(data)):
                for c in range(len(data[i][0])):
                    csi = np.squeeze(data[i][1][c]).T
                    Xdata = np.append(Xdata,csi,axis=0)
                    Ydata = np.append(Ydata, np.repeat(i+1, len(csi)))
    
    index = (Xdata == np.zeros([1, 52])).all(-1) #delect invalid data
    Xdata = np.delete(Xdata,index,axis=0)
    Ydata = np.delete(Ydata,index,axis=0)
    
    if model =="self-acc":
        subcarriers_index = list(np.arange(-26,0)) + list(np.arange(1,27))
        Xdata= np.unwrap(np.angle(Xdata),axis=1)
        z = (Xdata[:,-1]+ Xdata[:,0]) /2
        k = (Xdata[:,-1]+ Xdata[:,0]) / (112 * mt.pi)
        for j in range(52):
            Xdata[:,j] = Xdata[:,j] - (2 * mt.pi * k *subcarriers_index[j]) - z
        Xdata = Xdata[:,:,None]
    else:
        Xdata = np.stack((np.real(Xdata), np.imag(Xdata)), axis=2)
    data = dict(Xdata = Xdata, Ydata = Ydata, SNRdata = np.double(SNRdata))
    return data

def practical_fingerprint(N_csi, used_rx, data_position, testing, nic_num):
    """
    Prepare estimated fingerprint as SS baseline input
    """
    Xdata = np.empty((0,52))
    Ydata = np.empty((0,))
    for position in data_position:
        if position in ['d3','d10'] and testing:
            data, SNRdata = preprocessing(N_csi, used_rx, [position], nic_num)
            for i in range(len(data)):
                for c in range(len(data[i][0])):
                    csi = np.squeeze(data[i][0][c]).T
                    csi = csi[-int(csi.shape[0]*0.1):,:]
                    Xdata = np.append(Xdata,csi,axis=0)
                    Ydata = np.append(Ydata, np.repeat(i+1, len(csi)))
        else: 
            data, SNRdata = preprocessing(N_csi, used_rx, [position], nic_num)
            for i in range(len(data)):
                for c in range(len(data[i][0])):
                    csi = np.squeeze(data[i][0][c]).T
                    Xdata = np.append(Xdata,csi,axis=0)
                    Ydata = np.append(Ydata, np.repeat(i+1, len(csi)))

    index = np.isnan(Xdata).all(-1) 
    Xdata = np.delete(Xdata,index,axis=0)
    Ydata = np.delete(Ydata,index,axis=0)
    Xdata = np.stack((np.real(Xdata), np.imag(Xdata)), axis=1)

    data = dict(Xdata = Xdata, Ydata = Ydata, SNRdata = np.double(SNRdata))
    return data
