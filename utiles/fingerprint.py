import numpy as np
from scipy.linalg import dft

def SS_estimation(csi):
    csi_complex = csi[:,:,0] + 1j*csi[:,:,1]
    DFT = dft(64)
    n_taps=8
    K = list(range(38, 64))+list(range(1, 27))
    L = list(range(0,n_taps+2))+list(range(64-n_taps+1,64))
    P_DFT=DFT[K,:]
    P_DFT=np.matrix(P_DFT[:,L])
    A=np.matmul(  P_DFT,np.matmul( np.linalg.inv(np.matmul(P_DFT.getH(),P_DFT)) ,P_DFT.getH()  )  )
    CSI_R = np.matmul(A,np.transpose(csi_complex))
    Fingerprint = np.divide(csi_complex , np.transpose(CSI_R))
    return Fingerprint

class Fingerprint:
    def __init__(self, N_csi, rx):
        """
        N_csi: number of CSI measurements that used for denoise, if N_csi = 0 means using all CSI for denoising
        used_rx: number of using rx chains
        data_position: filter data from specific positions
        nic_num: filter nics
        """
        self.N_csi = N_csi
        self.devices = {}
        self.usedrx = rx

        """
        Initialize parameters for SS-based fingerprint extraction. 
        Please see more details in eq. (2) and (3) in CSI-RFF: Leveraging Micro-Signals on CSI for RF Fingerprinting of Commodity WiFi
        """
        DFT = dft(64)
        n_taps=8
        K = list(range(38, 64))+list(range(1, 27))
        L = list(range(0,n_taps+2))+list(range(64-n_taps+1,64))
        P_DFT = DFT[K,:]
        P_DFT = np.array(P_DFT[:,L])
        A = np.matmul(  P_DFT,np.matmul( np.linalg.inv(np.matmul(P_DFT.conj().T,P_DFT)) ,P_DFT.conj().T  )  )
        self.A = A

    def get_fingerprint(self, CSIs):
        pGF = []
        pGCSI = []
        for n in range(len(CSIs)):
            F, CSI = self.get_micro_csi(CSIs[n])
            if len(F)==0:
                continue
            pGF = pGF + [F]
            pGCSI = pGCSI + [CSI]

        pre_device = len(self.devices)
        self.devices["device" + str(pre_device + 1)] = [pGF, pGCSI]
            

    def get_micro_csi(self, CSI):
        # CSI shape: 52 * rx * num
        CSI = CSI[:,self.usedrx,:]
        if CSI.shape[2] < self.N_csi:
            F=[]
            CSI=[]
            H_ls=[]
            return F, CSI

        nrx = CSI.shape[1]
        scn = CSI.shape[0]
        if self.N_csi == 0:
            num = CSI.shape[2]
        else:
            num = self.N_csi
 
        if num == 1:
            CSI = np.reshape(np.transpose(CSI, (0, 2, 1)), (52, -1)) # CSI order [csi_{1,rx1},csi_{1,rx2},csi_{1,rx3},csi_{1,rx4},...,csi_{N,rx1},csi_{N,rx2},csi_{N,rx3},csi_{N,rx4}]
            H_ls = self.A  @ CSI
            F = np.divide(CSI , H_ls) 
        else:
            CSI = np.reshape(CSI, (52,-1)) # CSI order [csi_{1,rx1},csi_{2,rx1},...,csi_{N,rx1},csi_{1,rx2},...,csi_{N,rx2},csi_{1,rx3},...,csi_{N,rx3},csi_{1,rx4},...csi_{N,rx4}]
            index = (CSI.T == np.zeros([1,52])).all(-1) #find invalid data
            CSI = np.delete(CSI, index, axis=1) #delect invalid data
            CSI = CSI/np.mean(CSI,axis=0) 
            CSI = np.reshape(CSI[:,0:num * (CSI.shape[1] // num)], (scn, -1, num))
            CSI = np.reshape(np.mean(CSI, axis=2), (52,-1))
            H_ls = self.A  @ CSI
            fingerprint = CSI / H_ls
            gf = np.gradient(fingerprint, axis=0)
            variance = np.var(gf, axis=0)
            index = np.where(variance < 2 * 10 ** (-3))[0] #find invalid fingerprint
            F = fingerprint[ :,index] #delect invalid fingerprint
            CSI = CSI[:, index]
        return F, CSI


