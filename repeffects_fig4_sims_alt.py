import numpy as np
import scipy as sp
import torch

def produce_basic_statistics(y, plag):
    #initialise arrays
    n_subs = y.shape[0]
    # y_tensor = torch.stack(list(y.values())) # 18 x n_trials x n_voxels
    
    # avgAmp1r1 = torch.zeros((n_subs, 1), dtype=torch.float32, requires_grad=True)
    # avgAmp2r1 = torch.zeros((n_subs, 1), dtype=torch.float32, requires_grad=True)
    # avgAmp1r2 = torch.zeros((n_subs, 1), dtype=torch.float32, requires_grad=True)
    # avgAmp2r2 = torch.zeros((n_subs, 1), dtype=torch.float32, requires_grad=True)
    # wtc1 = torch.zeros((n_subs, 1), dtype=torch.float32, requires_grad=True)
    # wtc2 = torch.zeros((n_subs, 1), dtype=torch.float32, requires_grad=True)
    # # BC between class correlations
    # btc1 = torch.zeros((n_subs, 1), dtype=torch.float32, requires_grad=True)
    # btc2 = torch.zeros((n_subs, 1), dtype=torch.float32, requires_grad=True)
    # # CP classification performance
    # svm_init = torch.zeros((n_subs, 1), dtype=torch.float32, requires_grad=True)
    # svm_rep = torch.zeros((n_subs, 1), dtype=torch.float32, requires_grad=True)

    nBins = 6

    # sc_trend = torch.zeros((n_subs, nBins), dtype=torch.float32, requires_grad=True)
    # abs_ad_trend = torch.zeros((n_subs, nBins), dtype=torch.float32, requires_grad=True)

    """pattern was previously n_trials x n_voxels array for specific sub"""
    AM = calculate_AM(y)
    # print(n//4)
    #cond1_p1 has dimension 18 x 8 x 200
    # cond1_p1_corr = torch.corrcoef(cond1_p1.reshape(subs, n//4 * v).T).reshape(subs, n//4, n//4)
    # cond2_p1_corr = torch.corrcoef(cond2_p1.reshape(subs, n//4 * v).T).reshape(subs, n//4, n//4)
    # mean_corr = (cond1_p1_corr + cond2_p1_corr) / 2
    # wtc1[:, 0] = mean_corr.mean(dim=(1, 2))
    # print(wtc1)
    # print(wtc1.shape)


    # n = y.shape[1]
    # v = y.shape[2]
    # cond1_p1 = y[:,:n // 4, :v]
    # cond1_p2 = y[:, n // 4:n // 2, :v]
    # cond2_p1 = y[:, n // 2:3 * n // 4, :v]
    # cond2_p2 = y[:, 3 * n // 4:, :v]

    """This a test I am adding in this line"""
    
    """This is the previous code. I now want to vectorise this by setting 
        cond1_p1_corr to calculate for all sub at once. The calculate_AM
        function below shows how the AM looks after being vectorised"""
    # for sub in range(y.shape[0]):
    #     cond1_p1_corr = (torch.corrcoef(cond1_p1.T) + torch.corrcoef(cond2_p1.T)) / 2
    #     wtc1[sub, :] = torch.mean(torch.mean(cond1_p1_corr, axis=0))
    
    

    # cond1_p2_corr = (torch.corrcoef(cond1_p2.T)+torch.corrcoef(cond2_p2.T)) / 2
    # wtc2[sub, :] = torch.mean(torch.mean(cond1_p2_corr, axis=0))
    WC = calculate_WC(y)
    print("WC")
    print(WC)
    BC = calculate_BC(y)
    CP = calculate_CP(WC, BC)
    AMS = calculate_AMS(y)
    AMA = calculate_AMA(y)

    # print("WC.grad_fn")
    # print(WC.grad_fn)
    # print(WC)

    # def calculate_WC(y):
    #     n_subs = len(y) #y is a length 18 dictionary with arrays 32 x 200
    #     wtc1 = torch.zeros((n_subs, 1), dtype=torch.float32, requires_grad=True)
    #     for sub in range(n_subs):
    #         pattern = y[sub]
    #         v = pattern.shape[1]
    #         n = pattern.shape[0]
    #         cond1_p1 = pattern[:n // 4, :v]
    #         cond2_p1 = pattern[n // 2:3 * n // 4, :v]
    #         cond1_p1_corr = (torch.corrcoef(cond1_p1.T) + torch.corrcoef(cond2_p1.T)) / 2
    #         wtc1[sub, :] = torch.mean(torch.mean(cond1_p1_corr, axis=0))
    #     return wtc1


    # # print(y_tensor.shape)
    # for sub in range(n_subs):
    #     pattern = y[sub]
    #     v = pattern.shape[1] # voxels
    #     n = pattern.shape[0] # trials


    #     if n % 4 != 0:
    #         raise ValueError("Assumes 4 conditions with equal trials")
        
    #     # cond1_p = [
    #     #     pattern[:n // 4, :v],               # Rows 1 to n/4
    #     #     pattern[n // 4:n // 2, :v],         # Rows n/4 + 1 to n/2
    #     # ]
    #     # cond2_p = [
    #     #     pattern[n // 2:3 * n // 4, :v],     # Rows n/2 + 1 to 3n/4
    #     #     pattern[3 * n // 4:, :v],           # Rows 3n/4 + 1 to n
    #     # ]

    #     cond1_p1 = pattern[:n // 4, :v]
    #     cond1_p2 = pattern[n // 4:n // 2, :v]
    #     cond2_p1 = pattern[n // 2:3 * n // 4, :v]
    #     cond2_p2 = pattern[3 * n // 4:, :v]
    #     # print("cond1_p1")
    #     # print(cond1_p1)


    #     # Compute means
    #     # cond1r1 = torch.mean(cond1_p1, axis=0)
    #     # # print(cond1r1)
    #     # # print(cond1r1.size)
    #     # cond2r1 = torch.mean(cond2_p1, axis=0)
    #     # cond1r2 = torch.mean(cond1_p2, axis=0)
    #     # cond2r2 = torch.mean(cond2_p2, axis=0)
    #     # # temp_mean = torch.mean(cond1r1, axis=0)
    #     # # mask = torch.zeros_like(avgAmp1r1, dtype=torch.bool)
    #     # # mask[sub,:] = True
    #     # # expanded_values = temp_mean.expand(sub, -1)
    #     # # avgAmp1r1.masked_scatter_(mask, expanded_values)
    #     # avgAmp1r1[sub, :] = torch.mean(cond1r1, axis=0)
    #     # avgAmp2r1[sub, :] = torch.mean(cond2r1, axis=0)
    #     # avgAmp1r2[sub, :] = torch.mean(cond1r2, axis=0)
    #     # avgAmp2r2[sub, :] = torch.mean(cond2r2, axis=0)
    #     # avgR1 = (avgAmp1r1+avgAmp2r1)/2
    #     # avgR2 = (avgAmp1r2+avgAmp2r2)/2

    #     """AMA adaptation by amplitude"""
    #     # Sorting and indexing for AMA adaptation by amplitude
    #     # Combine and compute mean across rows, then sort
    #     sAmp1r = torch.sort(torch.mean(torch.hstack([cond1_p1.T, cond1_p2.T]), axis=1))
    #     ind1 = torch.argsort(torch.mean(torch.hstack([cond1_p1.T, cond1_p2.T]), axis=1))

    #     sAmp2r = torch.sort(torch.mean(torch.hstack([cond2_p1.T, cond2_p2.T]), axis=1))
    #     ind2 = torch.argsort(torch.mean(torch.hstack([cond2_p1.T, cond2_p2.T]), axis=1))

    #     # Reorder based on indices
    #     sAmp1r1 = cond1r1[ind1]
    #     sAmp2r1 = cond2r1[ind2]
    #     sAmp1r2 = cond1r2[ind1]
    #     sAmp2r2 = cond2r2[ind2]

    #     # Compute slope
    #     sAmp = ((sAmp1r1 - sAmp1r2) + (sAmp2r1 - sAmp2r2)) / 2



    #     # pp1 = torch.corrcoef(cond1_p[0].T, cond2_p[0].T, rowvar=False)
    #     pp1 = torch.corrcoef(torch.cat([cond1_p1.T, cond2_p1.T], dim=0))
    #     pp11 = (cond1_p1.T).shape[1]
    #     pp1 = pp1[:pp11, pp11:]

    #     btc1[sub, :] = torch.mean(torch.mean(pp1, axis=0))

    #     # pp2 = torch.corrcoef(cond1_p[1].T, cond2_p[1].T, rowvar=False)
    #     pp2 = torch.corrcoef(torch.cat([cond1_p2.T, cond2_p2.T], dim=0))
    #     pp22 = (cond1_p2.T).shape[1]
    #     pp2 = pp2[:pp22, pp22:]
    #     btc2[sub, :] = torch.mean(torch.mean(pp2, axis=0))

    #     svm_init[sub, :] = wtc1[sub, :] - btc1[sub, :]
    #     svm_rep[sub, :] = wtc2[sub, :] - btc2[sub, :]

    #     # Perform t-tests
    #     #tval1, pval1 = sp.stats.ttest_ind(torch.vstack([cond1_p[0], cond1_p[1]]), torch.vstack([cond2_p[0], cond2_p[1]]), axis=0)
    #     #tval2, pval2 = sp.stats.ttest_ind(torch.vstack([cond2_p[0], cond2_p[1]]), torch.vstack([cond1_p[0], cond1_p[1]]), axis=0)
    #     # """Detachment"""
    #     # # Detach tensors and convert to NumPy arrays
    #     # cond1_combined = torch.vstack([cond1_p[0], cond1_p[1]]).detach().numpy()
    #     # cond2_combined = torch.vstack([cond2_p[0], cond2_p[1]]).detach().numpy()

    #     # # Perform t-test using SciPy
    #     # tval1, pval1 = sp.stats.ttest_ind(cond1_combined, cond2_combined, axis=0)
        
    #     # # Detach tensors and convert to NumPy
    #     # cond2_combined = torch.vstack([cond2_p[0], cond2_p[1]]).detach().numpy()
    #     # cond1_combined = torch.vstack([cond1_p[0], cond1_p[1]]).detach().numpy()

    #     # # Perform t-test using SciPy
    #     # tval2, pval2 = sp.stats.ttest_ind(cond2_combined, cond1_combined, axis=0)
    #     # """End of Detachment"""

    #     cond1_combined = torch.vstack([cond1_p1, cond1_p2])
    #     cond2_combined = torch.vstack([cond2_p1, cond2_p2])
    #     tval1 = pytorch_ttest(cond1_combined, cond2_combined)

    #     cond2_combined = torch.vstack([cond2_p1, cond2_p2])
    #     cond1_combined = torch.vstack([cond1_p1, cond1_p2])
    #     tval2 = pytorch_ttest(cond2_combined, cond1_combined)





    #     # Sorting the t-values by their absolute values
    #     tval_sorted_ind1 = torch.argsort(torch.abs(torch.tensor(tval1, dtype=torch.float32, requires_grad=True)))
    #     tval_sorted_ind2 = torch.argsort(torch.abs(torch.tensor(tval2, dtype=torch.float32, requires_grad=True)))

    #     # Compute means for conditions
    #     c1_init = torch.mean(cond1_p1, axis=0)
    #     c1_rep = torch.mean(cond1_p2, axis=0)
    #     c2_init = torch.mean(cond2_p1, axis=0)
    #     c2_rep = torch.mean(cond2_p2, axis=0)

    #     # Reorder based on sorted indices
    #     c1_sinit = c1_init[tval_sorted_ind1]
    #     c1_srep = c1_rep[tval_sorted_ind1]
    #     c2_sinit = c2_init[tval_sorted_ind2]
    #     c2_srep = c2_rep[tval_sorted_ind2]

    #     # Compute trends
    #     abs_init_trend = (c1_sinit + c2_sinit) / 2
    #     abs_rep_trend = (c1_srep + c2_srep) / 2
    #     abs_adaptation_trend = abs_init_trend - abs_rep_trend
    #     #print(abs_adaptation_trend)

    #     #Binning the AMA and AMS trends
    #     AA = sAmp
    #     AS = abs_adaptation_trend

    #     # Compute the percentage indices (similar to MATLAB's rounding and indexing)
    #     #percInds = (torch.round((torch.arange(1, len(AA) + 1) * (nBins - 1)) / len(AA)) / (nBins - 1)) * (nBins - 1) + 1
    #     percInds = (torch.round((torch.arange(1, len(AA) + 1) * (nBins - 1)) / len(AA)) / (nBins - 1)) * (nBins - 1)



    #     for i in range(nBins):
    #         sc_trend[sub, i] = torch.mean(AA[percInds == i], axis=0)
    #         abs_ad_trend[sub, i] = torch.mean(AS[percInds == i], axis=0)



    

    return AM, CP, WC, BC, AMA, AMS

def calculate_AM(y):
    v = y.shape[2]
    n = y.shape[1] #y has shape 18 x 32 x 200
    subs = y.shape[0]
    cond1_p1 = y[:,:n // 4, :v]
    # print(cond1_p1)
    cond1_p2 = y[:, n // 4:n // 2, :v]
    cond2_p1 = y[:, n // 2:3 * n // 4, :v]
    cond2_p2 = y[:, 3 * n // 4:, :v]
    cond1r1 = torch.mean(cond1_p1, axis=1)
    cond2r1 = torch.mean(cond2_p1, axis=1)
    cond1r2 = torch.mean(cond1_p2, axis=1)
    cond2r2 = torch.mean(cond2_p2, axis=1)
    # print(cond1r1.shape)
    avgAmp1r1 = torch.mean(cond1r1, axis=1)
    avgAmp2r1 = torch.mean(cond2r1, axis=1)
    avgAmp1r2 = torch.mean(cond1r2, axis=1)
    avgAmp2r2 = torch.mean(cond2r2, axis=1)

    avgR1 = ((avgAmp1r1+avgAmp2r1)/2).unsqueeze(1)
    avgR2 = ((avgAmp1r2+avgAmp2r2)/2).unsqueeze(1)
    return torch.column_stack((avgR1, avgR2))

def calculate_WC(y):
    n_subs = y.shape[0]
    n = y.shape[1]
    v = y.shape[2]
    cond1_p1 = y[:,:n // 4, :v]
    cond2_p1 = y[:, n // 2:3 * n // 4, :v]

    cond1_p1_centered = cond1_p1 - cond1_p1.mean(dim=1, keepdim=True)
    cond2_p1_centered = cond2_p1 - cond2_p1.mean(dim=1, keepdim=True)

    def batched_corr(x):
        #x = [n_subs, n_timepoints, n_voxels]
        # x_t = x.transpose(1,2)
        x_t = x.transpose(1,2)
        cov = torch.matmul(x_t, x_t.transpose(1,2)) / (x.shape[1] -1)
        std = x_t.std(dim=2)
        return cov / std[:,:,None] * std[:, None, :] + 1e-6
        
        # cov = torch.einsum('snt,smt->snm', x, x) / (x.shape[1] - 1) # 18x8x8
        # print("cov.shape")
        # print(cov.shape)
        # std = x.std(dim=1, keepdim=False) # 18x200
        # print("std.shape")
        # print(std.shape)
        # print("std[:, None, :].shape")
        # print(std[:, None, :].shape)
        # corr = cov / (std[:, None, :] * std[:, :, None] + 1e-6)
        # return corr
    corr1_p1 = batched_corr(cond1_p1_centered)
    corr2_p1 = batched_corr(cond2_p1_centered)
    avgcorr1 = (corr1_p1 + corr2_p1) / 2
    wtc1 = avgcorr1.mean(dim=(1,2)).unsqueeze(1)
    print("wtc1.shape")
    print(wtc1.shape)

    cond1_p2 = y[:, n // 4:n // 2, :v]
    cond2_p2 = y[:, 3 * n // 4:, :v]

    cond1_p2_centered = cond1_p2 - cond1_p2.mean(dim=1, keepdim=True)
    cond2_p2_centered = cond2_p2 - cond2_p2.mean(dim=1, keepdim=True)

    corr1_p2 = batched_corr(cond1_p2_centered)
    corr2_p2 = batched_corr(cond2_p2_centered)
    avgcorr2 = (corr1_p2 + corr2_p2) / 2
    wtc2 = avgcorr2.mean(dim=(1, 2)).unsqueeze(1)



    return torch.column_stack((wtc1, wtc2))

def calculate_BC(y):
    subs = y.shape[0]
    btc1 = torch.rand(subs, 1, requires_grad=True)
    btc2 = torch.rand(subs, 1, requires_grad=True)
    return torch.column_stack((btc1, btc2))

def calculate_CP(WC, BC):
    return WC - BC

def calculate_AMS(y):
    nBins = 6
    subs = y.shape[0]
    return torch.rand(subs, nBins, requires_grad=True)

def calculate_AMA(y):
    nBins = 6
    subs = y.shape[0]
    return torch.rand(subs, nBins, requires_grad=True)

    
def produce_confidence_interval(y, pflag):
    
    AM, CP, WC, BC, AMA, AMS = produce_basic_statistics(y, pflag)
    # print(AM.grad_fn)

    # print("AM")
    # print(AM)
    # print("CP")
    # print(CP)
    # print("WC")
    # print(WC)
    # print("BC")
    # print(BC)
    # print("AMA")
    # print(AMA)
    # print("AMS")
    # print(AMS)

    def compute_slope(data):
        L = data.shape[1]
        X = torch.vstack((torch.arange(1, L+1), torch.ones(L))).T
        pX = torch.linalg.pinv(X)
        return torch.matmul(pX, data.T)[0]

    slopes = torch.stack([torch.mean(compute_slope(data)) for data in [AM, WC, BC, CP, AMS, AMA]
                          ])
    print("slopes.grad_fn")
    print(slopes.grad_fn)
    print(slopes)
    return slopes


# # Define functions to compute Pearson correlation matrices
# def corr_matrix(X, Y=None):

#     Y = X if Y is None else Y
#     return torch.corrcoef(torch.hstack((Y,X)), rowvar=False)[0,1:]

def pytorch_ttest(cond1, cond2):
    mean1 = torch.mean(cond1, dim=0)
    mean2 = torch.mean(cond2, dim=0)
    std1 = torch.std(cond1, dim=0)
    std2 = torch.std(cond2, dim=0)
    n1 = cond1.size(0)
    n2 = cond2.size(0)
    t_stat = (mean1 - mean2) / torch.sqrt((std1**2/n1)+(std2**2/n2))
    return t_stat