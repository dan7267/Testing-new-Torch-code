import numpy as np
import scipy as sp
import torch

def produce_basic_statistics(y, plag):
    #initialise arrays
    #Now y is shape (18, 32, 200)
    #So the different subs are along the first dimension.
    n_subs = y.shape[0]
    v = y.shape[2]
    n = y.shape[1]
    nBins = 6

    #MAM
    # cond1_p = torch.stack([y[:, :n // 4, :v], y[:, n // 4:n // 2, :v]], dim=1)  # (n_subs, 2, n_voxels)
    # cond2_p = torch.stack([y[:, n // 2:3 * n // 4, :v], y[:, 3 * n // 4:, :v]], dim=1)  # (n_subs, 2, n_voxels)
    # print("cond1_p.shape")
    # print(cond1_p.shape)
    # cond1_mean = torch.mean(cond1_p, dim=2)
    # cond2_mean = torch.mean(cond2_p, dim=2)

    cond1_p1 = y[:, :n//4, :]  # First quarter
    cond1_p2 = y[:, n//4:2*n//4, :]
    cond2_p1 = y[:, 2*n//4:3*n//4, :]
    cond2_p2 = y[:, 3*n//4:, :]

    # Compute means across trials
    cond1r1 = cond1_p1.mean(dim=1)
    cond1r2 = cond1_p2.mean(dim=1)
    cond2r1 = cond2_p1.mean(dim=1)
    cond2r2 = cond2_p2.mean(dim=1)

    avgAmp1r1 = cond1r1.mean(dim=1, keepdim=True)
    avgAmp2r1 = cond2r1.mean(dim=1, keepdim=True)
    avgAmp1r2 = cond1r2.mean(dim=1, keepdim=True)
    avgAmp2r2 = cond2r2.mean(dim=1, keepdim=True)

    avgR1 = (avgAmp1r1 + avgAmp2r1) / 2
    avgR2 = (avgAmp1r2 + avgAmp2r2) / 2

    print(torch.column_stack((avgR1, avgR2)))

    # Compute AMA by amplitude (sAmp) for each subject
    cond1_combined = torch.cat([cond1_p[:, 0, :, :], cond1_p[:, 1, :, :]], dim=2)  # Combine conditions 1 and 2
    cond2_combined = torch.cat([cond2_p[:, 0, :, :], cond2_p[:, 1, :, :]], dim=2)  # Combine conditions 1 and 2
    sAmp1r = torch.mean(cond1_combined, axis=2)  # Shape: (n_subs, n_voxels)
    sAmp2r = torch.mean(cond2_combined, axis=2)  # Shape: (n_subs, n_voxels)

    # Sorting and computing the slopes for AMA adaptation
    ind1 = torch.argsort(sAmp1r, dim=1)
    ind2 = torch.argsort(sAmp2r, dim=1)
    
    # Compute slopes
    sAmp = ((sAmp1r - sAmp1r) + (sAmp2r - sAmp2r)) / 2

    # Compute within-class correlations
    wtc1 = torch.corrcoef(cond1_p[:, 0, :, :].transpose(1, 2).reshape(n_subs, -1))
    wtc2 = torch.corrcoef(cond1_p[:, 1, :, :].transpose(1, 2).reshape(n_subs, -1))

    # Compute between-class correlations
    btc1 = torch.corrcoef(torch.cat([cond1_p[:, 0, :, :], cond2_p[:, 0, :, :]], dim=1).transpose(1, 2).reshape(n_subs, -1))
    btc2 = torch.corrcoef(torch.cat([cond1_p[:, 1, :, :], cond2_p[:, 1, :, :]], dim=1).transpose(1, 2).reshape(n_subs, -1))

    # Compute SVM-based classification performance
    svm_init = wtc1.mean(dim=1, keepdim=True) - btc1.mean(dim=1, keepdim=True)
    svm_rep = wtc2.mean(dim=1, keepdim=True) - btc2.mean(dim=1, keepdim=True)

    #AMA
    sc_trend = torch.zeros((n_subs, nBins), dtype=torch.float32, requires_grad=True)
    abs_ad_trend = torch.zeros((n_subs, nBins), dtype=torch.float32, requires_grad=True)

    
    sAmp1r = torch.sort(cond1_p.mean(dim=2), dim=2).values
    sAmp2r = torch.sort(cond2_p.mean(dim=2), dim=2).values

    percInds = torch.linspace(0, nBins - 1, steps=v).long()  # Create bin indices
    for i in range(nBins):
        sc_trend[:, i] = torch.mean(sAmp1r[:, :, percInds == i], dim=(1, 2))
        abs_ad_trend[:, i] = torch.mean(sAmp2r[:, :, percInds == i], dim=(1, 2))


    # avgAmp1r1 = torch.zeros((n_subs, 1))
    # avgAmp2r1 = torch.zeros((n_subs, 1))
    # avgAmp1r2 = torch.zeros((n_subs, 1))
    # avgAmp2r2 = torch.zeros((n_subs, 1))
    # wtc1 = torch.zeros((n_subs, 1), dtype=torch.float32, requires_grad=True)
    # wtc2 = torch.zeros((n_subs, 1), dtype=torch.float32, requires_grad=True)
    # # BC between class correlations
    # btc1 = torch.zeros((n_subs, 1), dtype=torch.float32, requires_grad=True)
    # btc2 = torch.zeros((n_subs, 1), dtype=torch.float32, requires_grad=True)
    # # CP classification performance
    # svm_init = torch.zeros((n_subs, 1), dtype=torch.float32, requires_grad=True)
    # svm_rep = torch.zeros((n_subs, 1), dtype=torch.float32, requires_grad=True)

    # nBins = 6

    # sc_trend = torch.zeros((n_subs, nBins), dtype=torch.float32, requires_grad=True)
    # abs_ad_trend = torch.zeros((n_subs, nBins), dtype=torch.float32, requires_grad=True)

    # for sub in range(n_subs):
    #     pattern = y[sub]
    #     v = pattern.shape[1] # voxels
    #     n = pattern.shape[0] # trials


        # if n % 4 != 0:
        #     raise ValueError("Assumes 4 conditions with equal trials")
        
        # cond1_p = [
        #     pattern[:n // 4, :v],               # Rows 1 to n/4
        #     pattern[n // 4:n // 2, :v],         # Rows n/4 + 1 to n/2
        # ]
        # cond2_p = [
        #     pattern[n // 2:3 * n // 4, :v],     # Rows n/2 + 1 to 3n/4
        #     pattern[3 * n // 4:, :v],           # Rows 3n/4 + 1 to n
        # ]


        # Compute means
        # cond1r1 = torch.mean(cond1_p[0], axis=0)
        # print(cond1r1)
        # print(cond1r1.size)
        # cond2r1 = torch.mean(cond2_p[0], axis=0)
        # cond1r2 = torch.mean(cond1_p[1], axis=0)
        # cond2r2 = torch.mean(cond2_p[1], axis=0)
        # avgAmp1r1[sub, :] = torch.mean(cond1r1, axis=0)
        # avgAmp2r1[sub, :] = torch.mean(cond2r1, axis=0)
        # avgAmp1r2[sub, :] = torch.mean(cond1r2, axis=0)
        # avgAmp2r2[sub, :] = torch.mean(cond2r2, axis=0)
        # avgR1 = (avgAmp1r1+avgAmp2r1)/2
        # avgR2 = (avgAmp1r2+avgAmp2r2)/2

        """AMA adaptation by amplitude"""
        # Sorting and indexing for AMA adaptation by amplitude
        # Combine and compute mean across rows, then sort
        # sAmp1r = torch.sort(torch.mean(torch.hstack([cond1_p[0].T, cond1_p[1].T]), axis=1))
        # ind1 = torch.argsort(torch.mean(torch.hstack([cond1_p[0].T, cond1_p[1].T]), axis=1))

        # sAmp2r = torch.sort(torch.mean(torch.hstack([cond2_p[0].T, cond2_p[1].T]), axis=1))
        # ind2 = torch.argsort(torch.mean(torch.hstack([cond2_p[0].T, cond2_p[1].T]), axis=1))

        # # Reorder based on indices
        # sAmp1r1 = cond1r1[ind1]
        # sAmp2r1 = cond2r1[ind2]
        # sAmp1r2 = cond1r2[ind1]
        # sAmp2r2 = cond2r2[ind2]

        # # Compute slope
        # sAmp = ((sAmp1r1 - sAmp1r2) + (sAmp2r1 - sAmp2r2)) / 2

        # cond1_p1_corr = (torch.corrcoef(cond1_p[0].T) + torch.corrcoef(cond2_p[0].T)) / 2

        # wtc1[sub, :] = torch.mean(torch.mean(cond1_p1_corr, axis=0))


        # cond1_p2_corr = (torch.corrcoef(cond1_p[1].T)+torch.corrcoef(cond2_p[1].T)) / 2
        # wtc2[sub, :] = torch.mean(torch.mean(cond1_p2_corr, axis=0))


        # # pp1 = torch.corrcoef(cond1_p[0].T, cond2_p[0].T, rowvar=False)
        # pp1 = torch.corrcoef(torch.cat([cond1_p[0].T, cond2_p[0].T], dim=0))
        # pp11 = (cond1_p[0].T).shape[1]
        # pp1 = pp1[:pp11, pp11:]

        # btc1[sub, :] = torch.mean(torch.mean(pp1, axis=0))

        # # pp2 = torch.corrcoef(cond1_p[1].T, cond2_p[1].T, rowvar=False)
        # pp2 = torch.corrcoef(torch.cat([cond1_p[1].T, cond2_p[1].T], dim=0))
        # pp22 = (cond1_p[1].T).shape[1]
        # pp2 = pp2[:pp22, pp22:]
        # btc2[sub, :] = torch.mean(torch.mean(pp2, axis=0))

        # svm_init[sub, :] = wtc1[sub, :] - btc1[sub, :]
        # svm_rep[sub, :] = wtc2[sub, :] - btc2[sub, :]

        # Perform t-tests
        #tval1, pval1 = sp.stats.ttest_ind(torch.vstack([cond1_p[0], cond1_p[1]]), torch.vstack([cond2_p[0], cond2_p[1]]), axis=0)
        #tval2, pval2 = sp.stats.ttest_ind(torch.vstack([cond2_p[0], cond2_p[1]]), torch.vstack([cond1_p[0], cond1_p[1]]), axis=0)
        # """Detachment"""
        # # Detach tensors and convert to NumPy arrays
        # cond1_combined = torch.vstack([cond1_p[0], cond1_p[1]]).detach().numpy()
        # cond2_combined = torch.vstack([cond2_p[0], cond2_p[1]]).detach().numpy()

        # # Perform t-test using SciPy
        # tval1, pval1 = sp.stats.ttest_ind(cond1_combined, cond2_combined, axis=0)
        
        # # Detach tensors and convert to NumPy
        # cond2_combined = torch.vstack([cond2_p[0], cond2_p[1]]).detach().numpy()
        # cond1_combined = torch.vstack([cond1_p[0], cond1_p[1]]).detach().numpy()

        # # Perform t-test using SciPy
        # tval2, pval2 = sp.stats.ttest_ind(cond2_combined, cond1_combined, axis=0)
        # """End of Detachment"""

        
        
        cond1_combined = torch.vstack([cond1_p[0], cond1_p[1]])
        cond2_combined = torch.vstack([cond2_p[0], cond2_p[1]])
        tval1 = pytorch_ttest(cond1_combined, cond2_combined)

        cond2_combined = torch.vstack([cond2_p[0], cond2_p[1]])
        cond1_combined = torch.vstack([cond1_p[0], cond1_p[1]])
        tval2 = pytorch_ttest(cond2_combined, cond1_combined)

        # Sorting the t-values by their absolute values
        tval_sorted_ind1 = torch.argsort(torch.abs(torch.tensor(tval1, dtype=torch.float32, requires_grad=True)))
        tval_sorted_ind2 = torch.argsort(torch.abs(torch.tensor(tval2, dtype=torch.float32, requires_grad=True)))

        # Compute means for conditions
        c1_init = torch.mean(cond1_p[0], axis=0)
        c1_rep = torch.mean(cond1_p[1], axis=0)
        c2_init = torch.mean(cond2_p[0], axis=0)
        c2_rep = torch.mean(cond2_p[1], axis=0)

        # Reorder based on sorted indices
        c1_sinit = c1_init[tval_sorted_ind1]
        c1_srep = c1_rep[tval_sorted_ind1]
        c2_sinit = c2_init[tval_sorted_ind2]
        c2_srep = c2_rep[tval_sorted_ind2]

        # Compute trends
        abs_init_trend = (c1_sinit + c2_sinit) / 2
        abs_rep_trend = (c1_srep + c2_srep) / 2
        abs_adaptation_trend = abs_init_trend - abs_rep_trend
        #print(abs_adaptation_trend)

        #Binning the AMA and AMS trends
        AA = sAmp
        AS = abs_adaptation_trend

        # Compute the percentage indices (similar to MATLAB's rounding and indexing)
        #percInds = (torch.round((torch.arange(1, len(AA) + 1) * (nBins - 1)) / len(AA)) / (nBins - 1)) * (nBins - 1) + 1
        percInds = (torch.round((torch.arange(1, len(AA) + 1) * (nBins - 1)) / len(AA)) / (nBins - 1)) * (nBins - 1)



        for i in range(nBins):
            sc_trend[sub, i] = torch.mean(AA[percInds == i], axis=0)
            abs_ad_trend[sub, i] = torch.mean(AS[percInds == i], axis=0)


        

    return torch.column_stack((avgR1, avgR2)), torch.column_stack((svm_init, svm_rep)), torch.column_stack((wtc1, wtc2)), torch.column_stack((btc1, btc2)), sc_trend, abs_ad_trend




# import torch

# # y=torch.tensor([[[ 3.3545e-01],
# #          [ 3.2854e-01],
# #          [ 3.8225e-01],
# #          [ 3.7080e-01],
# #          [ 3.2960e-01],
# #          [ 3.9118e-01],
# #          [ 3.7770e-01],
# #          [ 3.9081e-01],
# #          [ 9.6436e-02],
# #          [ 6.0175e-02],
# #          [ 5.6642e-02],
# #          [ 2.8732e-02],
# #          [ 4.0023e-02],
# #          [ 1.0841e-02],
# #          [ 4.5967e-02],
# #          [ 3.8519e-02],
# #          [ 1.2325e-01],
# #          [ 1.5234e-01],
# #          [ 1.0936e-01],
# #          [ 1.1581e-01],
# #          [ 1.6436e-01],
# #          [ 1.5507e-01],
# #          [ 1.2385e-01],
# #          [ 1.2393e-01],
# #          [ 3.2257e-02],
# #          [ 4.5760e-02],
# #          [-1.3099e-02],
# #          [ 2.0439e-02],
# #          [-1.5983e-03],
# #          [ 1.6229e-02],
# #          [ 4.1590e-02],
# #          [ 1.0364e-02]],

# #         [[-2.1091e-02],
# #          [-9.2347e-03],
# #          [-5.2297e-02],
# #          [ 4.4615e-02],
# #          [-9.4159e-02],
# #          [-1.5704e-02],
# #          [-2.3436e-02],
# #          [-1.5658e-02],
# #          [ 5.6854e-03],
# #          [ 2.1636e-03],
# #          [-1.1955e-02],
# #          [ 4.7987e-02],
# #          [-3.6330e-02],
# #          [-1.1914e-02],
# #          [ 1.3425e-02],
# #          [ 6.7884e-02],
# #          [ 3.2761e-02],
# #          [ 1.2930e-01],
# #          [ 1.6531e-01],
# #          [ 1.5036e-01],
# #          [ 1.1677e-01],
# #          [ 1.0589e-01],
# #          [ 1.0806e-01],
# #          [ 1.1016e-01],
# #          [ 6.0572e-03],
# #          [-1.9446e-03],
# #          [-1.3157e-02],
# #          [-2.0127e-02],
# #          [-2.4983e-02],
# #          [ 3.2301e-02],
# #          [ 3.0798e-03],
# #          [ 3.1145e-02]],

# #         [[ 2.2664e-01],
# #          [ 2.7439e-01],
# #          [ 2.3156e-01],
# #          [ 2.5431e-01],
# #          [ 2.4080e-01],
# #          [ 3.1112e-01],
# #          [ 2.2962e-01],
# #          [ 2.5251e-01],
# #          [ 5.4946e-02],
# #          [ 3.9503e-02],
# #          [-1.1424e-02],
# #          [ 6.0527e-02],
# #          [ 4.1744e-02],
# #          [ 2.4263e-02],
# #          [ 3.0575e-02],
# #          [ 5.3822e-03],
# #          [ 1.8343e-03],
# #          [ 1.7837e-02],
# #          [-3.8876e-03],
# #          [ 3.7169e-02],
# #          [ 2.6356e-02],
# #          [-2.1941e-04],
# #          [ 1.8659e-02],
# #          [-1.5298e-02],
# #          [-3.7453e-02],
# #          [-1.8385e-02],
# #          [ 2.6284e-02],
# #          [ 2.8627e-02],
# #          [ 5.1342e-02],
# #          [ 3.7546e-02],
# #          [-3.8820e-02],
# #          [-3.5421e-02]]])

# y=torch.tensor([[[ 1.5783e-02,  2.4424e-01],
#          [ 1.9821e-02,  1.8176e-01],
#          [ 2.7856e-02,  2.3010e-01],
#          [-9.2034e-04,  2.8233e-01],
#          [-3.6025e-03,  2.3207e-01],
#          [-6.9797e-03,  2.6045e-01],
#          [-2.0469e-02,  2.7576e-01],
#          [-2.8884e-02,  2.6414e-01],
#          [ 5.0411e-02, -4.5928e-03],
#          [-1.4884e-02, -1.8473e-02],
#          [ 6.1474e-02, -1.9037e-02],
#          [ 3.3989e-02,  2.2582e-02],
#          [-5.7027e-02,  2.6753e-02],
#          [ 4.0928e-02, -3.5556e-03],
#          [-1.8824e-02,  6.6928e-02],
#          [ 7.9621e-03, -2.3484e-03],
#          [ 1.3607e-01,  9.3618e-02],
#          [ 1.2743e-01,  6.0612e-02],
#          [ 1.2939e-01,  1.3482e-01],
#          [ 1.1004e-01,  8.2865e-02],
#          [ 9.8736e-02,  1.4473e-01],
#          [ 1.4151e-01,  1.8817e-01],
#          [ 7.8946e-02,  1.1323e-01],
#          [ 1.0764e-01,  1.5617e-01],
#          [ 1.3130e-03,  4.9296e-02],
#          [ 1.3552e-02,  2.7309e-02],
#          [ 5.3512e-02, -3.7325e-02],
#          [ 6.6527e-02,  6.3476e-03],
#          [ 1.2614e-02,  1.0123e-02],
#          [ 3.3975e-03, -3.1942e-02],
#          [ 7.0590e-03,  1.1581e-02],
#          [ 5.1372e-02, -5.5979e-02]],

#         [[ 2.1399e-01,  1.4884e-01],
#          [ 2.2543e-01,  1.0428e-01],
#          [ 2.3988e-01,  5.4024e-02],
#          [ 2.5690e-01,  1.4969e-01],
#          [ 2.4114e-01,  7.8089e-02],
#          [ 2.8528e-01,  1.3338e-01],
#          [ 2.3508e-01,  1.2182e-01],
#          [ 2.2863e-01,  7.3263e-02],
#          [ 3.0138e-05,  5.4690e-02],
#          [ 3.0930e-02, -1.0286e-03],
#          [ 2.2251e-02,  3.9463e-02],
#          [ 4.9854e-02,  3.3130e-02],
#          [ 1.8231e-02,  2.6489e-02],
#          [ 1.7113e-03,  2.3819e-02],
#          [-1.0705e-02,  1.0893e-02],
#          [-9.1946e-03,  5.6605e-02],
#          [ 1.1660e-01,  2.5224e-02],
#          [ 1.2080e-01,  4.2549e-02],
#          [ 1.2683e-01,  6.7365e-02],
#          [ 1.1538e-01,  1.7845e-02],
#          [ 1.3180e-01,  9.4229e-03],
#          [ 1.3757e-01,  4.1367e-02],
#          [ 1.1424e-01,  6.1090e-03],
#          [ 1.4308e-01, -1.9370e-02],
#          [-1.9670e-02,  9.1457e-03],
#          [-5.9468e-03, -3.0661e-02],
#          [ 2.7815e-03, -5.7741e-02],
#          [ 2.5267e-02, -1.8868e-02],
#          [ 5.7022e-02,  4.3829e-02],
#          [-1.3883e-02, -5.5703e-02],
#          [ 6.3913e-03, -1.1827e-02],
#          [ 4.9965e-03, -5.6241e-02]]])


# def produce_basic_statistics(y, plag):
#     """
#     Fully vectorized computation of adaptation statistics.

#     Args:
#         y (torch.Tensor): Shape (n_subs, n_trials, n_voxels)

#     Returns:
#         Various computed statistics as tensors.
#     """
#     n_subs, n_trials, n_voxels = y.shape

#     if n_trials % 4 != 0:
#         raise ValueError("Assumes 4 conditions with equal trials")

#     # Condition indices
#     quarter = n_trials // 4
#     cond1_p1 = y[:, :quarter, :]  # First quarter
#     cond1_p2 = y[:, quarter:2*quarter, :]
#     cond2_p1 = y[:, 2*quarter:3*quarter, :]
#     cond2_p2 = y[:, 3*quarter:, :]

#     # Compute means across trials
#     cond1r1 = cond1_p1.mean(dim=1)
#     cond1r2 = cond1_p2.mean(dim=1)
#     cond2r1 = cond2_p1.mean(dim=1)
#     cond2r2 = cond2_p2.mean(dim=1)

#     avgAmp1r1 = cond1r1.mean(dim=1, keepdim=True)
#     avgAmp2r1 = cond2r1.mean(dim=1, keepdim=True)
#     avgAmp1r2 = cond1r2.mean(dim=1, keepdim=True)
#     avgAmp2r2 = cond2r2.mean(dim=1, keepdim=True)

#     avgR1 = (avgAmp1r1 + avgAmp2r1) / 2
#     avgR2 = (avgAmp1r2 + avgAmp2r2) / 2

#     # Compute sorting indices
#     cond1_combined = torch.cat([cond1_p1, cond1_p2], dim=1)
#     cond2_combined = torch.cat([cond2_p1, cond2_p2], dim=1)

#     sAmp1r = cond1_combined.mean(dim=1).sort(dim=1).values
#     sAmp2r = cond2_combined.mean(dim=1).sort(dim=1).values

#     sAmp1r1 = torch.gather(cond1r1, 1, cond1_combined.mean(dim=1).argsort(dim=1))
#     sAmp2r1 = torch.gather(cond2r1, 1, cond2_combined.mean(dim=1).argsort(dim=1))
#     sAmp1r2 = torch.gather(cond1r2, 1, cond1_combined.mean(dim=1).argsort(dim=1))
#     sAmp2r2 = torch.gather(cond2r2, 1, cond2_combined.mean(dim=1).argsort(dim=1))

#     sAmp = ((sAmp1r1 - sAmp1r2) + (sAmp2r1 - sAmp2r2)) / 2

#     # #Compute correlation matrices and means
#     # wtc1 = (torch.corrcoef(cond1_p1.permute(0, 2, 1)) + torch.corrcoef(cond2_p1.permute(0, 2, 1))).mean(dim=[1, 2], keepdim=True) / 2
#     # wtc2 = (torch.corrcoef(cond1_p2.permute(0, 2, 1)) + torch.corrcoef(cond2_p2.permute(0, 2, 1))).mean(dim=[1, 2], keepdim=True) / 2

#     # btc1 = (torch.corrcoef(torch.cat([cond1_p1, cond2_p1], dim=1).permute(0, 2, 1))).mean(dim=[1, 2], keepdim=True)
#     # btc2 = (torch.corrcoef(torch.cat([cond1_p2, cond2_p2], dim=1).permute(0, 2, 1))).mean(dim=[1, 2], keepdim=True)

#     def batch_corrcoef(x):
#         """Computes correlation coefficient per subject."""
#         mean_x = x.mean(dim=-1, keepdim=True)
#         x_centered = x - mean_x
#         cov_matrix = torch.bmm(x_centered, x_centered.transpose(-2, -1)) / (x.shape[-1] - 1)
#         stddev = torch.sqrt(torch.diagonal(cov_matrix, dim1=-2, dim2=-1)).unsqueeze(-1)
#         corr_matrix = cov_matrix / (stddev @ stddev.transpose(-2, -1))
#         return corr_matrix

#     # Compute within-task correlations
#     corr1 = batch_corrcoef(cond1_p1.permute(0, 2, 1))  # (n_subs, voxels, voxels)
#     corr2 = batch_corrcoef(cond2_p1.permute(0, 2, 1))  # (n_subs, voxels, voxels)
#     corr3 = batch_corrcoef(cond1_p2.permute(0, 2, 1))  # (n_subs, voxels, voxels)
#     corr4 = batch_corrcoef(cond2_p2.permute(0, 2, 1))  # (n_subs, voxels, voxels)

#     wtc1 = (corr1 + corr2).mean(dim=[1, 2], keepdim=True) / 2  # (n_subs, 1)
#     wtc2 = (corr3 + corr4).mean(dim=[1, 2], keepdim=True) / 2  # (n_subs, 1)

#     # Compute between-task correlations
#     combined1 = torch.cat([cond1_p1.permute(0, 2, 1), cond2_p1.permute(0, 2, 1)], dim=-1)  # (n_subs, voxels, trials * 2)
#     combined2 = torch.cat([cond1_p2.permute(0, 2, 1), cond2_p2.permute(0, 2, 1)], dim=-1)  # (n_subs, voxels, trials * 2)

#     btc_corr1 = batch_corrcoef(combined1)  # (n_subs, voxels, voxels)
#     btc_corr2 = batch_corrcoef(combined2)  # (n_subs, voxels, voxels)

#     btc1 = btc_corr1[:, :cond1_p1.shape[-1], cond1_p1.shape[-1]:].mean(dim=[1, 2], keepdim=True)  # (n_subs, 1)
#     btc2 = btc_corr2[:, :cond1_p2.shape[-1], cond1_p2.shape[-1]:].mean(dim=[1, 2], keepdim=True)  # (n_subs, 1)


#     svm_init = wtc1 - btc1
#     svm_rep = wtc2 - btc2

#     # Compute t-values
#     tval1 = (cond1_combined.mean(dim=1) - cond2_combined.mean(dim=1)) / (
#         (cond1_combined.var(dim=1) / cond1_combined.shape[1]) +
#         (cond2_combined.var(dim=1) / cond2_combined.shape[1])
#     ).sqrt()

#     tval2 = (cond2_combined.mean(dim=1) - cond1_combined.mean(dim=1)) / (
#         (cond2_combined.var(dim=1) / cond2_combined.shape[1]) +
#         (cond1_combined.var(dim=1) / cond1_combined.shape[1])
#     ).sqrt()

#     tval_sorted_ind1 = tval1.abs().argsort(dim=1)
#     tval_sorted_ind2 = tval2.abs().argsort(dim=1)

#     c1_sinit = torch.gather(cond1r1, 1, tval_sorted_ind1)
#     c1_srep = torch.gather(cond1r2, 1, tval_sorted_ind1)
#     c2_sinit = torch.gather(cond2r1, 1, tval_sorted_ind2)
#     c2_srep = torch.gather(cond2r2, 1, tval_sorted_ind2)

#     abs_init_trend = (c1_sinit + c2_sinit) / 2
#     abs_rep_trend = (c1_srep + c2_srep) / 2
#     abs_adaptation_trend = abs_init_trend - abs_rep_trend

#     # Bin the AMA and AMS trends
#     nBins = 6
#     percInds = torch.round(torch.arange(1, sAmp.shape[1] + 1, device=y.device) * (nBins - 1) / sAmp.shape[1]).long()
    
#     sc_trend = torch.stack([sAmp[:, percInds == i].mean(dim=1) for i in range(nBins)], dim=1)
#     abs_ad_trend = torch.stack([abs_adaptation_trend[:, percInds == i].mean(dim=1) for i in range(nBins)], dim=1)

#     return torch.column_stack((avgR1, avgR2)), torch.column_stack((svm_init, svm_rep)), torch.column_stack((wtc1, wtc2)), torch.column_stack((btc1, btc2)), sc_trend, abs_ad_trend

# print(produce_basic_statistics(y, 1))

def produce_confidence_interval(y, pflag):
    
    AM, CP, WC, BC, AMA, AMS = produce_basic_statistics(y, pflag)

    def compute_slope(data):
        L = data.shape[1]
        X = torch.vstack((torch.arange(1, L+1), torch.ones(L))).T
        pX = torch.linalg.pinv(X)
        return torch.matmul(pX, data.T)[0]
    
    slopes = {name: compute_slope(data) for name, data in zip(
        ['AM', 'WC', 'BC', 'CP', 'AMS', 'AMA'], [AM, WC, BC, CP, AMS, AMA]
    )}

    # t_results = {name: sp.stats.ttest_1samp(slope, 0) for name, slope in slopes.items()}
    slopes = [torch.mean(slopes[key]) for key in slopes]

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