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
    wtc1 = torch.zeros((n_subs, 1), dtype=torch.float32, requires_grad=True)
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


    n = y.shape[1]
    v = y.shape[2]
    cond1_p1 = y[:,:n // 4, :v]
    cond1_p2 = y[:, n // 4:n // 2, :v]
    cond2_p1 = y[:, n // 2:3 * n // 4, :v]
    cond2_p2 = y[:, 3 * n // 4:, :v]
    
    """This is the previous code. I now want to vectorise this by setting 
        cond1_p1_corr to calculate for all sub at once. The calculate_AM
        function below shows how the AM looks after being vectorised"""
    for sub in range(y.shape[0]):
        cond1_p1_corr = (torch.corrcoef(cond1_p1.T) + torch.corrcoef(cond2_p1.T)) / 2
        wtc1[sub, :] = torch.mean(torch.mean(cond1_p1_corr, axis=0))
    
    

    # cond1_p2_corr = (torch.corrcoef(cond1_p2.T)+torch.corrcoef(cond2_p2.T)) / 2
    # wtc2[sub, :] = torch.mean(torch.mean(cond1_p2_corr, axis=0))
    # WC = calculate_WC(y)
    BC = calculate_BC(y)
    CP = calculate_CP(WC, BC)
    AMS = calculate_AMS(y)
    AMA = calculate_AMA(y)
    

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
    print(cond1_p1)
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
    subs = y.shape[0]
    wtc1 = torch.rand(subs, 1, requires_grad=True)
    wtc2 = torch.rand(subs, 1, requires_grad=True)
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

    




# def produce_basic_statistics(y, plag):
#     n_subs = len(y)
#     nBins = 6
    
#     # Create empty lists to store intermediate results
#     all_wtc1 = []
#     all_wtc2 = []
#     all_btc1 = []
#     all_btc2 = []
#     all_svm_init = []
#     all_svm_rep = []
#     all_sc_trend = [[] for _ in range(n_subs)]
#     all_abs_ad_trend = [[] for _ in range(n_subs)]
#     all_avgAmp1r1 = []
#     all_avgAmp2r1 = []
#     all_avgAmp1r2 = []
#     all_avgAmp2r2 = []
    
#     # Debug prints
#     print("Input requires_grad:", y[0].requires_grad)
    
#     for sub in range(n_subs):
#         pattern = y[sub]
#         v = pattern.shape[1] # voxels
#         n = pattern.shape[0] # trials

#         if n % 4 != 0:
#             raise ValueError("Assumes 4 conditions with equal trials")
        
#         # Create views without breaking the graph
#         cond1_p = [
#             pattern[:n // 4, :v],
#             pattern[n // 4:n // 2, :v],
#         ]
#         cond2_p = [
#             pattern[n // 2:3 * n // 4, :v],
#             pattern[3 * n // 4:, :v],
#         ]
        
#         # Check if inputs maintain gradients
#         print(f"Sub {sub} - cond1_p[0] requires_grad:", cond1_p[0].requires_grad)

#         # Compute means - use preserving operations
#         cond1r1 = torch.mean(cond1_p[0], dim=0)
#         cond2r1 = torch.mean(cond2_p[0], dim=0)
#         cond1r2 = torch.mean(cond1_p[1], dim=0)
#         cond2r2 = torch.mean(cond2_p[1], dim=0)
        
#         # Track intermediate gradients
#         print(f"cond1r1 requires_grad:", cond1r1.requires_grad)
        
#         # Calculate amp values and append to lists
#         avg1r1 = torch.mean(cond1r1, dim=0, keepdim=True)
#         avg2r1 = torch.mean(cond2r1, dim=0, keepdim=True)
#         avg1r2 = torch.mean(cond1r2, dim=0, keepdim=True)
#         avg2r2 = torch.mean(cond2r2, dim=0, keepdim=True)
        
#         all_avgAmp1r1.append(avg1r1)
#         all_avgAmp2r1.append(avg2r1)
#         all_avgAmp1r2.append(avg1r2)
#         all_avgAmp2r2.append(avg2r2)
        
#         # Track amp gradients
#         print(f"avg1r1 requires_grad:", avg1r1.requires_grad)

#         # AMA adaptation by amplitude
#         mean_cond1 = torch.mean(torch.hstack([cond1_p[0].T, cond1_p[1].T]), dim=1)
#         mean_cond2 = torch.mean(torch.hstack([cond2_p[0].T, cond2_p[1].T]), dim=1)
        
#         ind1 = torch.argsort(mean_cond1)
#         ind2 = torch.argsort(mean_cond2)
        
#         # Take care to preserve gradients when indexing
#         sAmp1r1 = cond1r1[ind1]
#         sAmp2r1 = cond2r1[ind2]
#         sAmp1r2 = cond1r2[ind1]
#         sAmp2r2 = cond2r2[ind2]
        
#         # Compute slope - this should maintain gradients
#         sAmp = ((sAmp1r1 - sAmp1r2) + (sAmp2r1 - sAmp2r2)) / 2
#         print(f"sAmp requires_grad:", sAmp.requires_grad)
        
#         # Calculate correlations
#         cond1_p1_corr = (torch.corrcoef(cond1_p[0].T) + torch.corrcoef(cond2_p[0].T)) / 2
#         wtc1_val = torch.mean(torch.mean(cond1_p1_corr, dim=0, keepdim=True))
#         all_wtc1.append(wtc1_val.view(1, 1))
        
#         cond1_p2_corr = (torch.corrcoef(cond1_p[1].T) + torch.corrcoef(cond2_p[1].T)) / 2
#         wtc2_val = torch.mean(torch.mean(cond1_p2_corr, dim=0, keepdim=True))
#         all_wtc2.append(wtc2_val.view(1, 1))
        
#         # Explicitly create new tensors to avoid in-place operations
#         pp1 = torch.corrcoef(torch.cat([cond1_p[0].T, cond2_p[0].T], dim=0))
#         pp11 = (cond1_p[0].T).shape[1]
#         pp1_sub = pp1[:pp11, pp11:]
#         btc1_val = torch.mean(torch.mean(pp1_sub, dim=0, keepdim=True))
#         all_btc1.append(btc1_val.view(1, 1))
        
#         pp2 = torch.corrcoef(torch.cat([cond1_p[1].T, cond2_p[1].T], dim=0))
#         pp22 = (cond1_p[1].T).shape[1]
#         pp2_sub = pp2[:pp22, pp22:]
#         btc2_val = torch.mean(torch.mean(pp2_sub, dim=0, keepdim=True))
#         all_btc2.append(btc2_val.view(1, 1))
        
#         # Calculate SVM values
#         svm_init_val = wtc1_val - btc1_val
#         svm_rep_val = wtc2_val - btc2_val
#         all_svm_init.append(svm_init_val.view(1, 1))
#         all_svm_rep.append(svm_rep_val.view(1, 1))
        
#         # Check correlation gradients
#         print(f"wtc1_val requires_grad:", wtc1_val.requires_grad)
#         print(f"btc1_val requires_grad:", btc1_val.requires_grad)
#         print(f"svm_init_val requires_grad:", svm_init_val.requires_grad)
        
#         # T-tests - make sure these don't break gradients
#         cond1_combined = torch.vstack([cond1_p[0], cond1_p[1]])
#         cond2_combined = torch.vstack([cond2_p[0], cond2_p[1]])
#         tval1 = pytorch_ttest(cond1_combined, cond2_combined)
#         tval2 = pytorch_ttest(cond2_combined, cond1_combined)
        
#         # Check if t-test output has gradients
#         # If your pytorch_ttest function breaks gradients, this is a problem point
#         if isinstance(tval1, torch.Tensor):
#             print(f"tval1 requires_grad:", tval1.requires_grad)
        
#         # Use detach if sorting breaks gradients, then reattach through indexing
#         tval1_tensor = torch.tensor(tval1, dtype=torch.float32)
#         tval2_tensor = torch.tensor(tval2, dtype=torch.float32)
        
#         tval_sorted_ind1 = torch.argsort(torch.abs(tval1_tensor))
#         tval_sorted_ind2 = torch.argsort(torch.abs(tval2_tensor))
        
#         # Compute means
#         c1_init = torch.mean(cond1_p[0], dim=0)
#         c1_rep = torch.mean(cond1_p[1], dim=0)
#         c2_init = torch.mean(cond2_p[0], dim=0)
#         c2_rep = torch.mean(cond2_p[1], dim=0)
        
#         # Create new tensors when indexing to avoid gradient issues
#         c1_sinit = c1_init[tval_sorted_ind1]
#         c1_srep = c1_rep[tval_sorted_ind1]
#         c2_sinit = c2_init[tval_sorted_ind2]
#         c2_srep = c2_rep[tval_sorted_ind2]
        
#         # Calculate trends
#         abs_init_trend = (c1_sinit + c2_sinit) / 2
#         abs_rep_trend = (c1_srep + c2_srep) / 2
#         abs_adaptation_trend = abs_init_trend - abs_rep_trend
        
#         # Binning
#         AA = sAmp
#         AS = abs_adaptation_trend
        
#         # Compute percentage indices (use float tensor to avoid int issues)
#         indices = torch.arange(1, float(len(AA) + 1), dtype=torch.float32)
#         percInds = (torch.round((indices * (nBins - 1)) / len(AA)) / (nBins - 1)) * (nBins - 1)
        
#         # Create bins one by one
#         for i in range(nBins):
#             # Create mask and select values
#             mask = (percInds == float(i))
#             if torch.any(mask):
#                 sc_val = torch.mean(AA[mask])
#                 abs_ad_val = torch.mean(AS[mask])
#             else:
#                 # Handle empty bins
#                 sc_val = torch.tensor(0.0, requires_grad=True)
#                 abs_ad_val = torch.tensor(0.0, requires_grad=True)
                
#             # Add to our lists
#             all_sc_trend[sub].append(sc_val)
#             all_abs_ad_trend[sub].append(abs_ad_val)
    
#     # Now construct the final tensors carefully
#     # Stack avgAmp values
#     avgAmp1r1 = torch.cat(all_avgAmp1r1, dim=0)
#     avgAmp2r1 = torch.cat(all_avgAmp2r1, dim=0)
#     avgAmp1r2 = torch.cat(all_avgAmp1r2, dim=0)
#     avgAmp2r2 = torch.cat(all_avgAmp2r2, dim=0)
    
#     # Calculate averages
#     avgR1 = (avgAmp1r1 + avgAmp2r1) / 2
#     avgR2 = (avgAmp1r2 + avgAmp2r2) / 2
    
#     # Stack other values
#     wtc1 = torch.cat(all_wtc1, dim=0)
#     wtc2 = torch.cat(all_wtc2, dim=0)
#     btc1 = torch.cat(all_btc1, dim=0)
#     btc2 = torch.cat(all_btc2, dim=0)
#     svm_init = torch.cat(all_svm_init, dim=0)
#     svm_rep = torch.cat(all_svm_rep, dim=0)
    
#     # Handle trend tensors
#     sc_trend_tensors = []
#     abs_ad_trend_tensors = []
    
#     for sub in range(n_subs):
#         sc_sub = torch.stack(all_sc_trend[sub]).view(1, nBins)
#         abs_ad_sub = torch.stack(all_abs_ad_trend[sub]).view(1, nBins)
#         sc_trend_tensors.append(sc_sub)
#         abs_ad_trend_tensors.append(abs_ad_sub)
    
#     sc_trend = torch.cat(sc_trend_tensors, dim=0)
#     abs_ad_trend = torch.cat(abs_ad_trend_tensors, dim=0)
    
#     # Verify gradients before returning
#     print("Final tensors require grad:")
#     print("wtc1:", wtc1.requires_grad)
#     print("svm_init:", svm_init.requires_grad)
#     print("sc_trend:", sc_trend.requires_grad)
    
#     # Return the stacked tensors
#     return torch.column_stack((avgR1, avgR2)), torch.column_stack((svm_init, svm_rep)), torch.column_stack((wtc1, wtc2)), torch.column_stack((btc1, btc2)), sc_trend, abs_ad_trend

def produce_confidence_interval(y, pflag):
    
    AM, CP, WC, BC, AMA, AMS = produce_basic_statistics(y, pflag)

    print("AM")
    print(AM)
    print("CP")
    print(CP)
    print("WC")
    print(WC)
    print("BC")
    print(BC)
    print("AMA")
    print(AMA)
    print("AMS")
    print(AMS)

    def compute_slope(data):
        L = data.shape[1]
        X = torch.vstack((torch.arange(1, L+1), torch.ones(L))).T
        pX = torch.linalg.pinv(X)
        return torch.matmul(pX, data.T)[0]

    slopes = torch.stack([torch.mean(compute_slope(data)) for data in [AM, WC, BC, CP, AMS, AMA]
                          ])
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