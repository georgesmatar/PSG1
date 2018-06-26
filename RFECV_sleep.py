#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:11:30 2017

@author: aumont
"""

def ExecuteRFECV(samples,y,featureNames,clusters,clusterNames,clf,kFolds,
                 nSplits,standardization,removedInfo,permutation,nPermutation,
                 currentDateTime,resultDir,debug,verbose):    
    import pandas
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split as tts
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import RFECV
    
    from slpClass_toolbox import Standardize
    from slpClass_toolbox import Permute
    from slpClass_toolbox import ComputePermutationAvgDA
    from slpClass_toolbox import PlotPermHist
    from slpClass_toolbox import ApplyStandardization
    from slpClass_toolbox import plot_confusion_matrix
    
    rfecv=RFECV(estimator=clf,
                cv=StratifiedKFold(kFolds),
                scoring='accuracy',
                n_jobs=-1)

    # Create empty Pandas dataframe
    cvResults           = pandas.DataFrame()
    decodingAccuracy    = pandas.DataFrame()
    permResults         = pandas.DataFrame()
    avg_perm_DA = []
    # Execute feature selection for nbOfSplit times
    for it in list(range(nSplits)) :
        # Randomly create stratified train and test partitions (1/3 - 2/3)
        xTrain,xTest,yTrain,yTest = tts(samples,y['Cluster'],
                                        test_size=0.33,
                                        stratify=y['Cluster'])
        # Data z-score standardization
        xTrainSet,zPrm = Standardize(xTrain,yTrain,standardization,debug)
        
        # "accuracy" is proportional to the number of correct classifications
        if verbose:
            print('  Fiting for split #{}'.format(it))
        rfecv.fit(xTrainSet,yTrain)
    
        # Append the dataframe with the new cross-validation results.
        cvResults['cv_Scores_'+str(it)]          = rfecv.grid_scores_
        cvResults['cv_Features_Rank_'+str(it)]   = rfecv.ranking_
        
        if debug:
            print('cvResults for it %d' % it)
            print(cvResults)
        
        # Plot number of features VS. cross-validation scores
        fig_cv = plt.figure(dpi=300)
        plt.subplot(211)
        plt.title('Best performance = %.2f with %d features' % \
                  (max(rfecv.grid_scores_), rfecv.n_features_))
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross-validation score %")
        plt.plot(range(len(rfecv.grid_scores_)), rfecv.grid_scores_)
        
        # subplot selected features
        plt.subplot(212)
        plt.title('Features selection')
        plt.xlabel("Features")
        plt.xticks(range(len(rfecv.grid_scores_)),
                   featureNames,
                   rotation='vertical')
        plt.ylabel("Selection")
        plt.scatter(range(len(rfecv.grid_scores_)), rfecv.support_)
        plt.grid()
        plt.tight_layout()
        savedPlotName = resultDir+'RFECV'+'_CV_DA_'+clusters+'_'+str(it+1)+ \
                        '_'+str(nSplits)+'.png'
        plt.savefig(savedPlotName, bbox_inches='tight')
        plt.close(fig_cv)
        
        if verbose:
            print('\tComplete')    
        
# ********************************** TEST *************************************
        # standardize test set using trainset standardization parameters
        xTestSet = ApplyStandardization(xTest,zPrm)
        
        if verbose:        
            print('  Testing')        
        # use score() function to calculate DAs
        if debug:
            print('scores'+str(it))
            print(rfecv.score(xTestSet,yTest))
        decodingAccuracy['test_DA_'+str(it)] = [rfecv.score(xTestSet,yTest)]
        
        # plot confusion matrix
        y_pred = rfecv.predict(xTestSet)
        cm = confusion_matrix(yTest, y_pred)
        fig_CM = plt.figure(dpi=300)
        plot_confusion_matrix(cm, clusterNames, normalize=True, precision=2)
        savedPlotName = resultDir+'RFECV'+'_'+clusters+'_ConfusionMatrix_'+ \
                        str(it+1)+'_'+str(nSplits)+'.png'
        plt.savefig(savedPlotName, bbox_inches='tight')
        plt.close(fig_CM)
        
        if it == nSplits-1:
            print('\nTest Decoding accuracy')
            decodingAccuracy['test_Avg_DA']=decodingAccuracy.iloc[0][:].mean()
            for i in list(range(len(decodingAccuracy.iloc[0]))):
                print('\t'+str(decodingAccuracy.iloc[0].index[i])+'\t'+ \
                      str(decodingAccuracy.iloc[0][i]))
            
            #formating test results to save in excel file
            fTest = []
            for i in range(len(list(decodingAccuracy))-1):
                fTest.append(decodingAccuracy.iloc[0][i])
        
            testDA =pandas.DataFrame()
            testDA['test_DA_per_epoch'] = fTest
            tmp = pandas.DataFrame(data=[np.mean(testDA['test_DA_per_epoch'])],
                                         columns=['avg_test_DA'])
            
            testDA = pandas.concat([testDA,tmp],axis=1)      
            print('\tComplete\n')

        
# ****************************** Permutation **********************************
        if permutation:
            if verbose:
                print('  Permutting')
            # Create subset based on selected best features
            xTrain_rfecv = rfecv.transform(xTrainSet)
            xTest_rfecv = rfecv.transform(xTestSet)
            permResults['permutation_DA_'+str(it)] = Permute(clusters,
                                                             xTrain_rfecv,
                                                             xTest_rfecv,
                                                             yTrain, yTest,
                                                             nPermutation,
                                                             debug_flag=0)
            avg_perm_DA.append(np.mean(permResults['permutation_DA_'+str(it)]))
        
#            savedHistName = resultDir+'/Permutation_hist_'+str(it)+'.png'
#            PlotPermHist(permResults,testDA.iloc[0][1],
#                         currentDateTime,savedHistName)
    if permutation:
        # compute permutation DA average and keep results in a dataframe
        epochedPermDA = ComputePermutationAvgDA(avg_perm_DA)
        
        print('Average permutation DA per train epoch')
        for i in epochedPermDA['Avg_Permutation_DA_per_epoch']:
            print('\t'+str(i))
            
        print('\nAverage permutation DA : {}'.format(
                            epochedPermDA['Global_Permutation_DA'][0]))
        
        savedHistName = resultDir+'Average_Permutation_hist.png'
        PlotPermHist(permResults,
                     testDA.iloc[0][1],
                     currentDateTime,
                     savedHistName)
        # formating permutation results to save in excel file
        permResults = pandas.concat([permResults,epochedPermDA], axis=1)
        
#    else : # binomial law
#        from scipy.stats import binom 
#        q=0.001 # p value 
#        n=300   # nombre d'observation (sujets)
#        p=0.5   # probablité d'avoir un essai correctement
#        chance_level= binom.isf (q, n, p) / n

#    excelResults = pandas.concat([cvResults,testDA,permResults],axis=1)

# ************************ Select best of best features ***********************
    ranks = cvResults.iloc[:,1::2]
    if debug:
        print(ranks)

    bestFeatures = pandas.DataFrame()
    bestFeatures = ranks[(ranks == 1).all(1)].index.tolist()
    print('\nBest features :')
    tmp = []
    for i in bestFeatures:
        tmp.append(featureNames[i])
        print('\t'+featureNames[i])
    bestFeaturesNames = pandas.DataFrame(data=tmp,columns=['Best_Features'])
    
    
    # Calculate number of time every features is selected as best
    bestFeaturesHist = ranks[(ranks == 1)].sum(axis=1)
    bestFeaturesHist.rename('Best_Features_Hist') 

    # Build structure of histogram data to save in excel
    hist = pandas.DataFrame(data=featureNames,columns=['Features_Name'])
    hist['Occurence_Best'] = bestFeaturesHist
    nbSubject = pandas.DataFrame(data=[len(samples)],
                                 columns=['Number_Of_Subjects'])
    nbFeature = pandas.DataFrame(data=[samples.shape[1]],
                                       columns=['Number_Of_Features'])
    dataSize = pandas.concat([nbSubject,nbFeature],axis=1)

    # Get the best test DA and corresponding training set of features
    bestDA = testDA['test_DA_per_epoch'].max()
    bestDAepoch = testDA['test_DA_per_epoch'].idxmax()
    colName = 'cv_Features_Rank_'+str(bestDAepoch)
    bTrainFeat = cvResults[colName][(cvResults[colName] == 1)].index.tolist()
    tmp = []
    tmp.append(bestDA)
    for i in bTrainFeat:
        tmp.append(featureNames[i])
    bTrainFeatName = pandas.DataFrame(data=tmp,
                                      columns=['Best_Train_Features_Set'])
    
    # Build results structure to be save in excel file
    excelResults = pandas.concat([cvResults,
                                  testDA,
                                  permResults,
                                  hist,
                                  bestFeaturesNames,
                                  removedInfo,
                                  dataSize,
                                  bTrainFeatName],axis=1)
#    excelResults.to_excel(resultDir+'results_RFECV_'+currentDateTime+'.xlsx',
#                          sheet_name=xlSheetName)

    return excelResults