#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 12:56:33 2017

@author: Tomy Aumont
"""
def ExecuteSFFS(x,y,featureNames,featureList,clusters,clusterNames,svc,
                kFolds,nbOfSplit,featMaxNbrSFFS,standardizationType,
                removedData,permutation_flag,nbPermutation,balance_flag,
                currentDateTime,resultDir,debug_flag,verbose):
    import scipy
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split as tts
    from sklearn.metrics import confusion_matrix
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
    from sklearn.model_selection import RandomizedSearchCV

    from slpClass_toolbox import BalanceClasses
    from slpClass_toolbox import Standardize
    from slpClass_toolbox import Permute
    from slpClass_toolbox import ComputePermutationAvgDA
    from slpClass_toolbox import PlotPermHist
    from slpClass_toolbox import ApplyStandardization
    from slpClass_toolbox import plot_confusion_matrix

    plt.rcParams.update({'figure.max_open_warning': 0})

    # Get features values since SFFS works only with numpy array!
    bestFeaturesHist = np.zeros([len(featureNames)])
    CvResult         = pd.DataFrame()
    permResults      = pd.DataFrame()
    tmpBest          = []
    DA               = []
    avg_perm_DA      = []
    skipFS           = False # flag to skip feature selection
    fitFeatOverTresh = False # fit classifier with most frequent features in best set

    #********************** TRAIN pre-procesing *******************************
    for it in list(range(nbOfSplit)):
        print('\nSplit #{}'.format(str(it)))

        # Use all features or given ones only
        if len(featureList) == 0:
            xx = x
        elif isinstance(featureList[0], float):
            xx = x
            fitFeatOverTresh = True
        else:
            xx = x[featureList]
            skipFS = True

        # Balance the number of old woman and old man or not
        if balance_flag:
            X, Y = BalanceClasses(xx,y)
        else:
            X, Y = xx, y

        # slpit dataset into train and test random subset
        X_train,X_test,y_train,y_test = tts(X,Y['Cluster'],
                                            test_size=0.33,
                                            stratify=Y['Cluster'])
        # Data z-score standardisation
        xTrainSet,zPrm = Standardize(X_train,y_train,standardizationType,
                                     debug_flag)

        #**************************** SVM optimisation ************************
        params_dict={'C': scipy.stats.expon(scale=100),
                     'kernel': ['linear'],
                     'class_weight': ['balanced', None]}

        n_iter_search = 20
        random_search = RandomizedSearchCV(svc,param_distributions=params_dict,
                                           n_iter=n_iter_search)

        random_search.fit(xTrainSet,y_train)
        optimClf = random_search.best_estimator_

        #*************************** TRAIN ************************************
        print('Fitting...')
        if skipFS:
            optimClf = optimClf.fit(xTrainSet.as_matrix(), y_train)

            yPred = optimClf.predict(xTrainSet.as_matrix())

            # Compute the accuracy of the test prediction
            acc = float((y_train == yPred).sum()) / yPred.shape[0]
            print('Train predicted accuracy: %.2f %%' % (acc * 100))
            fitRes = pd.DataFrame(data=[acc], columns=['CV_DA_'+str(it+1)])

        else:
            # set k_features = (1,X.shape[1]) to test all possible combinations
            sffs = SFS(optimClf,k_features=(1,featMaxNbrSFFS),forward=True,
                       floating=False,scoring='accuracy',cv=kFolds,n_jobs=-1)
            sffs = sffs.fit(xTrainSet.as_matrix(), y_train)

            print('Best combination for fit #%d (ACC: %.3f): %s' % \
                  (it,sffs.k_score_, sffs.k_feature_idx_))

            # Fit the estimator using the new feature subset and make a
            # prediction on the test data
            X_train_sfs = sffs.transform(xTrainSet.as_matrix())
            optimClf.fit(X_train_sfs, y_train)

            fitRes = pd.DataFrame.from_dict(sffs.get_metric_dict()).T
            fitRes['avg_over_std'] = fitRes['avg_score'] / fitRes['std_dev']

            if featMaxNbrSFFS > 1:
                # plot feature selection process metrics
                fig1=plot_sfs(sffs.get_metric_dict(), kind='std_err');
                savedPlotName = resultDir+'Decoding_accuracy_'+clusters+'_'+\
                                str(it)+'_'+str(nbOfSplit)+'.png'

                tmpBest.append(sffs.k_feature_idx_)
                bestFeaturesHist[[tmpBest[-1]]]+=1

                fig1.set_dpi(300)
                plt.tight_layout()
                plt.savefig(savedPlotName, bbox_inches='tight')
                plt.clf()
                plt.close(fig1)

                # plot mean / std
                plt.figure(dpi=300)
                plt.title('Moyenne sur ecart-type')
                plt.xlabel("nb attributs dans combinaison")
                plt.xticks(range(featMaxNbrSFFS))
                plt.ylabel("Moyenne sur ecart-type")
                plt.plot(list(range(1,featMaxNbrSFFS+1)),fitRes['avg_over_std'])
                figName = resultDir+'SFFS_'+clusters+'_bestSet_metric_'+ \
                          str(it)+'_'+str(nbOfSplit)
                plt.savefig(figName, bbox_inches='tight')
                plt.clf()
                plt.close()

        # add metrics iteration identifier
        fitRes = fitRes.add_suffix('_'+str(it+1))

        CvResult = pd.concat([CvResult, fitRes], axis=1)

        #***************************** TEST ***********************************
        print('Testing...')
        # standardize test set using trainset standardization parameters
        xTestSet = ApplyStandardization(X_test,zPrm)

        # prepare test data
        if skipFS:
            xTest = xTestSet
            savedPlotName = resultDir+clusters+'_ConfusionMatrix_'+str(it+1)+ \
                            '_'+str(nbOfSplit)
        else:
            # Generate a new subset of data according to selected features
            xTest = sffs.transform(xTestSet.as_matrix())
            savedPlotName = resultDir+'SFFS_'+clusters+'_ConfusionMatrix_'+ \
                        str(it+1)+'_'+str(nbOfSplit)

        # actually test classifier and compute decoding accuracy on predictions
        y_pred = optimClf.predict(xTest)
        acc = float((y_test == y_pred).sum()) / y_pred.shape[0]
        print('Test set accuracy: %.2f %%' % (acc * 100))
        DA.append(acc) # stack test DA for further use

        # plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_CM = plt.figure(dpi=300)
        plot_confusion_matrix(cm, clusterNames,title=savedPlotName,
                              normalize=True, precision=2)
        plt.clf()
        plt.close(fig_CM)

        #**************** STATISTICAL ASSESSMENT (PERMUTATION) ****************
        if permutation_flag:
            permResults['permutation_DA_'+str(it)]=Permute(clusters,
                                                           xTrainSet,
                                                           xTestSet,
                                                           y_train, y_test,
                                                           nbPermutation,
                                                           standardizationType,
                                                           debug_flag=0)
            avg_perm_DA.append(np.mean(permResults['permutation_DA_'+
                                                   str(it)]))

    dfDA     = pd.DataFrame(data=DA, columns=['DA_test'])
#    CvResult = pd.concat([CvResult, dfDA[:]], axis=1)
    CvResult = pd.concat([CvResult,
                          dfDA[:],
                          pd.DataFrame(data=[np.mean(DA)],
                          columns=['avg_DA'])],
                          axis=1)

    #***************** COMPUTE STATISTICAL ASSESSMENT RESULTS *****************
    if permutation_flag:
        # compute permutation DA average and keep results in a dataframe
        print('\nAverage permutation DA')
        for i in list(range(len(avg_perm_DA))):
            print('\t'+str(avg_perm_DA[i]))

        savedHistName = resultDir+'Average_Permutation_hist_'+clusters+'.png'
        PlotPermHist(permResults,
                     CvResult['avg_DA'].iloc[0],
                     currentDateTime,
                     savedHistName)
        #formating permutation results to save in excel file
        permResults = pd.concat([permResults,
                                 ComputePermutationAvgDA(avg_perm_DA)],
                                 axis=1)
        print('Mean permutation decoding accuracy : {}'.format(
                np.mean(permResults['Avg_Permutation_DA_per_epoch'])))
    else : # binomial law
        from scipy.stats import binom
        q = 0.001 # p value
        n = X.shape[0] + 1 # nombre d'observation (sujets)
        p = 1/len(clusterNames)   # probablité d'avoir un essai correctement
        luckLvl=pd.DataFrame(date=[binom.isf(q,n,p)/n],columns=['Chance_Level'])

#****************************** Compute results *******************************
    if not skipFS:
        # Build structure of histogram data to save in excel
        hist = pd.DataFrame(data=featureNames, columns=['Features_Name'])
        hist['Occurence_Best'] = bestFeaturesHist
        # Search best set across every iteration best set
        best_Combination = tmpBest[np.argmax(DA)]
        # Compute average size of best combination
        l=0
        for n in list(range(len(tmpBest))):
            l += len(tmpBest[n])
        avgBestCombSize = pd.DataFrame(data=[np.ceil(l/len(tmpBest))],
                                           columns=['avgBestCombSize'])

#    subsetHist = GetSubsetOccurence(tmpBest)
#    PlotHist(subsetHist[1],'Subsets occurences',subsetHist[0],'Comb_Hist.png')

        # Get best set's feature names
        tmp=[]
        tmp.append(np.max(DA))
        for i in best_Combination:
            tmp.append(featureNames[i])
            print('\t'+featureNames[i])
        bestFeatNames = pd.DataFrame(data=tmp, columns=['Best_Features_Set'])

        sffsRes = pd.concat([hist, bestFeatNames, avgBestCombSize],axis=1)

        # Plot best combination custom metric (mean / std_dev)
        from slpClass_toolbox import PlotBestCombinationMetrics
        filteredData = CvResult.filter(regex=r'avg_over_std_', axis=1)
        metrics = pd.DataFrame(data=filteredData)
        metrics.dropna(inplace=True)
        figName = resultDir+'SFFS_'+clusters+'_bestSet_metric_aggreg.png'
        PlotBestCombinationMetrics(metrics,figName)

    #save training and permutation results in an excel file
    nbSubject = pd.DataFrame(data=[len(X)], columns=['Number_Of_Subjects'])

    #************************ Build results structure *************************
    excelResults = pd.concat([CvResult,
                              permResults if permutation_flag else luckLvl,
                              sffsRes if not skipFS else None,
                              removedData,
                              nbSubject], axis=1)

    print('Mean Decoding accuracy :{}'.format(np.mean(DA)))

    # compute occurence of every subset in bestsets of every iteration
#    from slpClass_toolbox import GetSubsetOccurence
#    subsetHist = GetSubsetOccurence(tmpBest)
#    excelResults = pd.concat([excelResults, subsetHist], axis=1)
#    excelResults.to_excel(saveTo, sheet_name=xlSheetName)

    if fitFeatOverTresh:
        tresh = featureList[0] * nbOfSplit
        bestFeatColumns = hist.iloc[:,0][hist.iloc[:,1]>tresh]
        bestDataSet = xx[bestFeatColumns]
        classes = y
        DABestFeat = []
        print('Fitting with features occuring over %d times in best sets'%tresh)
        for i in list(range(nbOfSplit)):
            print('\rFit #{} of {}\n'.format(i+1,nbOfSplit),end='\r',flush=True)
            # Balance the number of old woman and old man or not
            if balance_flag:
                XX, YY = BalanceClasses(bestDataSet,classes)
            else:
                XX, YY = bestDataSet, classes

            # slpit dataset into train and test random subset
            XXtrain,XXtest,yytrain,yytest=tts(XX,YY['Cluster'],test_size=0.33,
                                                stratify=YY['Cluster'])
            # Data z-score standardisation
            xxTrainSet,zzPrm = Standardize(XXtrain,yytrain,standardizationType,
                                         debug_flag)

            # fit and predict on training data
            optimClf = optimClf.fit(xxTrainSet.as_matrix(), yytrain)
            yPred = optimClf.predict(xxTrainSet.as_matrix())
            # Compute accuracy of prediction on trainnnig set
            acc = float((yytrain == yPred).sum()) / yPred.shape[0]
            print('Train predicted accuracy: %.2f %%' % (acc * 100))
            fitRes = pd.DataFrame(data=[acc], columns=['CV_DA_'+str(it+1)])

            # test classifier and compute decoding accuracy on predictions
            xxTestSet = ApplyStandardization(XXtest,zzPrm)
            yypred = optimClf.predict(xxTestSet)
            acc = float((yytest == yypred).sum()) / yypred.shape[0]
            print('Test set accuracy: %.2f %%' % (acc * 100))
            DABestFeat.append(acc) # stack test DA for further use
            # plot confusion matrix
            cm = confusion_matrix(yytest, yypred)
            fig_CM = plt.figure(dpi=300)
            plot_confusion_matrix(cm, clusterNames,title=savedPlotName,
                                  normalize=True, precision=2)
            plt.clf()
            plt.close(fig_CM)
        df = pd.DataFrame(data=DABestFeat,columns=['optim DA'])
        df = pd.concat([df,pd.DataFrame(data=[np.mean(DABestFeat)],
                                        columns=['optim avg DA'])], axis=1)
        print('Classifier trained with best features (occ > %d) only' % tresh)
        print(df)
        excelResults = pd.concat([excelResults,df],axis=1)

    return excelResults
