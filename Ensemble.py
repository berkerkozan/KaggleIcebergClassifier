import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import log_loss

ResultDataFrame = None #all combination result will be held here!..
GeneralMean = None #Mean of models is hold as columns
GeneralMedian = None #Median of models is hold as columns
targetLabel = None
AllDataConcatenatedColumns = None #All columns will be concatenated

input_folder = "../PredictionsForCVData/"
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
csv_files.sort()

funcList = [(pd.DataFrame.mean,pd.DataFrame.mean),(pd.DataFrame.mean,pd.DataFrame.median),(pd.DataFrame.median,pd.DataFrame.mean),
            (pd.DataFrame.median,pd.DataFrame.median)]

print("********")

for funcLevel1,funcLevel2 in funcList:

    print("")
    print("")
    print("1st {0}, 2nd {1}".format(funcLevel1.__name__,funcLevel2.__name__))
    dataFrame = None
    keepTheseFiles = []

    minLogLoss = 1000000000000

    for i, csv in enumerate(csv_files):

        df = pd.read_csv(os.path.join(input_folder, csv), index_col=0)

        print("Correlation of Data")
        print(df.corr())

        if targetLabel is None:
            targetLabel = df.iloc[:,0]
            ResultDataFrame = df.iloc[:,0]

        if dataFrame is None:
            AllDataConcatenatedColumns = df.iloc[:, 1:]
            dataFrame = funcLevel1(df.iloc[:, 1:], axis=1)

            minLogLoss = log_loss(targetLabel, dataFrame, eps=1e-15)
            print("Logloss:{0}, File:{1}".format(minLogLoss,csv))

            keepTheseFiles.append(csv)

            #Add each dataframe result to have a general mean median result!..
            GeneralMean = np.mean(df.iloc[:, 1:], axis=1)
            GeneralMedian = pd.DataFrame.median(df.iloc[:, 1:], axis=1)

        else:
            ## Add each model's mean or median to have a general mean median result!..
            GeneralMean = pd.concat([GeneralMean, np.mean(df.iloc[:, 1:], axis=1)],axis=1)
            GeneralMedian = pd.concat([GeneralMedian, pd.DataFrame.median(df.iloc[:, 1:], axis=1)],axis=1)
            ##
            ##Add all columns to AllData
            AllDataConcatenatedColumns = pd.concat([AllDataConcatenatedColumns, df.iloc[:, 1:]],axis=1)
            ###

            tempLogLoss = log_loss(targetLabel, funcLevel2(pd.concat([dataFrame, funcLevel1(df.iloc[:, 1:], axis=1)], axis=1), axis=1), eps=1e-15)
            if tempLogLoss < minLogLoss:
                print("UPDATED!  Ex Loss:{0}, New:{1}. PROCESSED: {2}. ".format(minLogLoss,tempLogLoss,csv))
                minLogLoss = tempLogLoss
                keepTheseFiles.append(csv)
                dataFrame = pd.concat([dataFrame, funcLevel1(df.iloc[:, 1:], axis=1)], axis=1)
            else:
                print("Ex Loss:{0}, Tested Loss with this file:{1}. {2} was processed. ".format(minLogLoss, tempLogLoss, csv))

    ResultDataFrame = pd.concat([ResultDataFrame, funcLevel2(dataFrame, axis=1)],axis=1)

    print("****************")


print("")
print("")

ResultDataFrame = pd.concat([ResultDataFrame, np.mean(GeneralMean,axis=1)],axis=1)
ResultDataFrame = pd.concat([ResultDataFrame, pd.DataFrame.median(GeneralMean,axis=1)],axis=1)
ResultDataFrame = pd.concat([ResultDataFrame, pd.DataFrame.median(GeneralMedian,axis=1)],axis=1)
ResultDataFrame = pd.concat([ResultDataFrame, np.mean(GeneralMedian,axis=1)],axis=1)
ResultDataFrame = pd.concat([ResultDataFrame, np.mean(AllDataConcatenatedColumns,axis=1)],axis=1)
ResultDataFrame = pd.concat([ResultDataFrame, pd.DataFrame.median(AllDataConcatenatedColumns,axis=1)],axis=1)


print("***********")
print("Mean of Each Model without discarded, also shows all the models, look at correlation!!!!")
print(GeneralMean.head(10))
print("Correlation")
print(GeneralMean.corr())
print("***********")


print("Median of Each Model without discarded, also shows all the models, look at correlation!!!!")
print(GeneralMedian.head(10))
print("Correlation")
print(GeneralMedian.corr())
print("***********")

print("ALL RESULT PREDICTIONS (8 combination!!), CORR IS VERY IMPORTANT HERE!!!!!!!!")
print (ResultDataFrame.head(10))
print("Correlation")
print(ResultDataFrame.corr())
print("***********")

names = ["MeanMean","MeanMedian","MedianMean", "MedianMedian", "MeanOfMeanOfAllModels","MedianOfMeanOfAllModels",
         "MedianOfMedianOfAllModels","MeanOfMedianOfAllModels","MeanOfAllDataConcatenatedColumns","MedianOfAllDataConcatenatedColumns"]
print("***********")
print("Logloss summary for 10 combination!!!!..")
for i in range(10):
    print("logloss {0} is {1}".format(names[i], log_loss(targetLabel,ResultDataFrame.iloc[:,i+1]   , eps=1e-15)))



print("***********")
print("logloss of Mean of combinations: {0}".format(log_loss(targetLabel, np.mean(ResultDataFrame.iloc[:,1:],axis=1)   , eps=1e-15)))

print("***********")
print("logloss of Median of combinations: {0}".format(log_loss(targetLabel, pd.DataFrame.median(ResultDataFrame.iloc[:,1:],axis=1)   , eps=1e-15)))
###
#GeneralMean and np.mean(AllDataConcatenatedColumns,axis=1) should be the same!!!..
###

GeneralMedian.insert(loc=0, column='is_iceberg', value=targetLabel)

ResultDataFrame.to_csv("EnsembleResultForCVData.csv", index=None)
GeneralMedian.to_csv("EnsembleResultGeneralMedianForCVData.csv", index=None)
GeneralMean.to_csv("EnsembleResultGeneralMeanForCVData.csv", index=None)
#data = pd.concat([data1, data2], axis=1)
#print ( data.head())

print ("Finito")