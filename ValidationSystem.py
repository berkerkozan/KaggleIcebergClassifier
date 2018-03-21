import numpy as np
import torch
import gc
import data_utils as utils
import Solver
from sklearn.model_selection import StratifiedKFold
import time
import pandas as pd
from sklearn.metrics import log_loss
import Model
import matplotlib.pyplot as plt
from pathlib import Path
import json


class CVfor1Test(object):

    @staticmethod
    def CV(band1And2ForTrain, band1And2ForTest, labelForTrain, labelForTest, modelSelf, modelParameters, learningRate,
           weightDecay, CVorPrediction=True, stopCVforValidation=5, kFoldNumberForValidation=9, logPerNumber=50,
           channel3rd="average",
           numberOfEpochs=300, batchsizeTrain=128, batchsizeVal=64, batchsizeTest=64,inputBandNumber = 3,
           smoothInput = False, smooth3rdChannel=False, pixelNormalization=False):

        predictionsForTest = pd.DataFrame(data=labelForTest, columns=["is_iceberg"])

        testStackingMeanLoss = None
        testStackingLogMeanLoss = None
        testStackingMedianLoss = None
        testStackingMinMaxMeanLoss = None
        testStackingMinMaxMedianLoss = None

        isBreak = False

        # for validation data
        valLossForBestValLoss = []
        valAccForBestValLoss = []
        trainAccuracyForBestValLoss = []
        trainLossForBestValLoss = []
        epochForBestValLoss = []

        testAccuracy = []
        testLoss = []

        # For validation data!
        # indexForPandas = np.array([])
        # predictionsForPandas = np.array([])

        skf = StratifiedKFold(n_splits=kFoldNumberForValidation, shuffle=True, random_state=5)

        for i, (train_index, val_index) in enumerate(skf.split(band1And2ForTrain[0], labelForTrain)):
            train_time = time.time()
            print("##############################################")
            print("{0}th TRAIN FOLD BEGINS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(1 + i))

            train_data, val_data, test_data = utils.GetTrainValidationDataForKFoldCrossValidation2 \
                (train_index, val_index, band1And2ForTrain, labelForTrain, band1And2ForTest, labelForTest, channel3rd,
                 inputBandNumber = inputBandNumber,smoothInput = smoothInput, smooth3rdChannel=smooth3rdChannel,
                 pixelNormalization = pixelNormalization)

            print("Train size: %i" % len(train_data))
            print("Val size: %i" % len(val_data))
            print("Test size: %i" % len(test_data))

            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchsizeTrain, shuffle=True,
                                                       num_workers=4)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=batchsizeVal, shuffle=False, num_workers=4)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=batchsizeTest, shuffle=False, num_workers=4)

            solver = Solver.Solver(optim_args={"lr": learningRate, "weight_decay": weightDecay})
            model = modelSelf(**modelParameters)
            solver.train(model, train_loader, val_loader, num_epochs=numberOfEpochs, logPerNumber=logPerNumber)

            valLossForBestValLoss.append(solver.valLossForBestValLoss)
            valAccForBestValLoss.append(solver.valAccForBestValLoss)
            trainAccuracyForBestValLoss.append(solver.trainAccuracyForBestValLoss)
            trainLossForBestValLoss.append(solver.trainLossForBestValLoss)
            epochForBestValLoss.append(solver.epochForBestValLoss)

            predictions, testacc, testLossTemp = Solver.PredictForDataLoader()(model, test_loader)

            testAccuracy.append(testacc)
            testLoss.append(testLossTemp)

            columnName = "is_iceberg_" + str(i)
            predictionsForTest[columnName] = predictions

            print("--- %s seconds for this training" % (time.time() - train_time))

            if (CVorPrediction and valLossForBestValLoss[-1] > 0.38):
                print("-------BREAKKKKKKKK-------")
                print("VAL LOSS is > than 0.38..")
                isBreak = True
                break

            if (CVorPrediction and testLoss[-1] > 0.50):
                print("-------BREAKKKKKKKK-------")
                print("testLoss is > than 0.50..")
                isBreak = True
                break

            if (i == stopCVforValidation - 1):
                print("#####################################################")
                print("{0} TRAIN FOLD is processed, STATISTICS IS BELOW..pass to next test fold!..".format(
                    stopCVforValidation))
                break

        print(predictionsForTest.head(10))
        print(predictionsForTest.corr())

        print("")
        # print("valAccForBestValLoss: ", valAccForBestValLoss)
        # print("trainAccuracyForBestValLoss: ", trainAccuracyForBestValLoss)
        # print("trainLossForBestValLoss: ", trainLossForBestValLoss)
        print("epochForBestValLoss: ", epochForBestValLoss)
        print("testAccuracy: ", testAccuracy)
        print("valLossForBestValLoss: ", valLossForBestValLoss)
        print("testLoss: ", testLoss)
        print("")

        # print("Mean of valAccForBestValLoss: ", np.mean(valAccForBestValLoss))
        # print("Mean of trainAccuracyForBestValLoss: ", np.mean(trainAccuracyForBestValLoss))
        # print("Mean of trainLossForBestValLoss: ", np.mean(trainLossForBestValLoss))
        print("Mean of epochForBestValLoss: ", np.mean(epochForBestValLoss))
        print("Mean of testAccuracy: ", np.mean(testAccuracy))
        print("Mean of valLossForBestValLoss: ", np.mean(valLossForBestValLoss))
        print("Mean of testLoss: ", np.mean(testLoss))

        if CVorPrediction and np.mean(testLoss) > 0.30:
            print("-------BREAKKKKKKKK-------")
            print("Mean of test loss is > than 0.30..")
            isBreak = True

        if CVorPrediction:
            maxColumn = i + 2

            predictionsForTestCopy = predictionsForTest.copy()

            predictionsForTestCopy['is_iceberg_max'] = predictionsForTestCopy.iloc[:, 1:maxColumn].max(axis=1)
            predictionsForTestCopy['is_iceberg_min'] = predictionsForTestCopy.iloc[:, 1:maxColumn].min(axis=1)
            predictionsForTestCopy['is_iceberg_mean'] = predictionsForTestCopy.iloc[:, 1:maxColumn].mean(axis=1)
            predictionsForTestCopy['is_iceberg_median'] = predictionsForTestCopy.iloc[:, 1:maxColumn].median(axis=1)

            # set up cutoff threshold for lower and upper bounds, easy to twist
            cutoff_lo = 0.8
            cutoff_hi = 0.2

            print(" ")

            testStackingMeanLoss = predictionsForTestCopy.iloc[:, 1:maxColumn].mean(axis=1)
            testStackingMeanLoss = log_loss(predictionsForTestCopy[["is_iceberg"]], testStackingMeanLoss, eps=1e-15)

            testStackingLogMeanLoss = predictionsForTestCopy.iloc[:, 1:maxColumn].apply(np.log).mean(axis=1).apply(
                np.exp)
            testStackingLogMeanLoss = log_loss(predictionsForTestCopy[["is_iceberg"]], testStackingLogMeanLoss,
                                               eps=1e-15)

            testStackingMedianLoss = predictionsForTestCopy.iloc[:, 1:maxColumn].median(axis=1)
            testStackingMedianLoss = log_loss(predictionsForTestCopy[["is_iceberg"]], testStackingMedianLoss, eps=1e-15)

            testStackingMinMaxMeanLoss = np.where(
                np.all(predictionsForTestCopy.iloc[:, 1:maxColumn] > cutoff_lo, axis=1),
                predictionsForTestCopy['is_iceberg_max'],
                np.where(np.all(predictionsForTestCopy.iloc[:, 1:maxColumn] < cutoff_hi, axis=1),
                         predictionsForTestCopy['is_iceberg_min'], predictionsForTestCopy['is_iceberg_mean']))
            testStackingMinMaxMeanLoss = log_loss(predictionsForTestCopy[["is_iceberg"]], testStackingMinMaxMeanLoss,
                                                  eps=1e-15)

            testStackingMinMaxMedianLoss = np.where(
                np.all(predictionsForTestCopy.iloc[:, 1:maxColumn] > cutoff_lo, axis=1),
                predictionsForTestCopy['is_iceberg_max'],
                np.where(np.all(predictionsForTestCopy.iloc[:, 1:maxColumn] < cutoff_hi, axis=1),
                         predictionsForTestCopy['is_iceberg_min'], predictionsForTestCopy['is_iceberg_median']))
            testStackingMinMaxMedianLoss = log_loss(predictionsForTestCopy[["is_iceberg"]],
                                                    testStackingMinMaxMedianLoss, eps=1e-15)

            print("mean stacking error:", str(testStackingMeanLoss))

            print("Logmean stacking error:", str(testStackingLogMeanLoss))

            print("median stacking error:", str(testStackingMedianLoss))

            print("MinMax + Mean Stacking error:", str(testStackingMinMaxMeanLoss))

            print("MinMax + Median Stacking error:", str(testStackingMinMaxMedianLoss))

            del predictionsForTestCopy
            gc.collect()

        # predictionsForTest.drop(columns=['is_iceberg'])

        return predictionsForTest, valLossForBestValLoss, testLoss, testStackingMeanLoss, testStackingLogMeanLoss, \
               testStackingMedianLoss, testStackingMinMaxMeanLoss, testStackingMinMaxMedianLoss, isBreak


class CVforMultipleTests(object):

    @staticmethod
    def CV(band1ForAllData, band2ForAllData, labelForAllData, modelSelf, modelParameters,
           learningRate, weightDecay, CVorPrediction=True, kFoldNumberForTest=8, stopCVforTestData=4,
           kFoldNumberForValidation=7, stopCVforValidation=5, logPerNumber=50, channel3rd="",
           numberOfEpochs=300, passTestForNumberOfIteration=0, batchsizeTrain=128, batchsizeVal=64,
           batchsizeTest=64, inputBandNumber = 3,smoothInput = False, smooth3rdChannel = False,
           pixelNormalization=False):

        strParameters = ""
        for (i, j) in modelParameters.items():
            strParameters += str(i) + str(j) + "-"

        fileNameForJson = "../Backups/{0}-{1}-lr:{2}-decay:{3}-{4}-inputBandNo:{5}" \
                          "-smoothInput:{6}-" \
                          "smooth3rdChannel:{7}-pixel:{8}.json".format(

            modelSelf.__name__, strParameters, learningRate, weightDecay, channel3rd, inputBandNumber,
            smoothInput, smooth3rdChannel, pixelNormalization)


        tuning_time = time.time()
        skfForTestData = StratifiedKFold(n_splits=kFoldNumberForTest, shuffle=True, random_state=5)

        testStackingMeanLossList = []
        testStackingMedianLossList = []
        testStackingMinMaxMeanLossList = []
        testStackingMinMaxMedianLossList = []
        testStackingLogMeanLossList = []
        valLossForBestValLossMeanList = []
        testLossMeannList = []

        # testPredictions = []
        # testIndexes = []

        predictionsForSubmission = None




        # Save IN CASE!!!!!!!!!



        fileName = "../Backups/{0}-{1}-lr:{2}-decay:{3}-{4}-inputBandNo:{5}" \
                   "-smoothInput:{6}-" \
                   "smooth3rdChannel:{7}-pixel:{8}.csv".format(

            modelSelf.__name__, strParameters, learningRate, weightDecay, channel3rd, inputBandNumber,
            smoothInput, smooth3rdChannel, pixelNormalization)

        my_file = Path(fileName)

        if my_file.is_file():
            predictionsForSubmission = pd.read_csv(my_file)

        my_file = Path(fileNameForJson)

        if my_file.is_file():
            with open(fileNameForJson, 'r') as f:
                testStackingMeanLossList, testStackingLogMeanLossList,testStackingMedianLossList, \
                testStackingMinMaxMeanLossList,testStackingMinMaxMedianLossList,valLossForBestValLossMeanList, \
                testLossMeannList   = json.load(f)
            passTestForNumberOfIteration = len(testStackingMeanLossList)

        # Save IN CASE!!!!!!!!!




        print("")
        print("")
        print("")
        print("")
        print("---------------------------------------------------")
        print("---------------------------------------------------")
        print("Model is:{0},learningRate: {1},weight decay:{2}, model Params:{3}, channel3rd:{4}, smooth3rdChannel: {5}".format(
            modelSelf.__name__, learningRate, weightDecay, modelParameters, channel3rd, smooth3rdChannel))
        print("---------------------------------------------------")
        print("---------------------------------------------------")
        print("")
        isBreak = False

        for i, (trainingIndexForTraningTestSplit, testIndexForTraningTestSplit) \
                in enumerate(skfForTestData.split(band2ForAllData, labelForAllData)):
            test_time = time.time()

            print("")
            print("##############################################")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("{0}th TEST FOLD BEGINS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(1 + i))
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            if i < passTestForNumberOfIteration:
                continue

            labelForTest = labelForAllData[testIndexForTraningTestSplit]
            band1ForTest = band1ForAllData[testIndexForTraningTestSplit]
            band2ForTest = band2ForAllData[testIndexForTraningTestSplit]
            labelForTrain = labelForAllData[trainingIndexForTraningTestSplit]
            band1ForTraining = band1ForAllData[trainingIndexForTraningTestSplit]
            band2ForTraining = band2ForAllData[trainingIndexForTraningTestSplit]

            # Generate 8 pictures for each target and then take mean..
            # for i in len(band1ForTest):

            predictionsForTest, valLossForBestValLoss, testLoss, testStackingMeanLoss, testStackingLogMeanLoss, \
            testStackingMedianLoss, testStackingMinMaxMeanLoss, testStackingMinMaxMedianLoss, isBreak = \
                CVfor1Test.CV([band1ForTraining, band2ForTraining], [band1ForTest, band2ForTest], labelForTrain,
                      labelForTest,
                      modelSelf, modelParameters, learningRate, weightDecay, CVorPrediction=CVorPrediction,
                      stopCVforValidation=stopCVforValidation,
                      kFoldNumberForValidation=kFoldNumberForValidation, logPerNumber=logPerNumber,
                      channel3rd=channel3rd, numberOfEpochs=numberOfEpochs,
                      batchsizeTrain=batchsizeTrain, batchsizeVal=batchsizeVal, batchsizeTest=batchsizeTest,
                      inputBandNumber = inputBandNumber,smoothInput = smoothInput, smooth3rdChannel=smooth3rdChannel,
                      pixelNormalization=pixelNormalization)

            predictionsForTest.insert(loc=0, column='indexOfData', value=testIndexForTraningTestSplit)

            if predictionsForSubmission is None:
                predictionsForSubmission = predictionsForTest

            else:
                predictionsForSubmission = predictionsForSubmission.append(predictionsForTest)


            # testPredictions.append(predictionsForTest)
            # testIndexes.append(testIndexForTraningTestSplit)

            testStackingMeanLossList.append(testStackingMeanLoss)
            testStackingLogMeanLossList.append(testStackingLogMeanLoss)
            testStackingMedianLossList.append(testStackingMedianLoss)
            testStackingMinMaxMeanLossList.append(testStackingMinMaxMeanLoss)
            testStackingMinMaxMedianLossList.append(testStackingMinMaxMedianLoss)
            valLossForBestValLossMeanList.append(valLossForBestValLoss)
            testLossMeannList.append(testLoss)



            if isBreak:
                print("")
                print("-------BREAKKKKKKKK------- IN TEST TURN, Means numbers in CV is low..")
                print("")
                print("Model is:{0},learningRate: {1},weight decay:{2}, model Params:{3}, channel3rd:{4}, smooth3rdChannel: {5}".format(
                        modelSelf.__name__, learningRate, weightDecay, modelParameters, channel3rd, smooth3rdChannel))
                break

            if (i == 0 and np.mean(testStackingMeanLossList) > 0.20):
                print("")
                print("-------BREAKKKKKKKK------- testStackingMeanLossList IS > 0.20 for the first test fold!!!")
                print("")
                print("Model is:{0},learningRate: {1},weight decay:{2}, model Params:{3}, channel3rd:{4}, smooth3rdChannel: {5}".format(
                        modelSelf.__name__, learningRate, weightDecay, modelParameters, channel3rd, smooth3rdChannel))
                isBreak = True
                break

            if (np.mean(testStackingMeanLossList) > 0.27):
                print("")
                print("-------BREAKKKKKKKK------- testStackingMeanLossList IS > 0.27!!!")
                print("")
                print("Model is:{0},learningRate: {1},weight decay:{2}, model Params:{3}, channel3rd:{4}, smooth3rdChannel: {5}".format(
                        modelSelf.__name__, learningRate, weightDecay, modelParameters, channel3rd, smooth3rdChannel))
                isBreak = True
                break

            if (i == stopCVforTestData - 1):
                print("")
                print("##############################################")
                print("{0} TEST FOLD IS OVER, END OF THIS HYPERPARAMETERS!..".format(stopCVforTestData))
                print("##############################################")

                print("")
                print("Model is:{0},learningRate: {1},weight decay:{2}, model Params:{3}, channel3rd:{4} ".format(
                    modelSelf.__name__, learningRate, weightDecay, modelParameters, channel3rd))

                break

            #### Save IN CASE!!!!!!!!!

            fileName = "../Backups/{0}-{1}-lr:{2}-decay:{3}-{4}-inputBandNo:{5}" \
                       "-smoothInput:{6}-" \
                       "smooth3rdChannel:{7}-pixel:{8}.csv".format(

                modelSelf.__name__, strParameters, learningRate, weightDecay, channel3rd, inputBandNumber,
                smoothInput, smooth3rdChannel, pixelNormalization)

            predictionsForSubmission.to_csv(fileName, index=None)

            # Save IN CASE!!!!!!!!!


            with open(fileNameForJson, 'w') as outfile:
                json.dump((testStackingMeanLossList, testStackingLogMeanLossList,testStackingMedianLossList
                    ,testStackingMinMaxMeanLossList,testStackingMinMaxMedianLossList,valLossForBestValLossMeanList
                        ,testLossMeannList   ), outfile)


            ##### Save IN CASE!!!!!!!!!









        print("")
        print("testStackingMeanLoss: ", testStackingMeanLossList)
        print("testStackingLogMeanLoss: ", testStackingLogMeanLossList)
        print("testStackingMedianLoss: ", testStackingMedianLossList)
        print("testStackingMinMaxMeanLoss: ", testStackingMinMaxMeanLossList)
        print("testStackingMinMaxMedianLoss: ", testStackingMinMaxMedianLossList)
        print("valLossForBestValLossMean: ", valLossForBestValLossMeanList)
        print("testLossMeann: ", testLossMeannList)
        print("")
        print("testStackingMeanLoss mean: ", np.mean(testStackingMeanLossList))
        print("testStackingLogMeanLoss mean: ", np.mean(testStackingLogMeanLossList))
        print("testStackingMedianLoss mean: ", np.mean(testStackingMedianLossList))
        print("testStackingMinMaxMeanLoss mean: ", np.mean(testStackingMinMaxMeanLossList))
        print("testStackingMinMaxMedianLossList mean: ", np.mean(testStackingMinMaxMedianLossList))
        print("valLossForBestValLossMean mean: ", np.mean(np.concatenate(valLossForBestValLossMeanList)))
        print("testLossMeann mean: ", np.mean(np.concatenate(testLossMeannList)))

        print("")
        print("")

        #if not isBreak:
        if True:

            breakStr = ""
            if isBreak:
                breakStr = "BREAK!"

            strParameters = ""
            for (i, j) in modelParameters.items():
                strParameters += str(i) + str(j) + "-"

            fileName = "../Submission/{0:.4f}-{1:.4f}-{2:.4f}-{3}-{4}-lr:{5}-decay:{6}-{7}-inputBandNo:{8}" \
                       "-smoothInput:{9}-" \
                       "smooth3rdChannel:{10}-pixel:{11}-{12}.csv".format(
                np.mean(testStackingMeanLossList),
                np.mean(testStackingMedianLossList), np.mean(testLossMeannList),
                modelSelf.__name__, strParameters, learningRate, weightDecay, channel3rd,inputBandNumber,
                smoothInput,smooth3rdChannel,pixelNormalization,breakStr)

            # predictionsForSubmission = pd.DataFrame(
            #   {'is_iceberg': testPredictions.astype("float32")}, index=testIndexes)
            predictionsForSubmission.sort_values(by=['indexOfData'], inplace=True)
            predictionsForSubmission.to_csv(fileName, index=None)


if __name__ == '__main__':

    newband1 = []
    # Test for Target Augmentation
    test = pd.read_json('test2.json')
    band1 = utils.unflattenBand(test['band_1'])[2:3]
    for test in band1:
        newband1.append(test)
        # band1 = band1[0]
        plt.imshow(test)
        testRotated = utils.Rotate()(test, 90)
        newband1.append(testRotated)
        plt.imshow(testRotated)
        testFlipped = utils.HorizontalFlip()(test)
        newband1.append(testFlipped)
        plt.imshow(testFlipped)

    newband1 = np.array(newband1)
    plt.imshow(newband1[0])
    plt.imshow(newband1[1])
    plt.imshow(newband1[2])
    plt.imshow(newband1[3])
    plt.imshow(newband1[4])
    plt.imshow(newband1[5])
    plt.imshow(newband1[6])
    plt.imshow(newband1[7])

    print("qweqwe")