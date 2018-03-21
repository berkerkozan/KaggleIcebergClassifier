from random import shuffle
import numpy as np
import math
import torch
from torch.autograd import Variable
import gc
import copy
import torch.optim
import matplotlib.pyplot as plt
import data_utils as utils
import sys


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.BCEWithLogitsLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    # def getParameters(self, model):

    def train(self, model, train_loader, val_loader, num_epochs=10, logPerNumber=50, ifPrint=True):

        asd = None
        bestTrainLoss = 1000000000
        earlyStoppingLimit = 80
        ReduceLROnPlateauLimit = 20
        stopLimitForTrainLoss = 0.0075

        earlyStoppingTotalEpoch = 0
        self.valLossForBestValLoss = sys.float_info.max
        self.valAccForBestValAcc = -1.

        self._reset_histories()
        iter_per_epoch = len(train_loader)

        optim = self.optim(filter(lambda p: p.requires_grad, model.parameters()), **self.optim_args)
        scheduler = ReduceLROnPlateau(optim, mode='min', factor=1 / 3, patience=ReduceLROnPlateauLimit,
                                      verbose=True)

        if torch.cuda.is_available():
            model.cuda()

        # print('START TRAIN.')

        sigmoid = torch.nn.Sigmoid()

        for epoch in range(num_epochs):

            # self.adjust_learning_rate(optim, epoch)

            train_acc = None

            for i, (inputs, targets) in enumerate(train_loader):
                #plt.imshow(inputs[0][0].numpy().reshape(75,75), cmap='inferno')
                #plt.show()
                inputs, targets = Variable(inputs).float(), Variable(targets)
                if model.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                optim.zero_grad()
                outputs = model(inputs)
                loss = self.loss_func(outputs.view(-1), targets)
                loss.backward()

                optim.step()

                # Take the last full accuracy
                if (i == iter_per_epoch - 2):
                    preds = (sigmoid(outputs) > 0.5).float().view(-1)
                    train_acc = (preds == targets).float().mean().data.cpu().numpy()[0]
                    self.train_acc_history.append(train_acc)
                    self.train_loss_history.append(loss.data.cpu().numpy()[0])
                    scheduler.step(loss.data.cpu().numpy()[0])

            # preds = (sigmoid(outputs)>0.5).float().view(-1)

            # train_acc = (preds == targets).float().mean().data.cpu().numpy()[0]
            # self.train_acc_history.append(train_acc)

            # last_log_nth_losses = self.train_loss_history[-1:]
            # train_loss = np.mean(last_log_nth_losses)
            train_loss = self.train_loss_history[-1]

            trainLossIndicator = ""

            if bestTrainLoss > train_loss:
                bestTrainLoss = train_loss
                trainLossIndicator = "---"

            if train_loss <= stopLimitForTrainLoss:
                print("train loss is below ", str(stopLimitForTrainLoss), ", total epoch is: ", str(epoch))
                break

            lrCurrent = float((optim.param_groups)[0]['lr'])

            if lrCurrent <= 1e-6:
                print("lrCurrent is very low.. ", str(lrCurrent), ", total epoch is: ", str(epoch))
                break

            if (ifPrint and epoch % logPerNumber == 0 and epoch != 0):
                print('[Epoch %d/%d]    TRAIN acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                      num_epochs,
                                                                      train_acc,
                                                                      train_loss), trainLossIndicator)

            del inputs, outputs, targets, loss
            gc.collect()

            model.eval()

            totalElement = 0
            accuracyOfVal = 0
            lossOfVal = 0

            self.preds = np.array([])

            for inputs, targets in val_loader:
                # plt.imshow(inputs[0][0].numpy().reshape(75, 75), cmap='inferno')
                inputs, targets = Variable(inputs).float(), Variable(targets)
                if model.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                    self.loss_func.cuda()
                outputs = model.forward(inputs)
                loss = self.loss_func(outputs.view(-1), targets)
                # val_losses.append(loss.data.cpu().numpy())

                preds = (sigmoid(outputs) > 0.5).float().view(-1)

                # scores = np.mean((preds == targets).data.cpu().numpy())
                # val_scores.append(scores)

                totalElement += targets.size(0)
                accuracyOfVal += (preds == targets).float().sum().data.cpu().numpy()[0]
                lossOfVal += loss.data.cpu().numpy()[0] * targets.size(0)

                # self.preds.append(preds.data.cpu().numpy())
                self.preds = np.append(self.preds, preds.data.cpu().numpy())

            del inputs, outputs, targets, loss
            gc.collect()

            val_acc = accuracyOfVal / (totalElement * 1.0)
            val_loss = lossOfVal / (totalElement * 1.0)

            self.val_acc_history.append(val_acc)
            # self.val_loss_history.extend([val_loss] * (i+1))
            self.val_loss_history.append(val_loss)

            model.train()
            # val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)

            if (ifPrint and epoch % logPerNumber == 0 and epoch != 0):
                print('[Epoch %d/%d] VAL   acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   val_acc,
                                                                   val_loss))

            if val_acc > self.valAccForBestValAcc:
                '''paramList.clear()
                for param in model.parameters():
                    paramList.append(param)'''
                # asd = copy.deepcopy(model.state_dict())
                self.valAccForBestValAcc = val_acc
                self.valLossForBestValAcc = val_loss
                self.trainAccuracyForBestValAccuracy = train_acc
                self.trainLossForBestValAccuracy = train_loss
                self.epochForBestValAcc = epoch + 1
                # bestModel = copy.deepcopy(model)
                # bestModel = model
                gc.collect()
                # print('#Best Validation accuracy, iteration: ', str(i + epoch * iter_per_epoch))

                if (ifPrint and epoch % logPerNumber == 0 and epoch != 0):
                    print('Best val_acc: %f' % val_acc)
                # print('Best val_loss: %f' % val_loss)
                # print('Current train_acc: %f' % train_acc)
                # print('Current train_loss: %f' % train_loss)

            if val_loss < self.valLossForBestValLoss:
                earlyStoppingTotalEpoch = 0
                # paramList.clear()
                # for param in model.parameters():
                # paramList.append(param)
                asd = copy.deepcopy(model.state_dict())
                # asd = copy.deepcopy(model.state_dict())

                self.valAccForBestValLoss = val_acc
                self.valLossForBestValLoss = val_loss

                self.trainAccuracyForBestValLoss = train_acc
                self.trainLossForBestValLoss = train_loss
                self.epochForBestValLoss = epoch + 1
                # bestModel = copy.deepcopy(model)
                # bestModel = model
                # gc.collect()
                # print('#Best Validation Loss!!, iteration: ', str(i + epoch * iter_per_epoch))
                # print('Best val_acc: %f' % val_acc)
                if (ifPrint and epoch % logPerNumber == 0 and epoch != 0):
                    print('Best val_loss: %f' % val_loss)
                # print('Current train_acc: %f' % train_acc)
                # print('Current train_loss: %f' % train_loss)

                # targets, preds
                self.preds = preds
            else:
                earlyStoppingTotalEpoch += 1

            if earlyStoppingTotalEpoch == earlyStoppingLimit:
                print("Early stopped!!No increase in val loss.., total epoch is: ", str(epoch))
                break

            # if log_nth:
            if (ifPrint and epoch % logPerNumber == 0 and epoch != 0):
                print("----------------------------------------------")

        # model. = None

        # for item in paramList:
        #  model.parameters[item] = paramList.append(param)

        model.load_state_dict(asd)
        # model = bestModel
        # del bestModel
        # gc.collect()

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        # print("##############################################")
        '''print('Best val_acc: %f' % self.valAccForBestValAcc)
        print('Validation loss for best val_acc: %f' % self.valLossForBestValAcc)
        print('Train accuracy for best val_acc: %f' % self.trainAccuracyForBestValAccuracy)
        print('Train loss for best val_acc: %f' % self.trainLossForBestValAccuracy)
        print("epoch:", self.epochForBestValAcc)
        print("----------------------------------------------")'''
        # print('Best val_loss: %f' % self.valLossForBestValLoss)
        print(("Best Val acc: {0}, Val loss: {1}").format(self.valAccForBestValLoss, self.valLossForBestValLoss))
        '''print('Validation Acc for best val_loss: %f' % self.valAccForBestValLoss)
        print('Train accuracy for best val_loss: %f' % self.trainAccuracyForBestValLoss)
        print('Train loss for best val_loss: %f' % self.trainLossForBestValLoss)
        print("epoch:", self.epochForBestValLoss)'''
        # model = None

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.optim_args["lr"] * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class Predict():
    def __call__(self, model, bands, labels):
        model.eval()

        sigmoid = torch.nn.Sigmoid()
        totalElement = 0
        accuracyOfVal = 0
        lossOfVal = 0

        self.preds = np.array([])
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        datasetForTest = utils.getDatasetForTestData()(bands, labels)
        test_loader = torch.utils.data.DataLoader(datasetForTest, batch_size=2, shuffle=False, num_workers=4)

        for inputs, targets in test_loader:
            # plt.imshow(inputs[0][0].numpy().reshape(75, 75), cmap='inferno')
            inputs, targets = Variable(inputs).float(), Variable(targets)
            if model.is_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
                self.loss_func.cuda()
            outputs = model.forward(inputs)
            loss = self.loss_func(outputs.view(-1), targets)
            # val_losses.append(loss.data.cpu().numpy())

            preds = (sigmoid(outputs) > 0.5).float().view(-1)

            # scores = np.mean((preds == targets).data.cpu().numpy())
            # val_scores.append(scores)

            totalElement += targets.size(0)
            accuracyOfVal += (preds == targets).float().sum().data.cpu().numpy()[0]
            lossOfVal += loss.data.cpu().numpy()[0] * targets.size(0)

            # self.preds.append(preds.data.cpu().numpy())
            self.preds = np.append(self.preds, preds.data.cpu().numpy())

        del inputs, outputs, targets, loss
        gc.collect()

        val_acc = accuracyOfVal / (totalElement * 1.0)
        val_loss = lossOfVal / (totalElement * 1.0)
        # print("^^^^^^^^^^")
        print(format("Test acc: {0}, Test loss: {1}"), val_acc, val_loss)

        model.train()

        return val_acc, val_loss


class PredictForDataLoader():
    def __call__(self, model, loader):

        model.eval()

        sigmoid = torch.nn.Sigmoid()
        totalElement = 0
        accuracyOfVal = 0
        lossOfVal = 0

        self.preds = np.array([])
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        # datasetForTest = utils.getDatasetForTestData()(bands, labels)
        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1)

        predictionsLoss = np.array([], dtype="float32")

        for inputs, targets in loader:
            # plt.imshow(inputs[0][0].numpy().reshape(75, 75), cmap='inferno')
            inputs, targets = Variable(inputs).float(), Variable(targets)
            if model.is_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
                self.loss_func.cuda()
            outputs = model.forward(inputs)
            loss = self.loss_func(outputs.view(-1), targets)
            # val_losses.append(loss.data.cpu().numpy())

            preds = (sigmoid(outputs) > 0.5).float().view(-1)
            predictionsLoss = np.append(predictionsLoss, sigmoid(outputs).data.cpu().numpy())

            # scores = np.mean((preds == targets).data.cpu().numpy())
            # val_scores.append(scores)

            totalElement += targets.size(0)
            accuracyOfVal += (preds == targets).float().sum().data.cpu().numpy()[0]
            lossOfVal += loss.data.cpu().numpy()[0] * targets.size(0)

            # self.preds.append(preds.data.cpu().numpy())
            self.preds = np.append(self.preds, preds.data.cpu().numpy())

        del inputs, outputs, targets, loss
        gc.collect()

        val_acc = accuracyOfVal / (totalElement * 1.0)
        val_loss = lossOfVal / (totalElement * 1.0)
        # print("^^^^^^^^^^")
        print(("Test acc: {0}, Test loss: {1}").format(val_acc, val_loss))

        model.train()

        return predictionsLoss, val_acc, val_loss


class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. Default: 10.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
    Example:

    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('*******************************Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + mode + ' is unknown!')
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            self.is_better = lambda a, best: a < best * rel_epsilon
            self.mode_worse = float('Inf')
        elif mode == 'min' and threshold_mode == 'abs':
            self.is_better = lambda a, best: a < best - threshold
            self.mode_worse = float('Inf')
        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            self.is_better = lambda a, best: a > best * rel_epsilon
            self.mode_worse = -float('Inf')
        else:  # mode == 'max' and epsilon_mode == 'abs':
            self.is_better = lambda a, best: a > best + threshold
            self.mode_worse = -float('Inf')