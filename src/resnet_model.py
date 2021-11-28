import time
import json
from torch.utils.data import DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import (models,
                         transforms)
import copy


class ResnetModel:

    def __init__(self, path_to_pretrained_model=None, map_location='cpu', num_classes=5):
        """
        Allows for training, evaluation, and prediction of ResNet Models

        params
        ---------------
        path_to_pretrained_model - string - relative path to pretrained model - default None
        map_location - string - device to put model on - default cpu
        num_classes - int - number of classes to put on the deheaded ResNet
        """
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
       
        self.model = torch.load(
            path_to_pretrained_model, map_location=self.device
        )

    def feed_forward(self, model, inputs):
        """
        Feeds inputs through model

        params
        ---------------
        model - torch model to feed inputs through
        inputs - inputs to feed through model

        returns
        ---------------
        output_tensor - torch tensor - tensor of final output layer after feed forward
        """
        output_tensor = model(inputs)
        return output_tensor

    def fit(self, train_loader, val_loader, num_epochs=10, criterion=None,
            optimizer=None, batch_size=16, early_stop_min_increase=0.003,
            early_stop_patience=10, lr=0.0001):
        """
        Impliments transfer learning based on new data

        params
        ---------------
        train_loader - torch DataLoader - Configured DataLoader for training, helpful when images are flowing from folder
        val_loader - torch DataLoader - Configured DataLoader for validation, helpful when images are flowing from folder
        num_epochs - int - number of epochs to use during training
        criterion - Loss function to assess model during training and evaluation - default None and CrossEntropyLoss
        optimizer - Optimizer algorithm - default Adam
        batch_size - int - Number of images to use per optimizer update - default 16
        early_stop_min_increase - float - Minimum increase in accuracy score to indicate model improvement per epoch - default 0.003
        early_stop_patience - int - Number of epochs to allow less than or equal to minimum improvement - default 10
        lr - float - Learning rate for optimization algorithm - default 0.0001

        returns
        ---------------
        model - trained torch model
        loss - list - history of loss across epochs
        acc - list - history of accuracy across epochs
        """
        # TODO: change data loader params to data params then in this function,
        # Transform those to data loaders so that the batch size can be dynamic
        start = time.time()
        model = self.model
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0
        train_loss_over_time = []
        val_loss_over_time = []
        train_acc_over_time = []
        val_acc_over_time = []
        epochs_no_improve = 0
        early_stop = False
        phases = ['train', 'val']

        if not optimizer:
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if not criterion:
            criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            print(f"Epoch number: {epoch + 1} / {num_epochs}")

            for phase in phases:
                if phase == 'train':
                    data_loader = train_loader
                    model.train()
                else:
                    data_loader = val_loader
                    model.eval()

                running_loss = 0
                running_corrects = 0

                for inputs, labels in data_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.feed_forward(model, inputs)
                        _, pred = torch.max(outputs, dim=1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(pred == labels.data)

                if phase == 'train':
                    epoch_loss = running_loss / len(train_loader.dataset)
                    train_loss_over_time.append(epoch_loss)
                    epoch_acc = running_corrects.double() / len(train_loader.dataset)
                    train_acc_over_time.append(epoch_acc)

                else:
                    epoch_loss = running_loss / len(val_loader.dataset)
                    val_loss_over_time.append(epoch_loss)
                    epoch_acc = running_corrects.double() / len(val_loader.dataset)
                    val_acc_over_time.append(epoch_acc)

                print(f"{phase} loss: {epoch_loss:.3f}, acc: {epoch_acc:.3f}")

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model, 'trained_model_resnet50_checkpoint.pt')
                    epochs_no_improve = 0

                elif phase == 'val' and (epoch_acc - best_acc) < early_stop_min_increase:
                    epochs_no_improve += 1
                    print(
                        f"Number of epochs without improvement has increased to {epochs_no_improve}")

            if epochs_no_improve >= early_stop_patience:
                early_stop = True
                print('Early stopping!')
                break

            print('-' * 60)
            total_time = (time.time() - start) / 60
            print(
                f"Training completed. Time taken: {total_time:.3f} min\nBest accuracy: {best_acc:.3f}")
            model.load_state_dict(best_model_wts)
            self.model = model
            loss = {'train': train_loss_over_time, 'val': val_loss_over_time}
            acc = {'train': train_acc_over_time, 'val': val_acc_over_time}

            return model, loss, acc

    def evaluate(self, test_loader, model=None, criterion=None):
        """
        Feeds set of images through model and evaluates relevant metrics
        as well as batch predicts. Prints loss and accuracy

        params
        ---------------
        test_loader - torch DataLoader - Configured DataLoader for evaluation, helpful when images flow from directory
        model - trained torch model - Model to use during evaluation - default None which retrieves model from attributes
        criterion - Loss function to assess model - Default None which equates to CrossEntropyLoss.

        returns
        ---------------
        preds - list - List of predictions to use for evaluation of non-included metrics
        labels_list - list - List of labels to use for evaluation of non-included metrics
        """
        if not model:
            model = self.model
        if not criterion:
            criterion = nn.CrossEntropyLoss()

        # model = self.model
        model.eval()
        test_loss = 0
        test_acc = 0
        preds = list()
        labels_list = list()

        for inputs, labels in test_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = self.feed_forward(model, inputs)
                loss = criterion(outputs, labels)
                _, pred = torch.max(outputs, dim=1)
                preds.append(pred.item())
                labels_list.append(labels.item())

            test_loss += loss.item() * inputs.size(0)
            correct = pred.eq(labels.data.view_as(pred))
            accuracy = torch.mean(correct.type(torch.FloatTensor))
            test_acc += accuracy.item() * inputs.size(0)

        test_loss = test_loss / len(test_loader.dataset)
        test_acc = test_acc / len(test_loader.dataset)

        print(f"Test loss: {test_loss:.4f}\nTest acc: {test_acc:.4f}")

        return preds, labels_list

    def predict_proba(self, img, k, index_to_class_labels, show=False):
        """
        Feeds single image through network and returns top k predicted labels and probabilities

        params
        ---------------
        img - PIL Image - Single image to feed through model
        k - int - Number of top predictions to return
        index_to_class_labels - dict - Dictionary to map indices to class labels
        show - bool - Whether or not to display the image before prediction - default False

        returns
        ---------------
        formatted_predictions - list - List of top k formatted predictions formatted to include a tuple of 1. predicted label, 2. predicted probability as str
        """
        if show:
            img.show()
        # img = self.test_transform(img)
        # img = img.unsqueeze(0)
        # img = img.to(self.device)
        self.model.eval()
        output_tensor = self.feed_forward(self.model, img)
        prob_tensor = torch.nn.Softmax(dim=1)(output_tensor)
        top_k = torch.topk(prob_tensor, k, dim=1)
        probabilites = top_k.values.detach().numpy().flatten()
        indices = top_k.indices.detach().numpy().flatten()
        formatted_predictions = []

        for pred_prob, pred_idx in zip(probabilites, indices):
            predicted_label = index_to_class_labels[pred_idx].title()
            predicted_perc = pred_prob * 100
            formatted_predictions.append(
                (predicted_label, f"{predicted_perc:.3f}%"))

        return formatted_predictions

    # change predict proba to actually be in line with sklearn's API and create another function that formats the raw probabilities.
    # Then go back in to the website, and change the code accordingly.
