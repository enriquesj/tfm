from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import glob
import time
import copy

# Cambio


if __name__ == '__main__':

    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    NO_VALUE = 1995

    def plot_curve(n_epoch, train, val, title, x_label, y_label, img_name):

        value_train = [h for h in train]
        value_val = [h for h in val]
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.plot(range(1, n_epoch + 1), value_train, label="Train")
        plt.plot(range(1, n_epoch + 1), value_val, label="Validation")
        plt.ylim(bottom=0, top=max([max(value_val), max(value_train)]))
        plt.xticks(np.arange(1, n_epoch + 1, 1.0))
        plt.legend()
        plt.savefig('data/PIROPO/'+img_name+'.png')
        plt.show()


    def pred_acc(original, predicted):
        matrics = []
        # tensor_ones = torch.ones_like(predicted)

        eq_vector = torch.eq(predicted, original)
        acc_classifier = eq_vector.sum().numpy() / len(original)  # Compare classifiers individually
        acc_detection = 0
        if acc_classifier == 1:  # Only 100% correct predictions
            acc_detection = 1

        true_ones = predicted.int() & original.int()    # Ones in the position where both are 1
        false_ones = (original.int() - true_ones) ^ predicted.int()
        number_ones = original.sum().numpy()  # The sum will tell the number of ones
        if number_ones != 0:
            recall = true_ones.sum().numpy() / number_ones
        else:
            recall = NO_VALUE        # If recall has this value, it should be eliminated for the mean

        precision = true_ones.sum().numpy() / (len(false_ones) + len(true_ones))

        matrics.append(acc_classifier)   # Classifier-level accuracy
        matrics.append(acc_detection)    # Grid-level accuracy
        matrics.append(recall)           # Recall
        matrics.append(precision)        # Precision
        return matrics


    def train_model(model, dataloaders, criterion, optimizer, num_epochs):
        since = time.time()

        loss_history = []
        accuracy_history = []
        val_acc_history = []
        val_acc2_history = []
        val_acc3_history = []
        val_acc4_history = []
        val_loss_history = []
        train_acc_history = []
        train_acc2_history = []
        train_acc3_history = []
        train_acc4_history = []
        train_loss_history = []

        output_layer = nn.Sigmoid()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0


        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            temp_values = []
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = []
                running_acc = []
                running_acc2 = []
                running_acc3 = []
                running_acc4 = []

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device, dtype=torch.float)
                    labels = torch.squeeze(labels, 1)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        output_labels = model(inputs)

                        loss = criterion(output_labels, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    acc = []
                    acc2 = []
                    acc3 = []
                    acc4 = []

                    output_labels = output_layer(output_labels) # Sigmoid to make output between 0 and 1
                    output_labels = torch.round(output_labels)
                    for i, pred in enumerate(output_labels, 0):
                        accuracies = pred_acc(torch.Tensor.cpu(labels[i]), torch.Tensor.cpu(pred))
                        acc.append(accuracies[0])
                        acc2.append(accuracies[1])
                        acc3.append(accuracies[2])
                        acc4.append(accuracies[3])

                    running_loss.append(loss.item())
                    running_acc.append(np.asarray(acc).mean())
                    running_acc2.append(np.asarray(acc2).mean())
                    running_acc3.append(np.asarray(acc3[acc3 != NO_VALUE]).mean()) # Remove the recall value which has no sense
                    running_acc4.append(np.asarray(acc4).mean())

                # Loss and accuracy changes between train and val in each epoch
                epoch_loss = np.asarray(running_loss).mean()  # changes between train and val
                epoch_acc = np.asarray(running_acc).mean()  # changes between train and val
                epoch_acc2 = np.asarray(running_acc2).mean()  # changes between train and val
                epoch_acc3 = np.asarray(running_acc3).mean()  # changes between train and val
                epoch_acc4 = np.asarray(running_acc4).mean()  # changes between train and val

                print('{} Loss: {:.4f} Acc: {:.4f} Tot_Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_acc2))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)
                    val_acc2_history.append(epoch_acc2)
                    val_acc3_history.append(epoch_acc3)
                    val_acc4_history.append(epoch_acc4)
                    val_loss_history.append(epoch_loss)
                if phase == 'train':
                    train_acc_history.append(epoch_acc)
                    train_acc2_history.append(epoch_acc2)
                    train_acc3_history.append(epoch_acc3)
                    train_acc4_history.append(epoch_acc4)
                    train_loss_history.append(epoch_loss)

            if epoch > 0:   # At least two values to plot
                plt.close()
                plot_curve(epoch+1, train_loss_history, val_loss_history,
                           'loss vs epochs', 'loss', 'epochs', 'loss vs epochs')
                plot_curve(epoch+1, train_acc_history, val_acc_history,
                           'classifier-level accuracy vs epochs', 'classifier-level accuracy', 'epochs', 'classifier-level accuracy vs epochs')
                plot_curve(epoch+1, train_acc2_history, val_acc2_history,
                           'grid-level accuracy vs epochs', 'grid-level accuracy',
                            'epochs', 'grid-level accuracy vs epochs')
                plot_curve(epoch+1, train_acc3_history, val_acc3_history,
                           'recall for ones vs epochs', 'recall for ones', 'epochs', 'recall for ones vs epochs')
                plot_curve(epoch+1, train_acc4_history, val_acc4_history,
                           'precision for ones vs epochs', 'precision for ones', 'epochs', 'precision for ones vs epochs')

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc_tot: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        torch.save(best_model_wts, 'best_wts.pth')

        # Store the loss and accuracy values for train and validation phases
        loss_history.append(train_loss_history)
        loss_history.append(val_loss_history)
        accuracy_history.append(train_acc_history)
        accuracy_history.append(val_acc_history)
        accuracy_history.append(train_acc2_history)
        accuracy_history.append(val_acc2_history)
        accuracy_history.append(train_acc3_history)
        accuracy_history.append(val_acc3_history)
        accuracy_history.append(train_acc4_history)
        accuracy_history.append(val_acc4_history)

        return model, loss_history, accuracy_history


    def test_model(model, dir_test, transform):
        # Test the model
        model_ft.load_state_dict(torch.load('best_wts.pth'))
        model.eval()
        output_layer = nn.Sigmoid()

        output = []
        output2 = []
        output3 = []
        # Prepare data and test model
        img_names = [f for f in glob.glob(dir_test + "**/*.jpg", recursive=True)]
        with torch.no_grad():
            for ima in img_names:
                image = Image.open(ima)
                image = torch.unsqueeze(transform(image), 0)
                image = image.to(device)

                output_label = model(image)
                output_label = output_layer(output_label)   # We apply sigmoid to obtain a value between 0 and 1
                output_label = torch.Tensor.cpu(torch.squeeze(output_label))

                output.append(output_label.tolist())

                output2.append(output_label.round().int().tolist())

                tensor_zeros = torch.zeros([1, len(output_label)], dtype=torch.float32)
                output3.append(output_label.where(output_label >= 0.2, tensor_zeros).tolist())

        df = pd.DataFrame(output)
        df.to_csv('data/PIROPO/test.csv', index=False, header=False)

        df2 = pd.DataFrame(output2)
        df2.to_csv('data/PIROPO/test2.csv', index=False, header=False)

        df3 = pd.DataFrame(output3)
        df3.to_csv('data/PIROPO/test3.csv', index=False, header=False)

        return output


    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


    def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement.
        model_ft = None

        if model_name == "alexnet":
            model_ft = models.alexnet(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_features = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_features, num_classes)

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft


    class OmniCamDataset(Dataset):
        def __init__(self, csv_files, root_dirs, transform=None):
            """
            Args:
                csv_files (list of strings): Paths to the csv file with annotations.
                root_dirs (list of strings): Directories with all the images.
                transform (callable, optional): Optional transform to be applied
                    on a sample.
            """

            self.activations_info = pd.read_csv(csv_files[0], header=None)
            if len(csv_files) == 1:  # Just one csv to read
                self.activations_info = pd.read_csv(csv_files[0], header=None)
            elif len(csv_files) > 1:
                for f in csv_files[1:]:
                    csv_info = pd.read_csv(f, header=None)
                    self.activations_info = pd.concat([self.activations_info, csv_info], ignore_index=True)

            print(self.activations_info)

            self.img_names = [f for f in glob.glob(root_dirs[0] + "**/*.jpg", recursive=True)]
            if len(root_dirs) == 1:  # Just one dir to read
                self.img_names = [f for f in glob.glob(root_dirs[0] + "**/*.jpg", recursive=True)]
            elif len(root_dirs) > 1:
                for file in root_dirs[1:]:
                    image_list = [f for f in glob.glob(file + "**/*.jpg", recursive=True)]
                    self.img_names = self.img_names + image_list

            self.transform = transform

        def __len__(self):
            return len(self.activations_info)

        def return_number_classifiers(self):
            return len(self.activations_info.columns)

        def weights_active_classifiers(self):
            n_zeros = (self.activations_info == 0).sum(axis=0)  # Per column
            n_ones = (self.activations_info != 0).sum(axis=0)   # Per column
            n_ones = n_ones.where(n_ones != 0, n_zeros)     # To make the division result one instead of NaN
            return torch.from_numpy(np.array(n_zeros/n_ones))

        def global_weight(self):
            n_zeros = (self.activations_info == 0).sum(axis=0).sum()    # In total
            n_ones = (self.activations_info != 0).sum(axis=0).sum()     # In total
            return n_zeros/n_ones

        def load_image(self, idx):
            image = Image.open(self.img_names[idx])
            return image

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            image = self.load_image(idx)
            label = self.activations_info.iloc[idx, :]
            label[label == 255] = 1
            label = np.array([label])

            if self.transform is not None:
                image = self.transform(image)

            return image, label


    ## Variables for initialization
    # Directories
    dir_train = ['data/PIROPO/converted_omni2A_training']
    csv_file_train = ['data/PIROPO/activations_train2A.csv']

    dir_val = ['data/PIROPO/converted_omni2A_test1']
    csv_file_val = ['data/PIROPO/activations_test1_2A.csv']

    dir_test = 'data/PIROPO/converted_omni2A_test2'

    # Model
    model_name = "alexnet"

    # Number of classifiers per image
    num_classifiers = 825   # Will be overwritten later

    # Batch size for training (change depending on how much memory you have)
    batch_size = 256

    # Number of epochs to train for
    num_epochs = 200

    # Input size
    input_size = 224

    # Type of weighting used: 1 is weight per classifier, 2 is global weighting
    weight_type = 1

    # Learning rate (default 0.001)
    learning_rate = 0.001

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True

    ## Datasets ##
    print("Initializing Datasets and Dataloaders...")
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create training and validation datasets
    omni_datasets = {'train': OmniCamDataset(csv_files=csv_file_train,
                                             root_dirs=dir_train,
                                             transform=data_transforms['train']),
                     'val': OmniCamDataset(csv_files=csv_file_val,
                                           root_dirs=dir_val,
                                           transform=data_transforms['val'])}

    # Create training dataloaders
    dataloaders = {'train': DataLoader(omni_datasets['train'], batch_size=batch_size,
                                       shuffle=True, num_workers=0),
                   'val': DataLoader(omni_datasets['val'], batch_size=batch_size,
                                     shuffle=True, num_workers=0)}

    print('Initialization done')

    # We overwrite the number of classes to match with the .csv
    num_classifiers = omni_datasets['train'].return_number_classifiers()
    print("Number of classifiers: ")
    print(num_classifiers)

    ## Model ##
    # Initialize the model for this run
    model_ft = initialize_model(model_name, num_classifiers, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    print("Model initialized:")
    print(model_ft)

    ## Device ##
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Detected device: ')
    print(device)

    # Send the model to GPU
    model_ft = model_ft.to(device)

    ## Feature extraction ##
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters where requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn: ")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    ## Optimizer
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)


    ## Loss ##
    if weight_type == 1:    # Weights per classifier
        pos_weight = omni_datasets['train'].weights_active_classifiers()

    elif weight_type == 2:   # Global weights (calculated for all the classifiers)
        pos_weight = torch.empty(num_classifiers).fill_(omni_datasets['train'].global_weight())

    pos_weight = pos_weight.to(device, dtype=torch.float)

    print('pos_weight: ')
    print(pos_weight.size())
    print(pos_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)   # Sigmoid is not in the model anymore


    ## Train and evaluate ##
    model_ft, loss_hist, acc_hist = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs)


    ## Test model ##
    output = test_model(model_ft, dir_test, transform=data_transforms['val'])


    ## Plot ##
    title = 'Loss vs. Number of Training Epochs'
    plot_curve(num_epochs+1, loss_hist[0], loss_hist[1], title,
               x_label='Training Epochs', y_label='Loss', img_name='loss')

    title = 'Classifier-level Accuracy vs. Number of Training Epochs'
    plot_curve(num_epochs+1, acc_hist[0], acc_hist[1], title,
               x_label='Training Epochs', y_label='Classifier-level Accuracy', img_name='classifier_acc')

    title = 'Grid-level Accuracy vs. Number of Training Epochs'
    plot_curve(num_epochs+1, acc_hist[2], acc_hist[3], title,
               x_label='Training Epochs', y_label='Grid-level Accuracy', img_name='grid_acc')

    title = 'Recall vs. Number of Training Epochs'
    plot_curve(num_epochs+1, acc_hist[4], acc_hist[5], title,
               x_label='Training Epochs', y_label='Recall', img_name='recall')

    title = 'Precision vs. Number of Training Epochs'
    plot_curve(num_epochs+1, acc_hist[6], acc_hist[7], title,
               x_label='Training Epochs', y_label='Precision', img_name='precision')

