# -- IMPORTS
import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
import json
#from workspace_utils import active_session
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import timeit
import argparse

# -- INITIAL PROGRAM FUNCTIONS
# python ImageClassifier/train.py ImageClassifier/flowers -s cp_test.pth -a vgg16 -lr 0.001 -hu 512 --gpu
parser = argparse.ArgumentParser(description='Train the Image Classifier on Flowers')
parser.add_argument('data_dir', default='flowers', help='Set input data directory')
parser.add_argument('-s', '--save_dir', default='ImageClassifier/checkpoint_py.pth', help='Set output filename')
parser.add_argument('-a', '--arch', default='vgg16', help='Set pretrained model')
parser.add_argument('-lr', '--learning_rate', default=0.001, help='Set optimizer learning rate')
parser.add_argument('-hu', '--hidden_units', default=512, help='Set nr of hidden model units')
parser.add_argument('--gpu', action='store_true', help='Turn on gpu')
args = parser.parse_args()

if args.arch == 'vgg16':
    model_arch = args.arch
else:
    print(f'The current model only allows {args.arch} architecture')
    model_arch = 'vgg16'
    print(f'Model set to {model_arch}')

if args.gpu == True:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

print(f'GPU on? {args.gpu} and runs on: {device}')
print(f'Learning rate: {args.learning_rate}')
print(f'Nr of hidden units: {args.hidden_units}')

# -- LOAD DATA
print('Loading data..')
data_dir = args.data_dir
print(f'Chose data folder: {data_dir}')
train_dir = data_dir + '/train'
tune_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

print(f'Chose to save the file: {args.save_dir}')

# Define the transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(p=0.25),
                                      transforms.RandomVerticalFlip(p=0.25),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
tune_data = datasets.ImageFolder(tune_dir, transform=test_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
batch_size = 32
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
tuneloader = torch.utils.data.DataLoader(tune_data, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Labels for classification
with open('ImageClassifier/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# -- BUILD THE MODEL
print('Building model..')
# Run on GPU if possible
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the pre-built and pre-trained model
model = models.vgg16(pretrained=True)

# Freeze the parameters for the pre-trained model, so we dont backprop through them
for param in model.parameters():
    param.requires_grad = False

# Define model
nr_hidden_units = args.hidden_units # default 512

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, int(nr_hidden_units))),
    ('relu1', nn.ReLU()),
    ('do1', nn.Dropout(0.2)),
    ('fc2', nn.Linear(int(nr_hidden_units), 102)),
    ('output', nn.LogSoftmax(dim=1))
     ]))

# Define the model to be trained
model.classifier = classifier # the model pre-trained model VGG doesnt have .fc but .classifier

# Define loss function
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
lr = args.learning_rate
optimizer = optim.Adam(model.classifier.parameters(), lr=float(lr)) # model trained on 0.001

# Send model to device
model.to(device)

# -- TRAIN THE MODEL
print('Training model..')
start = timeit.timeit()

# For long running code keep the session active
#with active_session():

# Def of variables
epochs = 5 # nr of trainings
steps = 0
running_loss = 0
print_every = 5
run_accuracy = 0
train_losses, tune_losses = [], []
train_accuracy, tune_accuracy = [], []

# Training
for epoch in range(epochs): # nr of traning times

    for inputs, labels in trainloader: # loop through the data

        steps += 1 # train steps
        #print(inputs.size())
        # Move input and label tensors to the default device (GPU if available)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the gradients - very important
        optimizer.zero_grad()

        # Forward pass -> log probabilities
        logps = model.forward(inputs)

        # Define loss
        loss = criterion(logps, labels)

        # Backward pass
        loss.backward()

        # Take a step towards lower loss
        optimizer.step()

        # Keep track on the total loss
        running_loss += loss.item()

        # calculate the accuracy
        ps = torch.exp(logps) # get the actual probability
        top_p, top_class = ps.topk(1, dim=1) # top probabilities and classes

        equals = top_class == labels.view(*top_class.shape) # check how many images which have the correct classification.
        # That *top_class.shape is passing all of the items in the top_class.shape into the view function call as separate arguments, without us even needing to know how many arguments are in the list. For example if the top_class has a shape say (32, 1), so *top_class.shape will pass 32, 1 to view function. Now this is as good as passing (top_class.shape[0],top_class.shape[1]). labels.view(top_class.shape[0],top_class.shape[1]) is equal to labels.view(*top_class.shape). The labels.view(*top_class.shape) is very useful because we dont have to know the exact shape for us to reshape them, we just use * and it will pack everything into a list and pass on to the view function for reshaping

        run_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        # after getting the top_class of the top_probability from torch.exp(output), we equal the top_class with the images labels (targets) to check if they match or not. The result from this equality is binary values [0,1]. Does this mean that 1 refers to class-label matching and 0 refers to class-label mismatching? Yes

        # Tune/evaluate the model every print_every 5 times - OBS makes the model overfit! Move back and tune after each epoch and remove print every.
        if steps % print_every == 0:
            tune_loss = 0
            accuracy = 0
            model.eval() # set model in evaluation mode

            with torch.no_grad(): # reduces memory usage - we dont need to calculate the gradients in evaluation mode

                for images, labels in tuneloader:

                    # Move images and labels tensors to the default device (GPU if available)
                    images = images.to(device)
                    labels = labels.to(device)

                    # Calculate the loss
                    log_ps = model(images) # log of probability
                    loss = criterion(log_ps, labels)
                    tune_loss += loss.item()

                    # calculate the accuracy
                    ps = torch.exp(log_ps) # get the actual probability
                    top_p, top_class = ps.topk(1, dim=1) # top probabilities and classes

                    equals = top_class == labels.view(*top_class.shape) # check how many images which have the correct classification.
                    # That *top_class.shape is passing all of the items in the top_class.shape into the view function call as separate arguments, without us even needing to know how many arguments are in the list. For example if the top_class has a shape say (32, 1), so *top_class.shape will pass 32, 1 to view function. Now this is as good as passing (top_class.shape[0],top_class.shape[1]). labels.view(top_class.shape[0],top_class.shape[1]) is equal to labels.view(*top_class.shape). The labels.view(*top_class.shape) is very useful because we dont have to know the exact shape for us to reshape them, we just use * and it will pack everything into a list and pass on to the view function for reshaping

                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    # after getting the top_class of the top_probability from torch.exp(output), we equal the top_class with the images labels (targets) to check if they match or not. The result from this equality is binary values [0,1]. Does this mean that 1 refers to class-label matching and 0 refers to class-label mismatching? Yes

            # View model results
            train_losses.append(running_loss/len(trainloader))
            tune_losses.append(tune_loss/len(tuneloader))
            train_accuracy.append(run_accuracy/len(trainloader))
            tune_accuracy.append(accuracy/len(tuneloader))

            print(f"-----------\n"
                    f"Epoch {epoch+1}/{epochs}\n"
                    f"Train loss: {running_loss/print_every:.3f}. " # We have been running print_every nr of epochs
                    f"Tune loss: {tune_loss/len(tuneloader):.3f}\n" # len(X-loader) = nr of images in in a batch
                    f"Train accuracy: {run_accuracy/print_every:.3f}. "
                    f"Tune accuracy: {accuracy/len(tuneloader):.3f}\n")

            running_loss = 0
            tune_loss = 0
            run_accuracy = 0

            # Apply the training to the model
            model.train()

end = timeit.timeit()
print(f'Tid: {end - start}')


# -- TEST THE MODEL
test_loss = 0
accuracy = 0
model.eval() # set model in evaluation mode

with torch.no_grad(): # reduces memory usage - we dont need to calculate the gradients in evaluation mode

    for images, labels in testloader:

        # Move images and labels tensors to the default device (GPU if available)
        images = images.to(device)
        labels = labels.to(device)

        # Calculate the loss
        log_ps = model(images) # log of probability
        loss = criterion(log_ps, labels)
        test_loss += loss.item()

        # calculate the accuracy
        ps = torch.exp(log_ps) # get the actual probability
        top_p, top_class = ps.topk(1, dim=1) # top probabilities and classes

        equals = top_class == labels.view(*top_class.shape) # check how many images which have the correct classification.
        # That *top_class.shape is passing all of the items in the top_class.shape into the view function call as separate arguments, without us even needing to know how many arguments are in the list. For example if the top_class has a shape say (32, 1), so *top_class.shape will pass 32, 1 to view function. Now this is as good as passing (top_class.shape[0],top_class.shape[1]). labels.view(top_class.shape[0],top_class.shape[1]) is equal to labels.view(*top_class.shape). The labels.view(*top_class.shape) is very useful because we dont have to know the exact shape for us to reshape them, we just use * and it will pack everything into a list and pass on to the view function for reshaping

        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        # after getting the top_class of the top_probability from torch.exp(output), we equal the top_class with the images labels (targets) to check if they match or not. The result from this equality is binary values [0,1]. Does this mean that 1 refers to class-label matching and 0 refers to class-label mismatching? Yes

print(f"Test loss: {test_loss/len(testloader):.3f}.. "
      f"Test accuracy: {accuracy/len(testloader):.3f}")


# -- SAVE MODEL
print('Saving model..')
# Define the classifier definitions
model.class_to_idx = train_data.class_to_idx

# Save the checkpoint/model
checkpoint = {'arch': 'vgg16',
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'class_to_idx': train_data.class_to_idx,
              'epochs': epochs,
              'learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
              'input_size': 25088,
              'output_size': 102,
              'hidden_layers': nr_hidden_units,
              'state_dict': model.state_dict()}

name = args.save_dir
torch.save(checkpoint, name)
print(f'Model saved as {name}')
