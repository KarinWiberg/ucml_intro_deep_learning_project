# -- IMPORTS
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
import json
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import timeit
import argparse

# -- COMMAND LINE PROGRAM FUNCTIONS
# python ImageClassifier/predict.py ImageClassifier/flowers/test/83/image_01774.jpg ImageClassifier/checkpoint_py.pth --top_k 3
parser = argparse.ArgumentParser(description='Recreation of the Model from Checkpoint File and Prediction of the Image Classifier on a Flower')
parser.add_argument('image_path', default='ImageClassifier/flowers/test/83/image_01774.jpg', help='Set input data directory')
parser.add_argument('model_path', default='ImageClassifier/checkpoint_py.pth', help='Set save model data file')
parser.add_argument('-tk', '--top_k', default=5, help='Return the top k nr of most likely predictions')
parser.add_argument('-cn', '--category_names', default='ImageClassifier/cat_to_name.json', help='Specify the path to category to name json-file')
args = parser.parse_args()

print(f'Image path: {args.image_path}')
print(f'Model path: {args.model_path}')
print(f'Specified top k: {args.top_k}')

# -- FUNCTIONS
def load_model(filepath):
    '''Loading the model and rebuilding the model
    filepath = filepath to save checkpoint/model
    returns the saved and rebuilt model'''

    print('Loading model')

    # Run on GPU if possible
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Assign the model data into a variable
    checkpoint = torch.load(filepath)
    #print(checkpoint.keys())

    # Assign the name of the pretrained model as string
    new_model = models.vgg16(pretrained=True) # How to create from parameter?

    # Rebuild the model
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'])),
        ('relu1', nn.ReLU()),
        ('do1', nn.Dropout(0.2)),
        ('fc2', nn.Linear(checkpoint['hidden_layers'], checkpoint['output_size'])),
        ('output', nn.LogSoftmax(dim=1))
        ]))

    # Define new_model classifier
    new_model.classifier = classifier

    # Define new_model weights
    new_model.load_state_dict(checkpoint['state_dict'])

    # Define model class_to_dict
    new_model.class_to_idx = checkpoint['class_to_idx']
    #print(new_model.class_to_idx)

    # Send to the GPU?
    new_model.to(device)

    # Loading model to device
    new_model.to(device)

    return new_model


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a tensor
    '''

    print('Processing model')

    # Open 1 image
    img = Image.open(image_path)

    # Scale
    width, height = img.size

    smallest_len = 256
    ratio = max(smallest_len/width, smallest_len/height)
    rW, rH = round(width*ratio), round(height*ratio)

    img.thumbnail((rW, rH), Image.ANTIALIAS)

    # Crop
    crop_size = 224
    width, height = img.size
    left = (width - crop_size)/2 # zero is in the top left corner
    top = (height - crop_size)/2
    right = left + crop_size
    bottom = top + crop_size

    img = img.crop((left, top, right, bottom))

    # Change color channels (typically 0-255, but model expects floats 0-1)
    np_img = np.array(img)/255 # Normalize manually

    # Normalize the color channels
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - mean)/std

    # Transpose the color channels (pytorch expects
    # color channels to be the first dimension, in PIL
    # image and np array its the 3rd)
    processed_img = np_img.transpose(2,0,1)

    # Modify the output to a tensor
    tensor_img = torch.from_numpy(processed_img)

    return tensor_img
'''
def imshow(image, ax=None, title=None):
    """Imshow for Tensor: Displays the image from Image Tensor
    returns the axis of the picture plot"""

    print('Imshow - Show the picture')

    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    #image = image.numpy()[0].transpose((1, 2, 0))
    image = image.permute(1, 2, 0).numpy()

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)

    return ax
'''

def predict(image_path, model, topk=int(args.top_k)):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    Input: Image path to predict and model
    Return: Top k probabilities as list and top k image indexes (folder indexes - not converted) as list
    '''

    print(f'Predicting topk{args.top_k}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

     # Set the model in evaluation mode
    model.eval()

    # Process Image - From jpeg to Tensor
    #print(img.size())
    img = process_image(image_path);
    #print(img.size())

    # Send tensor to device
    #print(img.is_cuda)
    img = img.to(device=device)
    #print(img.is_cuda)

    # No need to Flatten Tensor - but need to add an dimension
    #print(img.size())
    img = img.unsqueeze(0).float()
    #print(f'Image output size: {img.size()}')

    # Prediction - Run feed forward with current
    # weights to make a prediction
    # Turn off auto_grad
    with torch.no_grad():
        output = model.forward(img)

    # Convert from log_probability to probability
    probs = torch.exp(output)

    # Pick the top 5 highest probabilities
    top_probs, top_idx = torch.topk(probs, topk)
    #print(top_probs, top_idx)

    return top_probs[0].tolist(), top_idx[0].tolist()

# Display an image along with the top 5 classes
def display_solution(image_path, model):

    '''
    Display the picture and the topk model predictions.
    Input: Image path and model
    Output: The picture and a barchart of the predictions showing probabilities
    '''

    print('Display solution')

    import pandas as pd
    #fig, (ax1, ax2) = plt.subplots(figsize=(6,10), ncols=1, nrows=2)

    # Display an image
    #img = process_image(image_path)
    #imshow(img, ax=ax1); # Plots flower

    # Display model predictions
    top_probs, top_classes = predict(image_path, new_model)
    #print(f'Top probabilities: {top_probs}\nTop indexes: {top_classes}')

    # From model_idx find flower_idx
    class_to_idx = model.class_to_idx
    idx_to_class = {value : key for (key, value) in class_to_idx.items()}

    flower_idx = []
    for i in top_classes:
        model_idx = i
        flower_idx.append(idx_to_class[model_idx])
    #print(f'Flower_idx: {flower_idx}')

    # From flower_idx find flower_name
    flower_name = []
    for idx, val in enumerate(flower_idx):
        #print(idx, val)
        #print(cat_to_name[val])
        flower_name.append(cat_to_name[val])

    #print(flower_name)

    # Create a dictionary/df of the top 5 highest probabilities and the corresponding flower name
    prob_dict = dict(zip(flower_name, top_probs))
    print(f'Result dictionary: {prob_dict}')

    #print(prob_dict)
    #df = pd.DataFrame.from_dict(prob_dict, orient='index', columns = ['probability'])

    # Barplot predictions
    #df.plot(kind = 'barh', ax=ax2);
    #plt.title('Top Predictions');
    #plt.gca().invert_yaxis()

    return

# -- LOAD MODEL
name = args.model_path

# Loading the model and rebuilding the model
new_model = load_model(name)
#print(new_model)

# -- PREPROCESSING OF IMAGE (to predict)
# image path
image_path = args.image_path # Works hibiscus

# label for classification
translator = args.category_names
with open(translator, 'r') as f:
    cat_to_name = json.load(f)

# preprocessing
image = process_image(image_path)
#print(f'Process Output shape: {image.shape}')

# -- PREDICTION
# predict(image_path, new_model)

# -- DISPLAY PICTURE & PREDICTION
display_solution(image_path, new_model);
