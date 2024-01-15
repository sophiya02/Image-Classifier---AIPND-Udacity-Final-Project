# Import necessary libraries
import torch
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from torchvision import models
from utils import load_model, process_image, get_flower_name

# Define the command-line arguments
parser = argparse.ArgumentParser(description='Predict flower name from an image with the probability of that name.')
parser.add_argument('input', help='Path to the input image')
parser.add_argument('checkpoint', help='Path to the checkpoint file')
parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping of categories to real names')
parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')


# Parse the command-line arguments
args = parser.parse_args()

# Image path
image_path= args.input

# Load the model checkpoint
print("----------Loading the saved Model--------------")
model, CLASSES, CLASS_TO_IDX = load_model(args.checkpoint)

# function to predict on a give image sample and draw a bargraph
print("----------Predicting the model--------------")
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Process the image
    image = process_image(image_path)
    
    # Add a batch dimension
    image = image.unsqueeze(0)
    
    # Move the model and input tensor to the same device
    model.to(image.device)
    
    # Move the model to evaluation mode
    model.eval()
    
    # Disable gradients during inference
    with torch.no_grad():
        # Forward pass
        output = model(image)
        
    # Calculate probabilities and class indices
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_probs, top_indices = torch.topk(probabilities, topk)
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in CLASS_TO_IDX.items()}
    top_classes = [idx_to_class[idx.item()] for idx in top_indices]
    top_flower_names=[get_flower_name(idx) for idx in top_classes]

    # Create a horizontal bar graph
    plt.figure(figsize=(10, 6))

    # Plot the image
    plt.subplot(2, 1, 1)
    plt.imshow(Image.open(image_path))
    plt.title(top_flower_names[0])
    plt.axis('off')

    # Plot the top-k classes
    plt.subplot(2, 1, 2)
    plt.barh([i for i in top_flower_names], top_probs, color='blue')
    plt.xlabel('Predicted Probability')
    plt.title(f'Top-{topk} Predicted Classes')

    plt.tight_layout()
    plt.show()
    
    return top_probs.numpy(), top_classes, top_flower_names

probs, classes, class_names = predict(image_path, model, args.top_k)


