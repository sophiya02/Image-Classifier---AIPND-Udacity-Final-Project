# Import necessary libraries
import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from utils import load_data, train_model, test_model, save_model
from utils import load_data

# Define the command-line arguments
parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint')
parser.add_argument('data_directory',type=str, help='Path to the data directory')
parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
parser.add_argument('--arch', type=str, default='vgg16', help='Architecture (e.g., "vgg16")')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_units', type=int, nargs=2, default=[256,128], help='Number of hidden units')
parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

# Parse the command-line arguments
args = parser.parse_args()

# constants:
HIDDEN_LAYER_1 = args.hidden_units[0]
HIDDEN_LAYER_2 = args.hidden_units[1]

# Load and preprocess the data
print("----------Loading Data--------------")
train_loader, valid_loader, test_loader, CLASS_TO_IDX = load_data(args.data_directory)

# Build and train the model
print("----------Building Model--------------")
model = getattr(models, args.arch)(pretrained=True)
input_size = model.classifier[0].in_features

for param in model.parameters():
    param.requires_grad = False
classifier = nn.Sequential(
    nn.Linear(input_size, HIDDEN_LAYER_1),  # Adjust input size based on the pre-trained model's architecture
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(HIDDEN_LAYER_1, HIDDEN_LAYER_2),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(HIDDEN_LAYER_2, len(CLASS_TO_IDX)),  # Adjust output size based on the number of classes in your dataset
    nn.LogSoftmax(dim=1)
)

# Replace the pre-trained classifier with the new classifier
model.classifier = classifier

print("----------Model Created--------------")

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
else:
    device = "cpu"
    
print("-----------training------------")
train_model(model, train_loader, valid_loader, criterion, optimizer, args.epochs, args.learning_rate, device)
print("----------testing--------------")
test_model(model, test_loader, criterion, device)

# # Save the checkpoint
print("----------Saving the Model--------------")
FILE_PATH = save_model(model, args.arch, HIDDEN_LAYER_1, HIDDEN_LAYER_2)
print("filepath:", FILE_PATH)

print("----Training completed and checkpoint saved.-----")
