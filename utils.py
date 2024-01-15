# Import necessary libraries
import torch
from torch import nn, optim
from torchvision import transforms, models, datasets
import json
from PIL import Image
# Constants
TRAIN_BATCH_SIZE = 64
BATCH_SIZE = 32
CLASSES = None
NUM_CLASSES = None
CLASS_TO_IDX = None
HIDDEN_LAYER_1 = None
HIDDEN_LAYER_2 = None

# Utility functions
def load_data(data_dir):
    global CLASSES, NUM_CLASSES, CLASS_TO_IDX
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train_transforms' : transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        'valid_transforms' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        'test_transforms' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    image_datasets = {
        'train_datasets': datasets.ImageFolder(train_dir, transform=data_transforms['train_transforms']),
        'test_datasets': datasets.ImageFolder(test_dir, transform=data_transforms['test_transforms']),
        'valid_datasets': datasets.ImageFolder(valid_dir, transform=data_transforms['valid_transforms'])
    }
    
    CLASSES = image_datasets["train_datasets"].classes
    NUM_CLASSES = len(CLASSES)
    CLASS_TO_IDX = image_datasets["train_datasets"].class_to_idx
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train_datasets'], batch_size=TRAIN_BATCH_SIZE, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test_datasets'], batch_size=BATCH_SIZE),
        'valid': torch.utils.data.DataLoader(image_datasets['valid_datasets'], batch_size=BATCH_SIZE)
    }
    return dataloaders['train'], dataloaders['valid'], dataloaders['test'], CLASS_TO_IDX

def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs, learning_rate, device):
    
    model.to(device)

    # Setting the model to train
    model.train()

    # Training the classifier
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                validation_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        batch_loss = criterion(outputs, labels)

                        validation_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(valid_loader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(valid_loader):.3f}")

                running_loss = 0
                model.train()

def test_model(model, test_loader, criterion, device):
    model.eval()

    test_loss = 0
    accuracy = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    # Print the test accuracy
    print(f"Test Loss: {test_loss/len(test_loader):.3f}.. "
          f"Test Accuracy: {accuracy/len(test_loader):.3f}")
    

def save_model(model, arch, HIDDEN_LAYER_1, HIDDEN_LAYER_2):
    global CLASSES, CLASS_TO_IDX

    # model save location
    checkpoint_path = f"./{arch}.pth"
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'arch': arch,
        'class_to_idx': CLASS_TO_IDX,
        "classes": CLASSES,
        "hidden_layer_1":HIDDEN_LAYER_1,
        "hidden_layer_2":HIDDEN_LAYER_2
    }

    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)

    print(f"Checkpoint saved to {checkpoint_path}")
    
    return checkpoint_path

def load_model(filepath):
    global CLASSES, CLASS_TO_IDX
    # Load the checkpoint and return the model, optimizer, criterion, class names, and other necessary information
    checkpoint = torch.load(filepath)
    CLASSES = checkpoint['classes']
    CLASS_TO_IDX = checkpoint['class_to_idx']
    
    # Determine the model architecture based on the stored information
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    input_size = model.classifier[0].in_features
    
    # Freeze parameters to avoid backpropagation
    for param in model.parameters():
        param.requires_grad = False

    # Build a custom classifier
    classifier = nn.Sequential(
        nn.Linear(input_size, checkpoint['hidden_layer_1']),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(checkpoint['hidden_layer_1'], checkpoint['hidden_layer_2']),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(checkpoint['hidden_layer_2'], len(CLASS_TO_IDX)),
        nn.LogSoftmax(dim=1)
    )

    # Replace the pre-trained classifier with the new classifier
    model.classifier = classifier

    # Load the model state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, CLASSES, CLASS_TO_IDX


def process_image(image_path):
    # Define the transformations
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path)
    processed_image = img_transform(image)
    
    return processed_image

def get_flower_name(label):
     # Load the cat_to_name mapping from the JSON file
     with open('cat_to_name.json', 'r') as f:
         cat_to_name = json.load(f)

     # Assuming 'category_number' is the category number you want to look up
     flower_name = cat_to_name.get(str(label), "Unknown")
     return flower_name
 

