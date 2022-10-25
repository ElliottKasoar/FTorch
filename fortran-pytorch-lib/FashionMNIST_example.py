import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import os
import pandas as pd
from torchvision.io import read_image
from pathlib import Path
import numpy as np
import csv

# Set the path this folder is contained in for relative use later
dirpath = Path(__file__).parent


# Generate a custom class object to store dataset
# Methods to:
#   - init, read in dataset
#   - getitem, return a single item
class CustomImageDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# subclass nn.Module
# Methods to:
#   - init, initialise the layers
#   - getitem, return a single item
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print("".join([
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, ",
        f"Avg loss: {test_loss:>8f} \n"
    ])
    )


if __name__ == "__main__":

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Importing data
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Load the FashionMNIST dataset (training and testing) from a folder /data
    # If not available download from the internet
    training_data = datasets.FashionMNIST(
        root=str(dirpath.joinpath("data")),
        train=True, download=True, transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root=str(dirpath.joinpath("data")),
        train=False, download=True, transform=ToTensor()
    )

    # Dataloader is an API that produces 'minibatches' from the dataset
    # It can also set up shuffling of data every epoch to reduce overfitting
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Build Neural Network
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Check if GPU available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Create instance of the Neural Net class defined above and load to device
    model = NeuralNetwork().to(device)
    print(model)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Train the model
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    learning_rate = 1e-3  # How much to update parameters at each epoch
    epochs = 10  # Number of times to iterate

    # Set loss function
    loss_fn = nn.CrossEntropyLoss()

    # Initialise optimiser with parameters to train and learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Saving model and re-loading
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    print("\nHow would you like to construct the model?")
    print(" 1. CPU scripted")
    print(" 2. CPU traced")
    print(" 3. GPU scripted")
    print(" 4. GPU traced")

    dummy_input = torch.ones(1, 28, 28)

    # Get input from the user, sanitise it and focus on the first character
    user_input = input("Enter 1, 2, 3, or 4: ")
    user_input = user_input.strip()[0]

    if user_input == "1":
        # Generate a TorchScript CPU model via scripting
        model_name = str(dirpath.joinpath("scripted_cpu.pt"))
        print("Generating a TorchScript model on the CPU using scripting...")
        # FIXME: torch.jit.optimize_for_inference() when PyTorch issue #81085 is resolved
        scripted_model_cpu = torch.jit.script(model)
        scripted_model_cpu.save(model_name)
        print("Wrote " + model_name)
        output = scripted_model_cpu(dummy_input)
        top5 = F.softmax(output, dim=1).topk(5).indices
        print('TorchScript model top 5 results:\n  {}'.format(top5))

    elif user_input == "2":
        # Generate a TorchScript CPU model via tracing with dummy input
        model_name = str(dirpath.joinpath('traced_cpu.pt'))
        print("Generating a TorchScript model on the CPU using tracing...")
        # FIXME: torch.jit.optimize_for_inference() when PyTorch issue #81085 is resolved
        traced_model_cpu = torch.jit.trace(model, dummy_input)
        traced_model_cpu.save(model_name)
        print("Wrote " + model_name)
        output = traced_model_cpu(dummy_input)
        top5 = F.softmax(output, dim=1).topk(5).indices
        print('TorchScript model top 5 results:\n  {}'.format(top5))

    elif user_input == "3":
        device = torch.device('cuda')
        model_gpu = model.to(device)
        model_gpu.eval()
        dummy_input_gpu = dummy_input.to(device)

        # Generate a TorchScript GPU model via scripting
        print("Generating a TorchScript model on the GPU using scripting...")
        model_name = str(dirpath.joinpath('scripted_gpu.pt'))
        # FIXME: torch.jit.optimize_for_inference() when PyTorch issue #81085 is resolved
        scripted_model_gpu = torch.jit.script(model_gpu)
        scripted_model_gpu.save(model_name)
        print("Wrote " + model_name)
        output = scripted_model_gpu(dummy_input_gpu)
        top5 = F.softmax(output, dim=1).topk(5).indices
        print('TorchScript model top 5 results:\n  {}'.format(top5))

    elif user_input == "4":
        device = torch.device('cuda')
        model_gpu = model.to(device)
        model_gpu.eval()
        dummy_input_gpu = dummy_input.to(device)

        print("Generating a TorchScript model on the GPU using tracing...")
        model_name = str(dirpath.joinpath('traced_gpu.pt'))
        # FIXME: torch.jit.optimize_for_inference() when PyTorch issue #81085 is resolved
        traced_model_gpu = torch.jit.trace(model_gpu, dummy_input_gpu)
        traced_model_gpu.save(model_name)
        print("Wrote " + model_name)
        output = traced_model_gpu(dummy_input_gpu)
        top5 = F.softmax(output, dim=1).topk(5).indices
        print('TorchScript model top 5 results:\n  {}'.format(top5))

    else:
        print("Invalid input, please type 1, 2, 3 or 4")

    # read from file
    model = torch.load(model_name)

    # set in evaluation mode i.e. module.train(False)
    model.eval()

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    # Save some test data to file to re-use
    with open("test_data.txt", "w") as f:
        for i in range(0, 10):
            np.savetxt(f, test_data[i][0].numpy()[0, :, :], delimiter=",", fmt="%s")

    # Example of using the model on the test data
    with open("test_data.txt", "r") as f:
        data_iter = csv.reader(f, delimiter=",")
        data = [data for data in data_iter]
    data_array = np.asarray(data, dtype=float)
    data_array = np.reshape(data_array, (10, 28, 28))

    for i in range(0, 10):
        x, y = test_data[i][0], test_data[i][1]
        with torch.no_grad():
            # pred = model(x)
            # predicted, actual = classes[pred[0].argmax(0)], classes[y]
            # print(f'Predicted: "{predicted}", Actual: "{actual}"')
            pred = model(
                torch.from_numpy(data_array[i, :, :].reshape((1, 28, 28))).float()
            )
            # pred = model(np.reshape(data_array[i, :, :], (28, 28)))
            print(f"Predicted: {pred[0]} - {classes[pred[0].argmax(0)]}")
