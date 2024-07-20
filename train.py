import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)


class Concrete_Dataset(Dataset):
    def __init__(self, directory="./data", transform=None, train=True):
        self.directory = directory
        self.transform = transform
        self.train = train

        positive_files = self._get_files("Positive")
        negative_files = self._get_files("Negative")

        number_of_samples = len(positive_files) + len(negative_files)
        self.all_files = [None] * number_of_samples
        self.all_files[::2] = positive_files
        self.all_files[1::2] = negative_files

        self.Y = torch.zeros([number_of_samples], dtype=torch.long)
        self.Y[::2] = 1
        self.Y[1::2] = 0

        if self.train:
            self.all_files = self.all_files[:30000]
            self.Y = self.Y[:30000]
        else:
            self.all_files = self.all_files[30000:]
            self.Y = self.Y[30000:]

        self.len = len(self.all_files)

    def _get_files(self, folder_name):
        folder_path = os.path.join(self.directory, folder_name)
        return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".jpg")]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image = Image.open(self.all_files[idx]).convert("RGB")
        y = self.Y[idx]

        if self.transform:
            image = self.transform(image)

        return image, y


def main(data_directory="./data", batch_size=100, n_epochs=1, learning_rate=0.001, model_path="model.pth"):
    # Data transformation
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    train_dataset = Concrete_Dataset(
        directory=data_directory, transform=transform, train=True)
    validation_dataset = Concrete_Dataset(
        directory=data_directory, transform=transform, train=False)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        dataset=validation_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = models.resnet18(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(512, 2)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training
    loss_list = []
    accuracy_list = []
    N_test = len(validation_dataset)
    N_train = len(train_dataset)

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss_list.append(running_loss / len(train_loader))

        correct = 0
        model.eval()
        with torch.no_grad():
            for x_test, y_test in validation_loader:
                x_test, y_test = x_test.to(device), y_test.to(device)
                z = model(x_test)
                _, yhat = torch.max(z.data, 1)
                correct += (yhat == y_test).sum().item()

        accuracy = correct / N_test
        accuracy_list.append(accuracy)

    print(f'Model accuracy: {accuracy:.4f}')
    torch.save(model, model_path)


if __name__ == "__main__":
    main()
