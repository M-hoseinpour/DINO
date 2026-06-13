from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def cifar10_loader(test_batch_size: int = 256):
    transforms_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    ])

    dataset_train = datasets.CIFAR10(root='./data', download=True, train=True, transform=transforms_)
    dataset_test = datasets.CIFAR10(root='./data', download=True, train=False, transform=transforms_)

    train_loader = DataLoader(dataset_train, batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader

def imagenet_loader(train_path, val_path, val_batch_size=16):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    val_dataset   = datasets.ImageFolder(val_path,   transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64,shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

def imagenet_val_loader(val_path, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(val_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

def imagenet_train_loader(train_path, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(train_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)