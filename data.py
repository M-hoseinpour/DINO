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