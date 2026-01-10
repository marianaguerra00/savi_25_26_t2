from torchvision import transforms, datasets

def get_mnist_datasets(data_path='./data'):
    # pre processing pipeline to get pytorch tensor ready for calculus and applies statistics normalization to increase training efficiency
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # automatically imports MNIST dataset and applies the transform defined above
    train_set = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    
    return train_set, test_set