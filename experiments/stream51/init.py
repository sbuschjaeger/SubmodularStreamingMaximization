import sys
sys.path.append("Stream-51/")
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision

from StreamDataset import *
import tqdm

def stream51():
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    x = StreamDataset(root="data", train=True, transform=transform)
    data = torch.utils.data.DataLoader(x, batch_size=1024, num_workers=8) # YOU CAN DECREASE BATCHSIZE HERE!!!

    # Load pretrained InceptionV3 Classifier
    model = torchvision.models.inception_v3(pretrained=True, transform_input=False).eval()

    # Replace Classifier with identity transformation
    model.fc = torch.nn.Linear(2048, 2048) 
    torch.nn.init.eye_(model.fc.weight)
    torch.nn.init.zeros_(model.fc.bias)

    dataset = []
    labels = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        for x, y in tqdm.tqdm(data):
            for yy in y:
                labels.append(yy.cpu().numpy())
            z = model(x.to(device)).cpu()
            for yy in z:
                dataset.append(yy.numpy())
    dataset = np.array(dataset)
    labels = np.array(labels)
    print(dataset.shape)
    np.save("data/stream51.npy", dataset)
    np.save("data/stream51_labels.npy", labels)


if __name__ == "__main__":
    stream51()


