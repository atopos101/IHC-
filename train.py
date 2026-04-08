import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from vit import MaskedViT
from sam import generate_mask
import cv2
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# 数据增强
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.ImageFolder(
    "/root/autodl-tmp/Patch-based-dataset/train_data_patch",
    transform=transform
)

test_dataset = torchvision.datasets.ImageFolder(
    "/root/autodl-tmp/Patch-based-dataset/test_data_patch",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# 加载 MaskedViT
model = MaskedViT(num_classes=4)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-4)

epochs = 20

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for imgs,labels in train_loader:

        masks = []
        for img in imgs:
            img_np = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
            mask = generate_mask(img_np)
            masks.append(torch.from_numpy(mask))
        masks = torch.stack(masks).to(device)

        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs, masks)

        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch:",epoch,"Train Loss:",total_loss)

    # 测试
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs,labels in test_loader:

            masks = []
            for img in imgs:
                img_np = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
                mask = generate_mask(img_np)
                masks.append(torch.from_numpy(mask))
            masks = torch.stack(masks).to(device)

            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs, masks)
            _,pred = torch.max(outputs,1)

            correct += (pred==labels).sum().item()
            total += labels.size(0)

    acc = correct/total

    print("Test Accuracy:",acc)

torch.save(model.state_dict(),"maskedvit.pth")