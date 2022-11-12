import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from torch.cuda.amp import autocast, GradScaler

import tqdm
import argparse
import wandb

from vit import ViT

#wandb.init(project="my-test-project")

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default = 2, type = int)
args = parser.parse_args()

DEVICE = 'cuda'
IMAGE_SIZE = 224
BATCH_SIZE = args.batch_size
LEARNING_RATE = 6e-4
EPOCHS = 100

train_transform = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.AutoAugment(policy = T.AutoAugmentPolicy.CIFAR10),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

test_transform = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

train_dataset = CIFAR10(
    root = './cifar_data_train/', 
    train = True,
    download = True,
    transform = train_transform,
)

test_dataset = CIFAR10(
    root = './cifar_data_train/',
    train = False,
    download = True,
    transform = test_transform,
)

train_loader = DataLoader(
    train_dataset, 
    shuffle = True,
    batch_size = BATCH_SIZE, 
)

test_loader = DataLoader(
    test_dataset, 
    batch_size = BATCH_SIZE,
)

model = ViT(
    image_size = IMAGE_SIZE,
    patch_size = 16,
    num_classes = 10,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr = LEARNING_RATE,
)

scaler = GradScaler(enabled = True)

for epoch in tqdm.tqdm(range(EPOCHS), desc='training'):
    epoch_loss = 0
    epoch_acc = 0
    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with autocast(enabled=True, dtype=torch.float16):
            outputs = model(images)
            assert outputs.dtype is torch.float16
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()

        acc = (outputs.argmax(dim = 1) == labels).float().mean()
        epoch_acc += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_val_acc = 0
        epoch_val_loss = 0
        
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            val_output = model(images)
            val_loss = criterion(val_output, labels)

            acc = (val_output.argmax(dim=1) == labels).float().mean()
            epoch_val_acc += acc / len(test_loader)
            epoch_val_loss += val_loss / len(test_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_acc:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_acc:.4f}\n"
    )
    #wandb.log({"epoch": epoch, "train loss": epoch_loss, "train acc": epoch_acc, "val loss": epoch_val_loss, "val acc": epoch_val_acc})