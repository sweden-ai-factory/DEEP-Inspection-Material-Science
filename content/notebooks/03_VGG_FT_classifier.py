#!/usr/bin/env python
# coding: utf-8



# # Steel Surface Defect Classification with Transfer Learning
#
# In this script, we'll use transfer learning with a pre-trained CNN model
# to classify steel surface defects from the Severstal Defect Dataset.
#
# The dataset contains four defect classes:
# - Scratches: Surface scratches and abrasions
# - Inclusions: Foreign particles inside the steel
# - Patches: Localized surface irregularities
# - Rolls Marks: Marks caused during the rolling process
#
# ## Dataset Setup
#
# The images are organized into four class folders:
#
# single_class_sampled/
# ├── Scratches/
# ├── Inclusions/
# ├── Patches/
# └── Rolls Marks/
#
# ## Transfer Learning Approach

# VGG_FT_ classifier: 
# The model is trained using:
# - Fine-tuning: Unfreeze selected layers to improve adaptation to steel defects (last conv block-block5-unfrozen for fine-tuning)



import os
import random
import time
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision import datasets, models
from packaging.version import Version as LV
from torchinfo import summary
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# Device configuration
torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)
assert(LV(torch.__version__) >= LV("1.0.0"))


SEED = 42
BATCH_SIZE = 32
IMAGE_SIZE = 255
AUG_MULTIPLIER = 4
EPOCHS = 10


def get_dataset_path() -> Path:
    data_dir = os.getenv("DATADIR")
    if data_dir:
        dataset_path = (Path(data_dir) / "single_class_sampled").resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_path}. Set DATADIR or place the dataset under content/data."
        )

    return dataset_path


def get_class_names(dataset_path: Path) -> list[str]:
    class_names = sorted(
        name for name in os.listdir(dataset_path) if (dataset_path / name).is_dir()
    )
    print(f"Classes found: {class_names}")
    print(f"Number of classes: {len(class_names)}")

    for class_name in class_names:
        class_path = dataset_path / class_name
        num_images = len(
            [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )
        print(f"{class_name}: {num_images} images")

    return class_names


def build_transforms() -> transforms.Compose:
    base_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return base_transform


def split_dataset(full_dataset, all_labels):
    all_indices = list(range(len(full_dataset)))

    train_idx, holdout_idx = train_test_split(
        all_indices, test_size=0.3, random_state=0, stratify=all_labels
    )

    holdout_labels = [all_labels[i] for i in holdout_idx]

    val_idx, test_idx = train_test_split(
        holdout_idx, test_size=0.5, random_state=0, stratify=holdout_labels
    )

    return train_idx, val_idx, test_idx


def augment_dataset(dataset: Subset, multiplier: int) -> ConcatDataset:
    """Return an augmented version of a dataset Subset, repeated `multiplier` times."""
    augment_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.20, scale=(0.02, 0.10)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    root = dataset.dataset.root
    indices = dataset.indices
    augmented_base = datasets.ImageFolder(root=root, transform=augment_transform)
    augmented_subset = Subset(augmented_base, indices)
    return ConcatDataset([augmented_subset] * multiplier)


def prepare_loaders(dataset_path: Path):
    base_transform = build_transforms()

    full_dataset = datasets.ImageFolder(root=str(dataset_path), transform=base_transform)
    all_indices = list(range(len(full_dataset)))
    all_labels = [full_dataset.targets[i] for i in all_indices]

    train_idx, val_idx, test_idx = split_dataset(full_dataset, all_labels)

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    train_dataset2 = augment_dataset(train_dataset, multiplier=AUG_MULTIPLIER)

    train_loader = DataLoader(train_dataset2, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Total images:       {len(full_dataset)}")
    print(
        f"Training samples:   {len(train_dataset2)} "
        f"({AUG_MULTIPLIER}x, on-the-fly augmentation)"
    )
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")

    return train_loader, val_loader, test_loader



def correct(output, target):
    predicted = output.argmax(1)
    correct_ones = (predicted == target).type(torch.float)
    return correct_ones.sum().item()


## Transfer Learning with VGG16 - last conv block (block5) unfrozen for fine-tuning 
class VGG16_FT_Model(nn.Module):
    """
    VGG16 transfer learning with block5 fine-tuning.
    Block5 is unfrozen while earlier convolution layers stay frozen.
    """

    def __init__(self, num_classes, unfreeze_layer=24):
        super(VGG16_FT_Model, self).__init__()

        # Load pretrained VGG16
        self.vgg16 = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
        )

        # Freeze all pretrained layers
        for param in self.vgg16.parameters():
            param.requires_grad = False


        # Remove final ImageNet classifier layer
        self.vgg16.classifier = nn.Sequential(
            *list(self.vgg16.classifier.children())[:-1]
        )


        # Custom classifier
        self.own_layers = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )


        # Unfreeze block5
        self.unfreeze_from(unfreeze_layer)


    def forward(self, x):
        x = self.vgg16(x)
        x = self.own_layers(x)
        return x


    def unfreeze_from(self, layer_idx=24):
        """
        Unfreeze VGG16 feature layers from layer_idx onward.

        VGG16 block5 starts at index 24.
        """

        for idx, layer in enumerate(self.vgg16.features):

            trainable = idx >= layer_idx

            for param in layer.parameters():
                param.requires_grad = trainable
                

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, model_name="Model"):
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    print(f"Training {model_name}...")

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{epochs}]  "
            f"loss: {train_loss:.4f}  acc: {train_acc:.4f}  "
            f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}"
        )

    print("\nTraining completed!")
    return history


def test(test_loader, model, criterion):
    model.eval()

    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)

    test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            total_correct += correct(output, target)

    return {
        "loss": test_loss / num_batches,
        "accuracy": total_correct / num_items,
    }


def main() -> None:
    dataset_path = get_dataset_path()
    class_names = get_class_names(dataset_path)
    train_loader, val_loader, test_loader = prepare_loaders(dataset_path)
   
    # Unfreeze the last conv block for fine-tuning
    model_vgg16_ft = VGG16_FT_Model(num_classes=len(class_names)).to(device)
    
    # Model summary
    print("VGG16 Fine-Tuning Model (block5 unfrozen)")
    print(summary(
        model_vgg16_ft,
        input_size=(1,3,224,224),
        depth=10,
        col_names=["input_size", "output_size", "kernel_size", "num_params"]))

    criterion = nn.CrossEntropyLoss()
    # --- learning rates -----------------------------
    # Backbone (unfrozen block5) trains slowly to preserve pretrained features;
    # the new classifier head trains faster.
    backbone_params = [p for p in model_vgg16_ft.vgg16.parameters() if p.requires_grad]
    head_params = list(model_vgg16_ft.own_layers.parameters())
    
    optimizer_vgg16_ft = optim.Adam([
        {"params": backbone_params, "lr": 1e-5},   # fine-tune slowly
        {"params": head_params,     "lr": 1e-4},   # train head faster
    ])

    num_epochs = EPOCHS
    start_time = time.time()

    # Model training
    history_vgg16_ft = train_model(
        model_vgg16_ft,
        train_loader,
        val_loader,
        optimizer_vgg16_ft,
        criterion,
        num_epochs,
        device,
        "VGG16_ft",
    )

    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTotal training time: {training_time:.2f} seconds")
    print(f"Average time per epoch: {training_time / num_epochs:.2f} seconds")

    # Save trained model
    storage_path = os.environ.get("STORAGE", ".")
    model_dir = os.path.join(storage_path, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "vgg16_ft_defect_classifier.pth")

    torch.save(
        {
            "model_state_dict": model_vgg16_ft.state_dict(),
            "history": history_vgg16_ft,
            "training_time": training_time,
        },
        model_path,
    )
    print(f"Model and history saved to: {model_path}")

    # Model validation
    test_ret = test(test_loader, model_vgg16_ft, criterion)
    print(f"\nTesting: loss: {test_ret['loss']:.6f} accuracy: {test_ret['accuracy']:.2%}")

if __name__ == "__main__":
    main()