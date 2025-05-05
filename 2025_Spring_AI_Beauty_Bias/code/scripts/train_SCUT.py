from PIL import Image
import os

import lightning as L

import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset

from l_module import ResNet152Module

torch.set_float32_matmul_precision('high')

class CustomDataset(Dataset):
    def __init__(self, root, df, transform):
        self.df = df
        self.file_paths = [os.path.join(root, f) for f in df['image'].values]
        try:
            self.labels = torch.tensor(df['label'].values, dtype=torch.float32)
        except KeyError:
            self.labels = torch.zeros(len(self.file_paths), dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        img = Image.open(img_path)

        img = self.transform(img)
        
        label = self.labels[idx]
        label = label.unsqueeze(0)

        return img, label
    
if __name__ == "__main__":
    # Load the dataset
    ff_df = pd.read_csv('../data/FairFace.csv')
    meb_df = pd.read_csv('../data/MEB.csv')
    scut_df = pd.read_csv('../data/SCUT.csv')

    # Train/val/test split
    train_df, val_df = train_test_split(scut_df, test_size=0.33, random_state=42)
    val_df, test_df = train_test_split(val_df, test_size=0.33, random_state=42)

    # Define the transformations
    train_transform = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(10),
        v2.RandomResizedCrop(256, scale=(0.8, 1.0)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = v2.Compose([
        v2.Resize((256, 256)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create datasets    
    train_dataset = CustomDataset(root='SCUT_cropped_images', df=train_df, transform=train_transform)

    val_dataset = CustomDataset(root='SCUT_cropped_images', df=val_df, transform=val_transform)
    test_dataset = CustomDataset(root='SCUT_cropped_images', df=test_df, transform=val_transform)

    fairface_dataset = CustomDataset(root='fairface_cropped_images', df=ff_df, transform=val_transform)
    meb_dataset = CustomDataset(root='MEB_cropped_images', df=meb_df, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )
    meb_loader = DataLoader(
        meb_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )
    fairface_loader = DataLoader(
        fairface_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )

    # Initialize the model
    model = ResNet152Module(lr=3e-3)
    model.freeze()
    model.unfreeze('fc')

    # Initialize the trainer
    trainer = L.Trainer(
        max_epochs=-1,
        precision='bf16-mixed',
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="scut_{epoch:02d}-{val_loss:.4f}",
            ),
            L.pytorch.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min",
            ),
        ]
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    layers = ['layer4', 'layer3']

    for layer in layers:
        print(f"Unfreezing {layer}")
        model.unfreeze(layer)
        trainer.fit(model, train_loader, val_loader, ckpt_path=trainer.checkpoint_callback.best_model_path)

    # Restore the best model
    model = ResNet152Module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test the model
    trainer.test(model, test_loader)

    # Predict on the MEB dataset
    predictions = trainer.predict(model, meb_loader)
    meb_df['predictions'] = torch.cat(predictions).to(torch.float32).numpy()
    meb_df.to_csv('../data/MEB_predictions.csv', index=False)

    # Predict on the FairFace dataset
    predictions = trainer.predict(model, fairface_loader)
    ff_df['scut_predictions'] = torch.cat(predictions).to(torch.float32).numpy()
    ff_df.to_csv('../data/FairFace_predictions.csv', index=False)