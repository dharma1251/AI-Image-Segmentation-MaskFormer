import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import MaskFormerForInstanceSegmentation, MaskFormerProcessor
from PIL import Image
import numpy as np
from tqdm import tqdm

# Custom Dataset class for loading your images and masks
class CustomSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, processor, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.processor = processor

        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')])
        self.masks = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png') or f.endswith('.jpg')])

        assert len(self.images) == len(self.masks), "Number of images and masks must be equal"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
        
        # Convert mask to numpy array and encode it as expected by the processor
        mask_np = np.array(mask)

        # Prepare the inputs for MaskFormer
        encoding = self.processor(images=image, annotations={"segmentation": mask_np}, return_tensors="pt")

        return {
            'pixel_values': encoding['pixel_values'].squeeze(),  # (3, H, W)
            'labels': encoding['labels'].squeeze()              # (H, W)
        }

def train(
    images_dir,
    masks_dir,
    output_dir,
    num_epochs=10,
    batch_size=4,
    learning_rate=5e-5,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"Using device: {device}")

    processor = MaskFormerProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
    ])

    dataset = CustomSegmentationDataset(images_dir, masks_dir, processor, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(dataloader, leave=True)
        epoch_loss = 0.0
        for batch in loop:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader):.4f}")

        # Save checkpoint every epoch
        checkpoint_path = os.path.join(output_dir, f"maskformer_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MaskFormer on custom dataset")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to training images directory")
    parser.add_argument("--masks_dir", type=str, required=True, help="Path to segmentation masks directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
