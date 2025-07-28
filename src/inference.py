import torch
from transformers import MaskFormerForInstanceSegmentation, MaskFormerProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_segmentation(image, masks, scores, threshold=0.5):
    image_np = np.array(image).astype(np.uint8)
    plt.imshow(image_np)
    ax = plt.gca()

    for mask, score in zip(masks, scores):
        if score > threshold:
            mask = mask.astype(bool)
            color = np.random.rand(3,)
            masked = np.ma.masked_where(~mask, mask)
            ax.imshow(masked, cmap='jet', alpha=0.5)

    plt.axis('off')
    plt.show()

def run_inference(image_path, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    processor = MaskFormerProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    encoding = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**encoding)

    pred_scores = outputs.logits.softmax(-1).max(-1).values.cpu().numpy()
    pred_masks = (outputs.pred_masks > 0.5).cpu().numpy()

    print(f"Predicted {len(pred_masks)} masks")

    visualize_segmentation(image, pred_masks, pred_scores)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MaskFormer inference on an image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pt file")

    args = parser.parse_args()

    run_inference(args.image_path, args.model_path)
