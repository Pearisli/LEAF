import argparse
import logging
import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from leaf import LeafPipeline, LeafOutput


def parse_args():
    parser = argparse.ArgumentParser(description="LeafPipeline inference script with argparse.")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to finetuned LeafPipeline."
    )
    parser.add_argument(
        "--input_image", 
        type=str, 
        required=True, 
        help="Path to input image."
    )
    parser.add_argument(
        "--output_image", 
        type=str, 
        default="./mask.png", 
        help="Path to save prediction mask."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda", 
        help="Device to run on, e.g., 'cuda' or 'cpu'."
    )

    return parser.parse_args()

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="'%(asctime)s - %(levelname)s -%(filename)s - %(funcName)s >> %(message)s'",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def main():
    args = parse_args()
    setup_logger()

    logging.info(f"Using device: {args.device}")
    device = torch.device(args.device)
    
    logging.info(f"Loading pipeline from {args.pretrained_path} ...")
    pipeline = LeafPipeline.from_pretrained(
        args.pretrained_path, 
        local_files_only=True
    ).to(device)

    pipeline.set_progress_bar_config(disable=False)
    pipeline.unet.eval()
    pipeline.vae.eval()
    logging.info("Pipeline loaded and set to eval mode.")

    # Load input image
    logging.info(f"Loading input image: {args.input_image}")
    input_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    rgb = Image.open(args.input_image).convert("RGB")
    rgb = input_transforms(rgb).unsqueeze(0).to(device)
    logging.info(f"Input image transformed. Shape: {rgb.shape}")

    # Run inference
    logging.info("Running inference ...")
    with torch.autocast(device.type):
        output: LeafOutput = pipeline(
            rgb,
            num_inference_steps=1,
            show_progress_bar=False,
        )
    logging.info("Inference completed.")

    # Process mask
    logging.info("Processing mask ...")
    mask = torch.where(
        torch.mean(output.mask_pred, dim=1, keepdim=True) > 0.5, 
        1, 
        0
    )
    mask = mask.squeeze(0).repeat(3, 1, 1).permute(1, 2, 0)
    mask_arr = (mask.cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(mask_arr)

    # Save result
    img.save(args.output_image)
    logging.info(f"Prediction mask saved to {args.output_image}")


if __name__ == '__main__':
    main()
