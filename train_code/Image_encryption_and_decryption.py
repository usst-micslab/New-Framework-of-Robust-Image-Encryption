import argparse
from itertools import islice
from pathlib import Path

import matplotlib
import torch
from pytorch_msssim import ssim
from torchvision import transforms
from torchvision.datasets import ImageFolder

import config as c
from Image_attack import image_attack
from loss_calculation import psnr
from model_inn.encoder_decoder import INL

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Evaluate image encryption and JPEG-resilient decryption.")
    parser.add_argument("--data-dir", type=Path, default=c.test_data_path, help="ImageFolder of RGB inputs")
    parser.add_argument("--checkpoint", type=Path, default=c.evaluation_checkpoint)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/encryption"))
    parser.add_argument("--limit", type=int, help="Evaluate the first N images")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in 1..100")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    test_dataset = ImageFolder(root=args.data_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = INL().to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict({name: value for name, value in state["net"].items() if "tmp_var" not in name})
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    attack_method = 1
    QF_list = [args.jpeg_quality]

    with torch.inference_mode():
        for QF in QF_list:
            idx = 0
            for original_img, _ in islice(test_loader, args.limit):
                idx += 1
                plt.figure(figsize=(12, 6))
                original_img = original_img.to(device)
                # ===========================<<Key generation>>=========================== #
                torch.manual_seed(26)
                ran_img_cor = torch.rand(1, original_img.shape[1], original_img.shape[2], original_img.shape[3])
                ran_img_cor = ran_img_cor.repeat(original_img.shape[0], 1, 1, 1)
                ran_img_cor = ran_img_cor.to(device)

                torch.manual_seed(27)
                ran_img_err = torch.rand(1, original_img.shape[1], original_img.shape[2], original_img.shape[3])
                ran_img_err = ran_img_err.repeat(original_img.shape[0], 1, 1, 1)
                ran_img_err = ran_img_err.to(device)

                # ============<<Image encryption>>============ #
                input_data_cor = [ran_img_cor, original_img]
                input_data_err = [ran_img_err, original_img]

                _, encrypted_img1, _, _ = model(input_data_cor)
                _, encrypted_img2, _, _ = model(input_data_err)
                encrypted_img1 = torch.clamp(encrypted_img1, 0, 1)
                encrypted_img2 = torch.clamp(encrypted_img2, 0, 1)

                # ====================<<攻击方法>>==================== #
                encrypted_img1_w_n = image_attack(attack_method, encrypted_img1, QF)

                # ============<<Image decryption>>============ #
                output_data_cor = [ran_img_cor, encrypted_img1]
                output_data_err = [ran_img_err, encrypted_img1]
                zeros_mat = torch.zeros_like(ran_img_err)
                output_data_irr = [zeros_mat, encrypted_img1]
                output_data_cor_w_n = [ran_img_cor, encrypted_img1_w_n]

                de_left_cor, de_img_cor = model(output_data_cor, rev=True)
                de_left_err, de_img_err = model(output_data_err, rev=True)
                de_left_irr, de_img_irr = model(output_data_irr, rev=True)
                de_left_cor_w_n, de_img_cor_w_n = model(output_data_cor_w_n, rev=True)

                de_img_cor = torch.clamp(de_img_cor, 0, 1)
                de_img_err = torch.clamp(de_img_err, 0, 1)
                de_img_irr = torch.clamp(de_img_irr, 0, 1)
                de_img_cor_w_n = torch.clamp(de_img_cor_w_n, 0, 1)

                results = (
                    ("Original", original_img, None),
                    ("Key image", ran_img_cor, None),
                    ("Encrypted", encrypted_img1, "en"),
                    ("After JPEG", encrypted_img1_w_n, None),
                    ("Correct key", de_img_cor, "de_cor"),
                    ("Correct key after JPEG", de_img_cor_w_n, "de_cor_w_n"),
                    ("Wrong key", de_img_err, "de_err"),
                    ("Zero key", de_img_irr, "de_irr"),
                )
                for index, (label, result, metric_name) in enumerate(results, 1):
                    panel = plt.subplot(2, 4, index)
                    panel.axis("off")
                    panel.set_title(label)
                    if not torch.isfinite(result).all():
                        if metric_name in {"en", "de_cor", "de_cor_w_n"}:
                            raise FloatingPointError(f"{label} produced non-finite pixels")
                        panel.text(0.5, 0.5, "Non-finite output", ha="center", va="center")
                        print(f"{label}: non-finite output, quality metrics unavailable")
                        continue
                    panel.imshow(result[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
                    if metric_name is not None:
                        psnr_value = psnr(result, original_img)
                        ssim_value = ssim(result, original_img, data_range=1.0)
                        panel.set_title(f"{label} ({psnr_value:.2f} dB)")
                        print(f"PSNR_{metric_name}: {psnr_value:.2f} dB | SSIM_{metric_name}: {ssim_value:.4f}")

                plt.tight_layout()
                plt.savefig(args.output_dir / f"result_{idx:04d}.png", dpi=160)
                plt.close()


if __name__ == "__main__":
    main()
