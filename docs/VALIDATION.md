# Validation record

Windows CPU with Python 3.10.20, PyTorch 2.7.1 and torchvision 0.22.1.

## Archived inputs

- Ran `Image_encryption_and_decryption.py --limit 1` with an archived RGB image and `INL_RIE.pth`. Verified the encrypted output, valid-key decryption, JPEG-degraded decryption and result PNG. The correct-key reconstruction gave finite PSNR and SSIM. The wrong-key and zero-key reconstructions produced non-finite pixels for this sample and were marked explicitly.
- Ran the training entry for one training image, one distinct validation image and one optimizer step. Verified finite loss, the validation output and the saved `INL.pth` checkpoint. The two images came from the project archive.

## Synthetic input

Ran the same one-step training and validation commands with independent 256 × 256 × 3 uint8 random RGB matrices saved as PNG files in ImageFolder layout. The training loss and validation outputs were finite. This checks loading, tensor dimensions, the forward and backward passes, and checkpoint writing.

The single-image evaluation and one-step training checks do not reproduce the paper's COCO benchmark.
