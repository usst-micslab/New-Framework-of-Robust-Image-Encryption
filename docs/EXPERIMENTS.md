# Experiment setup

## Environment

The checked environment used Python 3.10, PyTorch 2.7.1 and torchvision 0.22.1. Install dependencies from the repository root with `python -m pip install -r requirements.txt`.

## COCO images and checkpoint

The paper selects 5,000 training and 1,000 testing images from COCO. The archive's configuration uses COCO 2014. Obtain the images from the [COCO dataset site](https://cocodataset.org/#download) and prepare separate train and held-out sets. Each split is an ImageFolder root with at least one child directory of RGB images. A single child such as `images/` is sufficient because the model uses the image tensor alone. The loader resizes images to 256 × 256.

```text
repository/
  data/coco2014/
    train/images/<training images>.jpg
    val/images/<held-out images>.jpg
  train_code/
    save_model/INL_RIE.pth
```

Set `RIE_DATA_ROOT` to use another parent of `coco2014/`. The CLI also accepts direct ImageFolder paths. `INL_RIE.pth` is the archived INL checkpoint with a `net` state dictionary. Use the lab checkpoint or a checkpoint produced by the training command below.

## Evaluate

From `train_code/`, run one image first. Leave out `--limit` to evaluate the entire validation folder.

```bash
python Image_encryption_and_decryption.py --checkpoint save_model/INL_RIE.pth --limit 1
```

Use `--data-dir` for another ImageFolder or `--output-dir` for another result directory. By default the script applies JPEG quality 90, prints PSNR and SSIM, and saves `outputs/encryption/result_0001.png`. A wrong-key or zero-key reconstruction with non-finite values is marked explicitly in the output and figure.

## Train

The default training settings remain in `train_code/config.py`. To verify the setup on one image from each split, run the following from `train_code/`.

```bash
python train.py --epochs 1 --batch-size 1 --max-train-images 1 --max-val-images 1
```

Use `--train-dir` and `--val-dir` to set custom roots. Training writes checkpoints under `save_model/` and TensorBoard records under `logs/`. The final checkpoint is `save_model/INL.pth`, which can be passed to the evaluation command. The training defaults apply when the sample and epoch limits are omitted.

The optional LPIPS helper code is provided with its upstream notice. [LPIPS license](licenses/LPIPS-LICENSE.txt)

See the [validation record](VALIDATION.md).
