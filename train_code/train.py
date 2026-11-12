import argparse
import math
import torch
import warnings
import config as c
import os.path as osp
from loss_calculation import *
from pytorch_msssim import ssim
from model_inn.encoder_decoder import *
from noise_layer import transform_net
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter

# ================================<<环境设备设置>>================================ #
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore")
# ================================<<环境设备设置>>================================ #

def load(model, name):
    state_dicts = torch.load(name, map_location="cpu", weights_only=True)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    model.load_state_dict(network_state_dict)
    tem_step = state_dicts.get('tem_step', 0)
    return model, tem_step


class Log:
    def __init__(self, log_path):
        self.log_path = log_path

    def __call__(self, msg):
        print(msg, end='\n')
        with open(self.log_path, 'a') as f:
            f.write(msg)


def model_traning(train_loader, model, optim, epoch_idx, tem_step, writer):
    for original_img, _ in train_loader:
        tem_step += 1
        original_img = original_img.to(device)
        ran_img_cor, ran_img_err, ran_img_tar, ran_img_irr = ran_img_generation(original_img, device)
        input_data_cor = [ran_img_cor, original_img]
        input_data_irr = [ran_img_irr, original_img]

        # ==================<< forward >>================== #
        en_left, encrypted_img, ran_dwt_cor, encrypted_img_dwt = model(input_data_cor)
        en_left_irr, _, _, _ = model(input_data_irr)
        encrypted_img = torch.clamp(encrypted_img, 0, 1)
        residual_img = encrypted_img - ran_img_cor
        en_left = torch.clamp(en_left, 0, 1)

        # ==================<< Noise layer >>================== #
        ramp_fun = lambda ramp: np.min([tem_step / ramp, 1.])  # 一个线性坡度函数
        # QF = 100. - torch.rand(1)[0] * ramp_fun(c.step_thr) * (100. - c.QF_factor)
        QF = 100. - ramp_fun(c.step_thr) * (100. - c.QF_factor)
        noise_str = ramp_fun(c.step_thr) * c.noise_str_max
        scale_factor = 1.0 - torch.rand(1)[0] * ramp_fun(c.step_thr) * (1. - c.scale_factor_min)
        scale_factor = 1.0 - ramp_fun(c.step_thr) * (1. - c.scale_factor_min)
        noise_layer = JpegSS(50, noise_str, scale_factor)

        encrypted_img_w_n = noise_layer(encrypted_img.clone())
        # encrypted_img_w_n = transform_net(encrypted_img.clone(), tem_step)

        # ==================<< backward >>================== #
        output_data_cor = [ran_img_cor, encrypted_img]
        output_data_err = [ran_img_err, encrypted_img]
        output_data_stego = [encrypted_img, encrypted_img]
        output_data_irr = [en_left_irr, encrypted_img]
        output_data_cor_w_n = [ran_img_cor, encrypted_img_w_n]
        output_data_err_w_n = [ran_img_err, encrypted_img_w_n]
        output_data_stego_w_n = [encrypted_img_w_n, encrypted_img_w_n]

        _, de_img_cor = model(output_data_cor, rev=True)
        _, de_img_err = model(output_data_err, rev=True)
        _, de_img_irr = model(output_data_irr, rev=True)
        _, de_img_stego = model(output_data_stego, rev=True)
        _, de_img_cor_w_n = model(output_data_cor_w_n, rev=True)
        _, de_img_err_w_n = model(output_data_err_w_n, rev=True)
        _, de_img_stego_w_n = model(output_data_stego_w_n, rev=True)

        de_img_cor = torch.clamp(de_img_cor, 0, 1)
        de_img_err = torch.clamp(de_img_err, 0, 1)
        de_img_irr = torch.clamp(de_img_irr, 0, 1)
        de_img_stego = torch.clamp(de_img_stego, 0, 1)
        de_img_cor_w_n = torch.clamp(de_img_cor_w_n, 0, 1)
        de_img_err_w_n = torch.clamp(de_img_err_w_n, 0, 1)
        de_img_stego_w_n = torch.clamp(de_img_stego_w_n, 0, 1)

        writer.add_image('Original/original_image', original_img[0], tem_step)
        writer.add_image('Original/ran_img_cor', ran_img_cor[0], tem_step)
        writer.add_image('Original/de_img_cor', de_img_cor[0], tem_step)
        writer.add_image('Original/de_img_cor_w_n', de_img_cor_w_n[0], tem_step)

        writer.add_image('Encryption/encrypted_img', encrypted_img[0], tem_step)
        writer.add_image('Encryption/en_left', en_left[0], tem_step)
        writer.add_image('Encryption/residual_img', residual_img[0] + 0.6, tem_step)
        writer.add_image('Encryption/encrypted_img_w_n', encrypted_img_w_n[0], tem_step)

        writer.add_image('De_err/de_img_err', de_img_err[0], tem_step)
        writer.add_image('De_err/de_img_irr', de_img_irr[0], tem_step)
        writer.add_image('De_err/de_img_stego', de_img_stego[0] + 0.6, tem_step)
        writer.add_image('De_err/de_img_err_w_n', de_img_err_w_n[0], tem_step)
        writer.add_image('De_err/de_img_stego_w_n', de_img_stego_w_n[0] + 0.6, tem_step)

        # ==================<< loss function >>================== #
        # =====<< MSE loss >>===== #
        # ==<<  Encryption  >>== #
        mse_en_tar = MSE_loss_fn(encrypted_img, ran_img_cor)
        mse_en_dwt = MSE_loss_fn(encrypted_img_dwt[:, 0:3, :, :], ran_dwt_cor[:, 0:3, :, :])
        # ==<<  Decryption  >>== #
        mse_de_cor = MSE_loss_fn(de_img_cor, original_img)
        mse_de_cor_w_n = MSE_loss_fn(de_img_cor_w_n, original_img)

        # ==<<Others>>== #
        mse_de_err = 0.25 - MSE_loss_fn(de_img_err, original_img)
        mse_de_err_w_n = 0.25 - MSE_loss_fn(de_img_err_w_n, original_img)
        loss_res = torch.sum(torch.abs(residual_img))

        # =====<< Lpips loss >>===== #
        # lpips_en_cor = torch.abs(torch.mean(lpipsTorch.lpips(stego_img, original_img)) - 0.2)
        # lpips_de_err = torch.abs(torch.mean(lpipsTorch.lpips(de_img_err, original_img)) - 0.75)
        # lpips_de_err_w_n = torch.abs(torch.mean(lpipsTorch.lpips(de_img_err_w_n, original_img)) - 0.75)
        # lpips_de_err = torch.mean(lpipsTorch.lpips(de_img_err, original_img))
        # lpips_de_err_w_n = torch.mean(lpipsTorch.lpips(de_img_err_w_n, original_img))
        # # lpips_en_cor = torch.mean(lpipsTorch.lpips(encrypted_img, original_img))
        # lpips_de_cor = torch.mean(lpipsTorch.lpips(de_img_cor, original_img))
        # lpips_de_cor_w_n = torch.mean(lpipsTorch.lpips(de_img_cor_w_n, original_img))

        # =====<< SSIM loss >>===== #
        ssim_en_ori = ssim(encrypted_img, original_img, data_range=1.0)
        ssim_de_cor = ssim(de_img_cor, original_img, data_range=1.0)
        ssim_de_err = ssim(de_img_err, original_img, data_range=1.0)
        ssim_de_irr = ssim(de_img_irr, original_img, data_range=1.0)
        ssim_de_cor_w_n = ssim(de_img_cor_w_n, original_img, data_range=1.0)
        ssim_de_err_w_n = ssim(de_img_err_w_n, original_img, data_range=1.0)

        # =====<< Total loss >>===== #
        encryption_loss = mse_en_tar + 10 * mse_en_dwt
        de_cor_loss = 1.0 * mse_de_cor + 1.0 * mse_de_cor_w_n
        total_loss = encryption_loss + de_cor_loss
        total_loss.backward()
        optim.step()
        optim.zero_grad()

        # =====<< PSNR >>===== #
        PSNR_en_ori = psnr(encrypted_img, original_img)
        PSNR_de_cor = psnr(de_img_cor, original_img)
        PSNR_de_err = psnr(de_img_err, original_img)
        PSNR_de_irr = psnr(de_img_irr, original_img)
        PSNR_de_cor_w_n = psnr(de_img_cor_w_n, original_img)
        PSNR_de_err_w_n = psnr(de_img_err_w_n, original_img)

        print('The index of epoch:{}, | learning rate:{}'.format(epoch_idx, optim.param_groups[0]['lr']))
        print('# ================================<<Visual quality>>================================= #')
        print('En_img ==> PSNR_en_ori:{:.2f}, | ssim_en_ori:{:.4f}'.format(PSNR_en_ori, ssim_en_ori))
        print('De_img_cor ==> PSNR_de_cor:{:.2f}, | ssim_de_cor:{:.4f}'.format(PSNR_de_cor, ssim_de_cor))
        print('De_img_cor_w_n ==> PSNR_de_cor_w_n:{:.2f}, | ssim_de_cor_w_n:{:.4f}'.format(PSNR_de_cor_w_n,
                                                                                           ssim_de_cor_w_n))
        print('Decrypted img_irr ==> PSNR_de_irr_ori:{:.2f}, | ssim_de_irr_ori:{:.4f}'.format(PSNR_de_irr,
                                                                                              ssim_de_irr))
        print('Decrypted img_err ==> PSNR_de_err_ori:{:.2f}, | ssim_de_err_ori:{:.4f}'.format(PSNR_de_err,
                                                                                              ssim_de_err))
        print('Decrypted img_err ==> PSNR_de_err_ori_w_n:{:.2f}, | ssim_de_err_ori_w_n:{:.4f}'.format(
            PSNR_de_err_w_n, ssim_de_err_w_n))
        print('# =====================================<<Loss value>>===================================== #')
        print('The total_step/step: {}/{}, | total_loss: {:.5f}'.format(c.num_epochs * len(train_loader),
                                                                        tem_step, total_loss))
        print(
            'The total_step/step: {}/{}, | mse_de_cor: {:.5f}, | mse_de_cor_w_n: {:.5f}'.format(
                c.num_epochs * len(train_loader), tem_step, mse_de_cor, mse_de_cor_w_n))
        print('\n')

        # =============================<<tensorboard visualization>>============================= #
        writer.add_scalar('MSE_loss/mse_en_tar', mse_en_tar, tem_step)
        writer.add_scalar('MSE_loss/de_img_cor', mse_de_cor, tem_step)
        writer.add_scalar('MSE_loss/de_img_cor_w_n', mse_de_cor_w_n, tem_step)
        writer.add_scalar('MSE_loss/de_img_err', mse_de_err, tem_step)
        writer.add_scalar('MSE_loss/de_img_err_w_n', mse_de_err_w_n, tem_step)

        writer.add_scalar('PSNR/en_ori', PSNR_en_ori, tem_step)
        writer.add_scalar('PSNR/de_cor', PSNR_de_cor, tem_step)
        writer.add_scalar('PSNR/de_irr', PSNR_de_irr, tem_step)
        writer.add_scalar('PSNR/de_cor_w_n', PSNR_de_cor_w_n, tem_step)
        writer.add_scalar('PSNR/de_err_ori', PSNR_de_err, tem_step)
        writer.add_scalar('PSNR/de_err_ori_w_n', PSNR_de_err_w_n, tem_step)

        writer.add_scalar('SSIM/ssim_en_ori', ssim_en_ori, tem_step)
        writer.add_scalar('SSIM/ssim_de_cor', ssim_de_cor, tem_step)
        writer.add_scalar('SSIM/ssim_de_cor_w_n', ssim_de_cor_w_n, tem_step)
        writer.add_scalar('SSIM/ssim_de_err_ori', ssim_de_err, tem_step)
        writer.add_scalar('SSIM/ssim_de_err_ori_w_n', ssim_de_err_w_n, tem_step)

        # writer.add_scalar('Lpips_loss/lpips_de_cor', lpips_de_cor, tem_step)
        # writer.add_scalar('Lpips_loss/lpips_de_cor_w_n', lpips_de_cor_w_n, tem_step)
        # writer.add_scalar('Lpips_loss/lpips_de_err', lpips_de_err, tem_step)
        # writer.add_scalar('Lpips_loss/lpips_de_err_w_n', lpips_de_err_w_n, tem_step)

        writer.add_scalar('other loss/total_loss', total_loss, tem_step)
        writer.add_scalar('other loss/loss_res', loss_res, tem_step)
        writer.add_scalar('other loss/loss_dwt', mse_en_dwt, tem_step)

        torch.save({'opt': optim.state_dict(),
                    'net': model.state_dict(),
                    'tem_step': tem_step},
                   c.model_save_path + 'INL_new' + '.pth')
        if tem_step % 1000 == 0:
            torch.save({'opt': optim.state_dict(),
                        'net': model.state_dict(),
                        'tem_step': tem_step},
                       c.model_save_path + 'INL_' + str(tem_step) + '.pth')

    return model, tem_step, writer


def model_test(test_loader, model, text_log, epoch_idx):
    model.eval()
    with torch.no_grad():
        image_num = 0
        psnr_en = 0
        psnr_cor = 0
        psnr_cor_w_n = 0
        psnr_err = 0
        psnr_irr = 0
        psnr_err_w_n = 0
        ssim_en = 0
        ssim_cor = 0
        ssim_cor_w_n = 0
        ssim_err = 0
        ssim_irr = 0
        ssim_err_w_n = 0
        for original_img, _ in test_loader:
            image_num += 1
            original_img = original_img.to(device)
            torch.manual_seed(1226)
            ran_img_cor = torch.rand(1, original_img.shape[1], original_img.shape[2], original_img.shape[3])
            ran_img_cor = ran_img_cor.repeat(original_img.shape[0], 1, 1, 1)
            ran_img_cor = ran_img_cor.to(device)

            torch.manual_seed(999)
            ran_img_err = torch.rand(1, original_img.shape[1], original_img.shape[2], original_img.shape[3])
            ran_img_err = ran_img_err.repeat(original_img.shape[0], 1, 1, 1)
            ran_img_err = ran_img_err.to(device)
            input_data_cor = [ran_img_cor, original_img]
            input_data_irr = [ran_img_err, original_img]
            _, encrypted_img, _, _ = model(input_data_cor)
            en_left_irr, _, _, _ = model(input_data_irr)
            encrypted_img = torch.clamp(encrypted_img, 0, 1)

            noise_layer = JpegTest(50)
            encrypted_img_w_n = noise_layer(encrypted_img.clone())

            output_data_cor = [ran_img_cor, encrypted_img]
            output_data_err = [ran_img_err, encrypted_img]
            output_data_irr = [en_left_irr, encrypted_img]
            output_data_cor_w_n = [ran_img_cor, encrypted_img_w_n]
            output_data_err_w_n = [ran_img_err, encrypted_img_w_n]

            _, de_img_cor = model(output_data_cor, rev=True)
            _, de_img_err = model(output_data_err, rev=True)
            _, de_img_irr = model(output_data_irr, rev=True)
            _, de_img_cor_w_n = model(output_data_cor_w_n, rev=True)
            _, de_img_err_w_n = model(output_data_err_w_n, rev=True)

            de_img_cor = torch.clamp(de_img_cor, 0, 1)
            de_img_err = torch.clamp(de_img_err, 0, 1)
            de_img_irr = torch.clamp(de_img_irr, 0, 1)
            de_img_cor_w_n = torch.clamp(de_img_cor_w_n, 0, 1)
            de_img_err_w_n = torch.clamp(de_img_err_w_n, 0, 1)

            ssim_en += ssim(encrypted_img, original_img, data_range=1.0)
            ssim_cor += ssim(de_img_cor, original_img, data_range=1.0)
            ssim_err += ssim(de_img_err, original_img, data_range=1.0)
            ssim_irr += ssim(de_img_irr, original_img, data_range=1.0)
            ssim_cor_w_n += ssim(de_img_cor_w_n, original_img, data_range=1.0)
            ssim_err_w_n += ssim(de_img_err_w_n, original_img, data_range=1.0)

            psnr_en += psnr(encrypted_img, original_img)
            psnr_cor += psnr(de_img_cor, original_img)
            psnr_err += psnr(de_img_err, original_img)
            psnr_irr += psnr(de_img_irr, original_img)
            psnr_cor_w_n += psnr(de_img_cor_w_n, original_img)
            psnr_err_w_n += psnr(de_img_err_w_n, original_img)

        msg = f'The training epoch index: {epoch_idx}\n'
        text_log(msg)
        msg = f'PSNR of en: {psnr_en / image_num:.2f} | cor: {psnr_cor / image_num:.2f} | cor_w_n:{psnr_cor_w_n / image_num:.2f}\n'
        text_log(msg)
        msg = f'PSNR of err: {psnr_err / image_num:.2f} | irr: {psnr_irr / image_num:.2f} | err_w_n:{psnr_err_w_n / image_num:.2f}\n'
        text_log(msg)
        msg = f'SSIM of en: {ssim_en / image_num:.4f} | cor: {ssim_cor / image_num:.4f} | cor_w_n:{ssim_cor_w_n / image_num:.4f}\n'
        text_log(msg)
        msg = f'SSIM of err: {ssim_err / image_num:.4f} | irr: {ssim_irr / image_num:.4f} | err_w_n:{ssim_err_w_n / image_num:.4f}\n\n'
        text_log(msg)


def main():
    parser = argparse.ArgumentParser(description="Train the invertible image encryption model.")
    parser.add_argument("--train-dir", default=c.training_data_path)
    parser.add_argument("--val-dir", default=c.test_data_path)
    parser.add_argument("--epochs", type=int, default=c.num_epochs)
    parser.add_argument("--batch-size", type=int, default=c.batch_size)
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-val-images", type=int)
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "max_train_images", "max_val_images"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    for path in (c.model_save_path, c.logs_path, c.text_recoder_path):
        os.makedirs(path, exist_ok=True)

    # ==========================<< Dataset load >>========================== #
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    train_dataset = ImageFolder(root=args.train_dir, transform=transform)
    test_dataset = ImageFolder(root=args.val_dir, transform=transform)
    if args.max_train_images is not None:
        train_dataset = Subset(train_dataset, range(min(args.max_train_images, len(train_dataset))))
    if args.max_val_images is not None:
        test_dataset = Subset(test_dataset, range(min(args.max_val_images, len(test_dataset))))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # ==========================<< Model load >>========================== #
    model = INL()
    model.to(device)
    init_model(model)
    params_trainable = (list(filter(lambda p: p.requires_grad, model.parameters())))
    optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, 2, 0.5)
    if c.train_continue:
        model, tem_step = load(model, './save_model/INL_RIE.pth')
    else:
        tem_step = 0

    log_path = os.path.join(c.logs_path, str(c.exp_name))  # 路径拼接
    writer = SummaryWriter(log_path)
    text_log = Log(osp.join(c.text_recoder_path, 'log.txt'))
    # ==========================<< Model training >>========================== #
    for epoch_idx in range(args.epochs):
        model.train()
        for param_group in optim.param_groups:
            param_group['lr'] = max(param_group['lr'], 1e-6)

        model, tem_step, writer = model_traning(train_loader, model, optim, epoch_idx, tem_step, writer)
        model_test(test_loader, model, text_log, epoch_idx)
        weight_scheduler.step()

    writer.close()
    torch.save({'opt': optim.state_dict(),
                'net': model.state_dict(),
                'tem_step': tem_step},
               c.model_save_path + 'INL' + '.pth')


if __name__ == '__main__':
    main()
