import os
from pathlib import Path


num_epochs = 200
clamp = 2.0
exp_name = 0
batch_size = 5
image_size = 128

# ==== <<optimizer>> ==== #
lr = 1e-4
betas = (0.5, 0.999)
gamma = 0.5
weight_decay = 1e-6
train_continue = False

# ==== <<Noise layer setting>> ==== #
QF_factor = 50
noise_str_max = 0.
scale_factor_min = 1.
step_thr = 5000

rnd_trans = 0.1
rnd_bri = 0.3
rnd_noise = 0.02
rnd_sat = 1.0
rnd_hue = 0.1
contrast_low = 0.5
contrast_high = 1.5
jpeg_quality = 50
no_jpeg = False
rnd_trans_ramp = 10000
rnd_bri_ramp = 10000
rnd_sat_ramp = 10000
rnd_hue_ramp = 10000
rnd_noise_ramp = 10000
contrast_ramp = 10000
jpeg_quality_ramp = 10000

# ============ <<Training Path>> ============ #
data_root = Path(os.environ.get("RIE_DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
training_data_path = data_root / "coco2014" / "train"
test_data_path = data_root / "coco2014" / "val"
evaluation_checkpoint = Path(__file__).resolve().parent / "save_model" / "INL_RIE.pth"
logs_path = './logs/'
model_save_path = './save_model/'
text_recoder_path = './text_rec/'

# =========== <<Encryption path>> =========== #
original_path = './100-256/original_img/'
cor_encryption_path = './100-256/Encrypted_img/Cor/'
err1_encryption_path = './100-256/Encrypted_img/DC/'
err2_encryption_path = './100-256/Encrypted_img/KS/'

# =========== <<Test path>> =========== #
# original_image_path = './simple_test/original_image/COCO/'
original_image_path = './simple_test/test_image/'

noise_image_path = './simple_test/noise_image/'
Encrypted_image_path1 = './simple_test/encrypted_image1/'
Encrypted_image_path2 = './simple_test/encrypted_image2/'
lost_info_path = './simple_test/lost_info/'
residual_img_path = './simple_test/residual_imgage/'
Attacked_image_path = './simple_test/attacked_image/'
Decrypted_image_cor_path = './simple_test/De_cor/'
Decrypted_image_cor_path2 = './simple_test/De_cor2/'
Decrypted_image_err_path1 = './simple_test/De_err1/'
Decrypted_image_err_path2 = './simple_test/De_err2/'
Decrypted_image_irr_path = './simple_test/De_irr/'
Decrypted_image_cor_wn_path = './simple_test/De_cor_wn/'
de_img_lost_path = './simple_test/De_lost/'






