from omegaconf import OmegaConf
import torch
from ldm.util import instantiate_from_config
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from fcdiffusion.dataset import TestDataset
torch.cuda.set_device(0)


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model


# setting of model config and model checkpoint
yaml_file_path = "configs/model_config.yaml"
# set the checkpoint path in the lightning_logs dir
# the model checkpoint should be consistent with the "control_mode" parameter in the yaml config file
ckpt_file_path = "lightning_logs/fcdiffusion_mid_pass_checkpoint/epoch=0-step=9999.ckpt"
scale_factor = 0.18215

# create mode
config = OmegaConf.load(yaml_file_path)
device = torch.device("cuda")
model = load_model_from_config(config, ckpt_file_path, device)
assert model.control_mode in ckpt_file_path.split('/')[1], \
    'the checkpoint model is not consistent with the config file in control mode'
model.eval()


# setting of test image path and target prompt
test_img_path = 'test_img.jpg'   # the path of the test image to be translated
target_prompt = 'photo of an office room'  # the target text prompt for image-to-image translation
target_prompt = target_prompt + ', best quality, highly detailed'
test_res_num = 4


dataset = TestDataset(test_img_path, target_prompt, test_res_num)
dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=False)
for step, batch in enumerate(dataloader):
    log = model.log_images(batch, ddim_steps=50)
    if step == 0:
        reconstruction = log['reconstruction'].squeeze()
        reconstruction = reconstruction.permute(1, 2, 0)
        reconstruction = torch.clamp(reconstruction, -1, 1)
        Image.fromarray(((reconstruction.cpu().numpy() + 1) * 127.5).astype(np.uint8)).show()
    sample = log["samples"].squeeze()
    sample = sample.permute(1, 2, 0)
    sample = torch.clamp(sample, -1, 1)
    Image.fromarray(((sample.cpu().numpy() + 1) * 127.5).astype(np.uint8)).show()















