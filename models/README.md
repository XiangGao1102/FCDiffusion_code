Our model is based on the pretrained text-to-image latent diffusion model. Specifically, we use **Stable Diffusion v2-1-base** model in our method. Download the model checkpoint file **v2-1_512-ema-pruned.ckpt** [here](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main) and put it in the **models** folder of the project. Then, run the python script **tool_add_control_sd21.py** to create our initialized model: 
<pre><code>
python tool_add_control_sd21.py ./models/v2-1_512-ema-pruned.ckpt ./models/FCDiffusion_ini.ckpt
</code></pre>
This script will create a ckpt file of our model with the parameters initialized from the pretrained Stable Diffusion v2-1-base. The created ckpt file named **FCDiffusion_ini.ckpt** will be in the **models** folder of the project, as shown below:
<div style="padding-left: 4%; padding-right: 4%;">
                <div align="center">
                    <img src="pictures/ckpt_file.png" width="70%"> <br>
		</div>
</div>
The training of the model will be started from the generated FCDiffusion_ini.ckpt.
