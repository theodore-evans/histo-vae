### Training VAE

```bash
mkdir weights
mkdir logs

bash train.sh
```
The resulting weights can be loaded from file in the VAE Reconstruction notebook and visualised

### Reconstruction of Images

The weights for different VAEs (latent dims) can be found [here](https://nx9836.your-storageshare.de/apps/files/?dir=/Weights%20Archive/VAE&),
the downloaded weight folders can be moved to a folder name **weights** and the notebook can be executed with the preferred latent dimension
