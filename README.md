### Training VAE

```bash
mkdir weights
mkdir logs

bash train.sh
```
The resulting weights can be loaded from file in the VAE Reconstruction notebook and visualised

### Reconstruction of Images

The weights for different VAEs (latent dims) can be found [here](https://nx9836.your-storageshare.de/s/YTxTDxL2C3EWCFj),
the downloaded weight folders can be moved to a folder name **weights** and the notebook can be executed with the preferred latent dimension

### Reconstruction of Images with Dimension Discovery

The weights for the A model and R model can be found [here](https://nx9836.your-storageshare.de/s/YTxTDxL2C3EWCFj),
the downloaded weight folders can be moved to a folder name **weights_dim** and the notebook can be executed with the preferred dimension number