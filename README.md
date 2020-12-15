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

Direction discovery algorithm from Voynov, Andrey, and Artem Babenko. “Unsupervised Discovery of Interpretable Directions in the GAN Latent Space.” ArXiv:2002.03754 [Cs, Stat], June 24, 2020. http://arxiv.org/abs/2002.03754.

A similar, independent approach to the same problem (only coincidentally with the same name) can be found here: https://github.com/willgdjones/HistoVAE

The weights for the A model and R model can be found [here](https://nx9836.your-storageshare.de/s/YTxTDxL2C3EWCFj),
the downloaded weight folders can be moved to a folder name **weights_dim** and the notebook can be executed with the preferred dimension number