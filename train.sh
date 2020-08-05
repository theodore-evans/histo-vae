docker build -t vae-test -f Dockerfile.vae .

docker run --rm -d --gpus=all --name=$USER-$(uuidgen) \
-v $(pwd)/logs:/logs \
-v $(pwd)/Datasets:/Datasets \
-v $(pwd)/weights:/weights \
-v $(pwd)/data:/data \
vae-test \
vae_script.py --weights-dir=./weights --logs-dir=./logs -dim=64