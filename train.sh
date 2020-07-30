docker build -t vae-test -f Dockerfile.vae .

docker run --rm --gpus=all \
-v $(pwd)/logs:/logs \
-v $(pwd)/Datasets:/Datasets \
-v $(pwd)/weights:/weights \
vae-test \
vae_script.py --weights-dir=./weights --logs-dir=./logs