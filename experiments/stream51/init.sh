#/bin/bash
# download git repo with data loader
git clone https://github.com/tyler-hayes/Stream-51

# download video frames
echo "Trying to download https://drive.google.com/file/d/15huZ756N2cp1CCO4HxF-MVDsMx1LMoIn/view?usp=sharing"
echo "Should this fail, to so manually and place at $( pwd )"
gdown --id "15huZ756N2cp1CCO4HxF-MVDsMx1LMoIn" --output Stream-51.zip

# extract video frames from zip
mkdir -p "data"
unzip Stream-51.zip -d "data"

# extract feature vectors with pytorch and InceptionV3
echo "Extracting Features. If CUDA GPUs are available, this will be done on GPU. By default we use a batch_size of 1024. On smaller devices this may lead to Out-of-Memory Errors. Head into $( pwd )/init.py and decrase the batch size to solve this issue."
python init.py
