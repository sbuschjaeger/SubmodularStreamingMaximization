#/bin/bash
# download git repo with data loader
git clone https://github.com/tyler-hayes/Stream-51

# download video frames
gdown --id "15huZ756N2cp1CCO4HxF-MVDsMx1LMoIn" --output Stream51.zip

# extract video frames from zip
mkdir "data"
unzip Stream51.zip -d "data"

# extract feature vectors with pytorch and InceptionV3
python init.py
