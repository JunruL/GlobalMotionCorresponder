mkdir datasets
cd datasets

wget http://download.cs.stanford.edu/orion/gmc_data/dynamicgaussian_realworld_scenes.zip
wget http://download.cs.stanford.edu/orion/gmc_data/gmc_realworld_scenes.zip
wget http://download.cs.stanford.edu/orion/gmc_data/gmc_synthetic_scenes.zip
wget http://download.cs.stanford.edu/orion/gmc_data/papr_realworld_scenes.zip
wget http://download.cs.stanford.edu/orion/gmc_data/papr_synthetic_scenes.zip

for f in *.zip; do
    unzip "$f"
done