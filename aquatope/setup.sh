conda create --name aqua python=3.10
conda install pytorch pandas tqdm numpy

mkdir data && cd data
wget https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/azurefunctions-dataset2019.tar.xz
tar xfv azurefunctions-dataset2019.tar.xz
