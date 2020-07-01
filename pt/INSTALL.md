## Installation instruction (conda)

```
# installation pytorch
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch-nightly

# annoyingly this will need installation of old torchvision
pip uninstall torchvision
conda install av -c conda-forge
pip install torchvision==0.5

pip install submitit

# arrgh
pip install tensorflow

pip install -e .

```