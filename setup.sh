wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
exec bash

conda init
exec bash

conda env create -f environment.yml
conda activate chess_AI
