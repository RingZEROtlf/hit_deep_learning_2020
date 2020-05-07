## Install

``` bash
# get repo
git clone https://github.com/RingZEROtlf/hit_deep_learning_2020.git

# conda activate your_conda_environment

# install
cd hit_deep_learning_2020
# install requirements (can be passed if already satisified)
pip install -r requirements.txt
pip install --user -e .
```

## Usage

```bash
# find a work directory
deep_learning init YOUR_LAB_FOLDER_NAME # after this, configs will be moved into the folder

cd YOUR_LAB_FOLDER_NAME
deep_learning run YOUR_LAB_NAME [--no-cuda] [--fp16] --config=PATH_TO_YOUR_CONFIG_FILE
```

Tensorboard logs will be wrote into `./runs/YOUR_LAB_NAME`.

In `./YOUR_LAB_NAME`,  there will be checkpoint files, and best checkpoint file.

