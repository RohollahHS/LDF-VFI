REPO=LDF-VFI
git clone https://rohollahhs:${GH_TOKEN}@github.com/RohollahHS/${REPO}.git
cd LDF-VFI

module purge all --force
module load python/3.11
module load StdEnv/2023 gcc/12.3 ffmpeg/7.1.1
module load cuda/12.6

virtualenv .venv
source .venv/bin/activate

export PIP_CONFIG_FILE=''
export PYTHONPATH=''

python -m pip install pip setuptools wheel --upgrade

python -m pip install -r requirements.txt
