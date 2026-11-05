# Week2 - Setting Up YOLO


## 1. Setup on LUMI Supercomputer


You should now be able to login to LUMI via [On-demand interface web interface](https://www.lumi.csc.fi/public/). Click the `Log in to LUMI` button and you can choose Jupyter.


### Settings for interactive jupyter session

Below are the detailed settings for each section:

<div class='alert alert-info'>

:::{note}
- **Setup your environment**
	- Project: `project_465002387`
	- Partition: `dev-g`
- **Resources**
	- Number of CPU cores: `8`
	- Memory (GiB): `16`
	- Number of GPUs: `1`
	- Time: `3:00:00`
- **Settings**
	- Working directory: `choose your $HOME folder from the list: /users/<Your_UserName>`
	- Show advanced settings: `1`
	- Python: `pytorch`
	- Module version: `pytorch/2.7`
	- Custom Python type: `Container`
	- Modules to load: **Leave empty**
	- Path to container with Python: `/projappl/project_465002387/DEEP_Inspection_Material_Science/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif`
- **Container arguments**
	```console
	-B /var/spool/slurmd,/opt/cray/,/usr/lib64/libcxi.so.1 --env LD_LIBRARY_PATH=/opt/cray-deps:/opt/cray/libfabric/1.15.2.0/lib64:/opt/cray/pe/mpich/8.1.29/ofi/crayclang/17.0/lib:/opt/cray/pe/lib64:/opt/cray/pe/lib64/cce:/opt/rocm-6.2.0/lib:/opt/aws-ofi-rccl:/.singularity.d/libs
	```
- **Init script for container**
	```console
	export JUPYTER_RUNTIME_DIR=$HOME/.local/share/jupyter/runtime; export JUPYTER_DATA_DIR=$HOME/.local/share/jupyter; source /projappl/project_465002387/DEEP_Inspection_Material_Science/workshopvenv/bin/activate
	```
- **Enable virtual environment**: `1`
- **Virtual environment path**
	- `/projappl/project_465002387/DEEP_Inspection_Material_Science/workshopvenv`
:::
</div>


After these settings, click the button `Launch`, and then a `Connect to jupyter` button will appear in a few minutes.


### Creating your own working directory under your $HOME folder

Open the terminal and paste these lines listed below to the terminal.

```shell
mkdir -p $HOME/.jupyter $HOME/.jupyter_checkpoints

grep -q "jupyter_checkpoints" $HOME/.jupyter/jupyter_lab_config.py 2>/dev/null || \
  echo "c.FileContentsManager.checkpoints_kwargs = {'root_dir': '$HOME/.jupyter_checkpoints'}" \
  >> $HOME/.jupyter/jupyter_lab_config.py

source /projappl/project_465002387/DEEP_Inspection_Material_Science/workshopvenv/bin/activate

python -m ipykernel install --user --name=python3-venv --display-name "Python 3 (venv)"
```

Then you can create your own working directory under your $HOME folder and copy relevant material to your own subdirectory.

```shell
cd $HOME
mkdir -p Yolo_handson
cd Yolo_handson

cp /projappl/project_465002387/DEEP_Inspection_Material_Science/YOLO_1001/Yolo_Detection/detection_deeppcb.ipynb .
cp /projappl/project_465002387/DEEP_Inspection_Material_Science/YOLO_1001/Yolo_Segmentation/segmentation_magnetic_tile.ipynb .
```

Only the notebooks are copied. The datasets stay in the shared project directory and are read directly by the notebooks — you do not need to copy them.

Create the matching checkpoint directory in your home folder:

```shell
mkdir -p "$HOME/.jupyter_checkpoints/Yolo_handson"
```

This lets Jupyter save notebook checkpoints in your home folder and avoids permission errors caused by falling back to `/tmp`. The name after `.jupyter_checkpoints/` must match your working directory name. If you called your folder something other than `Yolo_handson`, use that name here instead.

When you open a notebook, select the kernel **Python 3 (venv)** from the kernel picker in the top right.


## 2. Setup in Local Computer
If you want to use your local computer for the tutorial part of the workshop, you will need to install the required libraries as described below.

### Installing Python

Python is a popular language for machine learning projects. To install Python, follow the [Beginner's Guide](https://wiki.python.org/moin/BeginnersGuide/Download) or head straight to the [download page](https://www.python.org/downloads/).

Please set up your python environment at least a day in advance of the workshop.
If you encounter problems with the installation procedure, ask your workshop organizers via e-mail for assistance so
you are ready to go as soon as the workshop begins.


### Installing required packages

[Pip](https://pip.pypa.io/en/stable/) is the package management system built into Python. Pip should be available in your system once you installed Python successfully.


Open a terminal (Mac/Linux) or Command Prompt (Windows) and run the following commands.

1. Create a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments) called `workshopvenv`.
- On Linux/macOs: `python3 -m venv workshopvenv`
- On Windows: `py -m venv workshopvenv`

2. Activate the newly created virtual environment:
- On Linux/macOs: `source workshopvenv/bin/activate`
- On Windows: `workshopvenv\Scripts\activate.ps1`

<div class='alert alert-warning'>

:::{warning}
Remember that you need to activate your environment every time you restart your terminal!
:::
</div>

3. Install the required packages:
- On Linux/macOs: `python3 -m pip install jupyter seaborn scikit-learn numpy pandas matplotlib tqdm torch torchinfo torchaudio torchvision torchsummary anomalib ultralytics supervision`
- On Windows: `py -m pip install jupyter seaborn scikit-learn numpy pandas matplotlib tqdm torch torchinfo torchaudio torchvision torchsummary anomalib ultralytics supervision`


### Starting Jupyter Lab

We will teach using Python in Jupyter Lab, a programming environment that runs in a web browser. Jupyter Lab is compatible with Firefox, Chrome, Safari and Chromium-based browsers.
Note that Internet Explorer and Edge are *not* supported. See the [Jupyter Lab documentation](https://jupyterlab.readthedocs.io/en/latest/getting_started/accessibility.html#compatibility-with-browsers-and-assistive-technology) for an up-to-date list of supported browsers.

To start Jupyter Lab, open a terminal (Mac/Linux) or Command Prompt (Windows), 
make sure that you activated the virtual environment you created for this course,
and type the command: `jupyter lab`


### Check your setup

To check whether all packages installed correctly, start a jupyter notebook in jupyter lab as explained above. Run the following lines of code:

```python
import sklearn; print('sklearn version: ', sklearn.__version__)
import seaborn; print('seaborn version: ', seaborn.__version__)
import pandas; print('pandas version: ', pandas.__version__)
import torchinfo; print('torchinfo version: ', torchinfo.__version__)
import torch; print('PyTorch version: ', torch.__version__)
```

This should output the versions of all required packages without giving errors. Most versions will work fine with this lesson.


## 3. Using Google Colab


Alternatively, you can run tutorials use [Google Colab](https://colab.research.google.com/).

Go to Google Colab and sign in with your Google account. Most of the required packages are already pre-installed. Note that Google Colab uses Jupyter Notebook rather than JupyterLab.

The prepared dataset and notebooks for this workshop can be downloaded from the following link: [workshop materials](https://uppsalauniversitet-my.sharepoint.com/:f:/g/personal/marzieh_saeedimasine_uppmax_uu_se/IgC95jx7o7ZlS7qiSxkQ0UyvAc4RoIzzJXtkeOQOUEnDN7k?e=1tnauL) under the folder `Yolo_week2`

Download the workshop materials to your computer. Then:
Upload the dataset to your Google Drive so that it can be accessed from Google Colab.
Upload the notebook (.ipynb) files to Google Colab and open them there.
Mount your Google Drive in Colab when prompted so that the notebooks can access the uploaded dataset.

Note that for the segmentation task, the training dataset is directly downloaded using `git clone https://github.com/abin24/Magnetic-tile-defect-datasets.`


## 4. Training Datasets

We use two datasets today: [DeepPCB](https://universe.roboflow.com/tack-hwa-wong-zak5u/deeppcb-4dhir) for object detectio and [Magnetic Tile Defect](https://github.com/abin24/Magnetic-tile-defect-datasets.) for instance segmentation. Both are already staged in the shared project directory, with the paths set in the first cells of the corresponding notebook.
