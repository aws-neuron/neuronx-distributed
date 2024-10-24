# Llama 3.2-1B Setup Guide

To run Llama3.2-1B on trn1.2xl instance, clone this repo to your instance and follow the steps below. Run it in the virtual environment (created below) so that this does not affect the primary python environment that is currently working with the official release 2.20. Make sure to run the notebook in this virtual environment as well, or it wont pick up the new changes. 

1. On the trn1 instance, create a folder for our experiment in the home directory
```
cd ~ && mkdir -p nxd-llama3.2
cd nxd-llama3.2
```

2. Clone this repo and checkout the llama_3.2_1B branch
```
git clone https://github.com/aws-neuron/neuronx-distributed.git
cd neuronx-distributed
git checkout llama_3.2_1B
```

3. Create a virtual environment so that we don’t mess with the primary python packages on the trn1 instance: 
```
python3 -m venv nxd-llama3.2
source nxd-llama3.2/bin/activate
```

**Make sure you are now in the virtual environment before proceeding, there should be a (nxd-llama3.2) prefix in your terminal. Your terminal should look similar to this:**
```
(nxd-llama3.2) ubuntu@ip-###-##-##-##:~/nxd-llama3.2/neuronx-distributed$ 
```

If you restart the terminal and do not see the virtual environment, run ```source nxd-llama3.2/bin/activate``` again.

4. Install neuron dependencies (similar to public guide):
```
python -m pip install neuronx_distributed --extra-index-url https://pip.repos.neuron.amazonaws.com
```


**Important:**
The installation script above will install NxD from the official release, but in this repo, we have modified the official NxD code to make it work with llama3.2, so we need to install NxD from the local working directory. The -e flag makes the installation editable:


```
pip install wheel && pip install -e ~/nxd-llama3.2/neuronx-distributed --no-deps
```

5. Download the huggingface [llama3.2-1b](https://huggingface.co/meta-llama/Llama-3.2-1B) model to a ```~/models``` folder in the root directory.

You can download the model with many methods, including [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli), which is shown below. You can install huggingface-cli by running ``pip3 install huggingface_hub[cli]`` in your virtual environment. 

```
cd ~
mkdir models
huggingface-cli download --token YOURTOKEN meta-llama/Llama-3.2-1B --local-dir /home/ubuntu/models/llama-3.2-1b
```

You can create an access token [here](https://huggingface.co/settings/tokens)

6. Now, you can continue on the [notebook](llama3_2_inference.ipynb) to run the llama3.2-1b model. 
You must run the notebook from the virtual environment we created (nxd-llama3.2)

If you are using ``jupyter notebook``  from the terminal, make sure the notebook is run within the virtual environment and uses the kernel of the venv. If unsure, follow https://medium.com/@WamiqRaza/how-to-create-virtual-environment-jupyter-kernel-python-6836b50f4bf4 

If you are using VSCode, remember to choose our new venv-created kernel. You may need to install the Python + Jupyter extensions for VSCode. If unsure, follow https://code.visualstudio.com/docs/python/environments