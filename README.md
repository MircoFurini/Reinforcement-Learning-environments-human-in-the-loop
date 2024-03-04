# Reinforcement-Learning-environments-GUI
I have developed two interfaces that allow you to study the gym environments and the SafePlace environment. This was developed in the thesis "A graphical tool for human-in-the-loop reinforcement learning"

## Installation

### SafePlace_GUI
First of all, download Miniconda from the website and install it with the following
instructions:
```bash
bash Miniconda3-latest-Linux-x86_64.sh # install miniconda
source ~/. bashrc # Restart the terminal
conda -V # Print conda version
conda deactivate # To disable base env if active
```
Create a conda environment (we name ours SafePlace-tf) with Python 3.9:
```bash
conda create --name SafePlace-tf python=3.9
conda activate SafePlace-tf # To enable env
```
Make sure your conda/virtual env is active. Then:
```python
pip install --upgrade pip
pip install tensorflow==2.5.2 numpy==1.19.3 pandas==1.4.4 execnet matplotlib SciencePlot==1.0.9 scipy==1.7.3 seaborn==0.11.2
```
To start the program(in virtual env) go to the SafePlace GUI folder and run:
```bash
python SafePlace/codice/dynamic_computation.py
```

### Gymnasium_GUI
You can create a conda environment(follow the same instructions for SafePlace) and
install the following libraries in conda env:
```
pip install --upgrade pip
pip install gym
pip install gymnasium
pip install gymnasium[toy-text]
pip install gymnasium[classic-control]
pip install gymnasium[box2d]
```
Maybe, if you use Windows, you have to install swig:
```python
pip install swig
```
To start the program(in virtual env) go to the Gymnasium GUI folder and run:
```bash
python code/mainApp.py
```
