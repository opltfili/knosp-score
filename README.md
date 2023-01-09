# README

## Installation instructions

### Initial instructions

In order to run the program, you will first need to create conda environment with all necessary packages. Open your conda terminal (e.g. Anaconda Prompt) and create the environment with following command (its name will be `knosp`):  
`conda create --name knosp python==3.10.6 --file requirements.txt`

You need to do this setup only the first time you use this program. After creating the conda environment, you can proceed with the instructions bellow any other time you want to run the program.

### Running the program 

If you already have the `knosp` environment, you can run the program. Open conda terminal and move to the folder of this project (where README.md is). For example (on Windows):  
`cd C:/Users/Filip/knosp_project`

Activate the conda environment:
`conda activate knosp`

And run the program:
`python main.py`

You need to have the inputs in the folder called `data`. The outputs will be saved to `output` folder. 

## Project structure

- *data* - folder with the input data (numbered subfolders for each patient)
- *main.py* should be located on the same level as the dataset (folder *data*)
- *scratch.py* is a file used for debugging and testing the functionality of developed code
- *output* is a folder where output images will be saved