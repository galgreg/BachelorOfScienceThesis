# Bachelor of Science Thesis
The project has been created in order to receive the **Bachelor of Science** degree. It contains of two main parts:
1. The thesis - document written in Polish language. The theme is:
  ,\,***Application of evolutionary algorithms in the process of artificial neural networks training***''
3. The application - an appendix to the thesis. It represents the practical aspects of thesis.

The subject of thesis was using machine learning techniques to solve the problem of autonomous car control. 
As a part of thesis, I created the simple system, which runs the neural networks training based on simulations provided by Learning Environment. The task of neural networks was driving a car through one of prepared racetracks. In order to train the networks successfully, I took advantage of two evolutionary algorithms: [Differential Evolution](https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python) and [Particle Swarm Optimization](https://medium.com/analytics-vidhya/implementing-particle-swarm-optimization-pso-algorithm-in-python-9efc2eb179a6).

## Implementation details
Thesis has been created in [LaTeX](https://www.latex-project.org/), and later compiled to PDF file. The source code of thesis is stored in *latex_doc* directory.
To create the application, a lot of technologies have been used. Below is the list of most important of them, together with their version numbers:

1. [Unity](https://docs.unity3d.com/2019.1/Documentation/Manual/index.html) - 2019.1.12f
2. C# - Mono 6.4.0.198
3. Python - 3.6.8
4. [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/tree/0.9.1) - 0.9.1
5. [PyTorch](https://pytorch.org/) - 1.1.0
6. [ddt](https://ddt.readthedocs.io/en/latest/) - 1.2.1
7. [docopt](https://www.docopt.org) - 0.6.2
8. [matplotlib](https://matplotlib.org/) - 3.1.1
9. [Conda](https://docs.conda.io/en/latest/) - 4.7.5

The currently supported systems are Linux and Windows.

## How to build & run the application

### 1. Install Python 3
Version 3.6.8 is preferred. Detailed description of installation process is available [here](https://realpython.com/installing-python/).

### 2. Install Unity (optional)
To install Unity, follow the [official documentation](https://docs.unity3d.com/2019.1/Documentation/Manual/GettingStartedInstallingUnity.html). Unity is necessary **only if** you would like to open Learning Environment in Unity Editor. Otherwise, you can use standalone builds, which are available on branches `master_Windows` and `master_Linux`. Choose branch which matches to your OS.

### 3. Install Conda
Please follow [this guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to install Conda. 

### 4. Clone Git repo
```
git clone https://github.com/galgreg/BachelorOfScienceThesis
```
### 5. Create virtual environment and install dependencies
Go to *application/python_external_process* directory and call below commands:
```
conda create --name <your-env-name> --file requirements.txt
conda activate <your-env-name>
```
### 6. Run tests
```
python -m unittest discover
```
### 7. Train neural networks, using Differential Evolution
```
python train_de.py <argument_list>
```
### 8. Train neural networks, using PSO
```
python train_pso.py <argument_list>
```
### 9. Run trained model
```
python run.py <argument_list>
```
### 10. Create standalone builds
```
python make_builds.py <argument_list>
```
### 11. Run experiments sequence
```
python experiment.py <argument_list>
```
### 12. Check help to see how to call given script
```
python <script_name> -h
```

## Acknowledgements
Special thanks for dr [Rafał Skinderowicz](https://www.researchgate.net/profile/Rafat_Skinderowicz), who is my promoter and helped me a lot while writing the thesis.

## Terms of use
Author takes no responsibility for any damage or loss caused by improper use of above project.

