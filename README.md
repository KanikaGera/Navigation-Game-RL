[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project : Navigation

### Project Details

In this project, I have trained  an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started
1. Install anaconda on __Windows__ using windows installer at https://www.anaconda.com/products/individual#windows

2. Create (and activate) a new environment with Python 3.6.
	- __Windows__: 
	```bash
	conda create --name drl python=3.6 
	activate drl
	```
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/KanikaGera/Navigation-Game-RL.git
cd setup 
conda install pytorch=0.4.1 cuda90 -c pytorch
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drl` environment.  
```bash
python -m ipykernel install --user --name drl --display-name "drl"
```

5. Before running code in a notebook, change the kernel to match the `drl` environment by using the drop-down `Kernel` menu. 

### Structure of Repository

`DQN` Folder consist of Deep Q Network Implementation . It has four files:

	1.`Navigation.ipynb`  consist of unityagent ml library to interact with unity environment and train the agent.
	
	2.`model.py` consists of structure of RL model coded in pytorch.
	
	3.`dqnagent.py` consist of DQN Algorithm Implementation 
	
	4. `model.pt`  is saved trained model with weights.

`Duel_DQN` Folder consist of Duel Deep Q Network Implementation. It has four files

	1.`Navigation-Duel DQN.ipynb`  consist of unityagent ml library to interact with unity environment and train the agent.
	
	2.`duel_model.py` consists of structure of RL model coded in pytorch.
	
	3.`duel_dqn_agent.py` consist of Duel-DQN Algorithm Implementation 
	
	4. `model_1.pt` is saved trained model with weights.

`Double DQN` Folder consist of Duel Deep Q Network Implementation. It has three files

	1.`Navigation-DDQN.ipynb`  consist of unityagent ml library to interact with unity environment and train the agent.
	
	2.`Navigation-DDQN-Test.ipynb` is to test trained agent .
	
	3. `model_3.pt` is saved trained model with weights.

`Video` A video of the agent collecting yellow bananas for a episode is uploaded in this folder.

### Instructions
1. Install Dependies by following commands in __Getting Started__
	
2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
   - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

3. Place the file in the downloaded GitHub repository, in the main repository folder, and unzip (or decompress) the file. 

4. Open the Folder according to the algorithm you want to run. Should open __Double DQN__ if you want to train the best agent only. 

#### Train The Agent
5. Open Navigation-[Model_Name].ipynb 
6. Run Jupyter Notebook 
7. Run the cells to train the model.

#### Test the Agent
8. In case of Double DQN , run cells of  `Navigation-DDQN-Test.ipynb`.
9. For other algoruhms, run the  last Cell of `Navigation-[Model_Name].ipynb` notebook  to calculate the average reward over 100 episodes.

## BEST MODEL - SUBMISSION
The best average reward over 100 episodes when training mode is off , was achieved by  DDQN Algorithm. It should be considered
as solution to Project 1: Navigation Submission. Hence Open Folder Double DQN Folder for evaluation. Report is uploaded in main folder.


