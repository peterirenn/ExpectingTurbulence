# ExpectingTurbulence

ExpectingTurbulence

Reinforcement learning implementation for aerodynamic control in turbulent wake. 

Due to the experimental nature of this work, the full functionality cannot be realized or reproduced without access to our experimental setup. Still, we have provided the full code for review. The core algorithms should be widely applicable to problems in any suitable RL simulated environment. 

Access guide: 
	-Code for running each experiment is broken into three parts. 
		-"agent__.py" files contain core algorithm.
			-Should be applicable to any simulated tasks (MuJuCo, etc.) 
			-TF2 implementations of TD3 and LSTM-TD3
		-"Machine_.py" files contain abstraction and access to "agent" files. 
			-Gets actions, saves data, updates ReplayBuffer, sets reward function, etc. 
			-Should also allow for easy application to other tasks. 
		-"RunTests__.py" files contain code that implement and run RL algorithms on the hardware wing system. 
			-Cannot be used without wing hardware

	-Additionally, "Expecting_Turbulence_Data_Access.ipynb" allows for direct access/plotting to all collected data.
		-Runs on Jupyter Notebook. Just download data and run the cells. 
	

Installation/Running Guide: 
	-Download this full repository. 
	-Navigate to the subfolder for the case you wish to run via the command line. 
	-Type "python RunTests__.py -fil Kalman", and the test will run. 
	-NOTE: This work was experimental, and without the correct experimental setup it cannot be run. 

Dependencies: All code tested/run on Windows 10/11 Intel systems. 

	-Core RL algorithms (agent__.py files):
		-OS: MacOS, Linux, Windows 10/11
		-Python 3.6-8
			-Tensorflow (r2.3+)
			-numpy 
	
	-Machine interface classes: 
		-OS: MacOS, Linux, Windows 10/11
		-Python 3.6-8
			-Tensorflow (r2.3+)
			-numpy 
			-datetime
			-pickle

	-RunTest files (runs hardware): 
		-OS: Windows 10/11
		-Python 3.6-8
			-Tensorflow (r2.3+)
			-numpy 
			-datetime
			-pickle
			-serial
			-nidaqmx

