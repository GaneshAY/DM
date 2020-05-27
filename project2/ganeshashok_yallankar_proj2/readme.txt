#readme.txt


CSE 572 Spring 2020

	Project 2


Requirements:
	Python 3.7.5 
	pip

Python pip modules:
	re
	os
	sys 
	numpy
	pandas
	pywt
	tsfresh
	sklearn
	pickle
	argparse

Can be installed by following command:

	pip install requirements.txt


Running train.py and test.py:

	- Navigate to the directory containing train.py and test.py
	
	- To run test.py
		* Run the command
			python test.py --file <absolute_path_of_csv.csv>
		
		* Output is printed on console. also stored inside output.csv

	- To run train.py
		* Run the command
			python train.py
			trains the model and stores inside <PWD>/models/classifier.pkl 
		* Optionally can pass --test 1, to test model with training data and
		  it prints - accuracy,precision, recall and f1score.

