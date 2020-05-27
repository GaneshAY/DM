
CSE 572 Spring 2020

	Project 3


Requirements:
    Language:
        Python 3.7 with pip3
    Modules:
        sklearn~=0.0
        scikit-learn~=0.21.3
        pandas~=1.0.1
        numpy~=1.18.1
        tsfresh~=0.15.1

Installation:
    pip3 install requirements.txt

Running train.py and test.py:

Training:
    - Navigate to the directory "code"
    - Run the command:
        python train.py
    - Optionally "python train.py --kfold 1" can be used to run kfold cross validation.

Testing:
    - Navigate to the directory "code"
    - Run the command:
        python test.py --file <Absolute path of the test meal data file>
            This will print output for kmeans and dbscan.
