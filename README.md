Initial code pulled from [http://mlwave.com/kaggle-connectomics-python-benchmark-code/](http://mlwave.com/kaggle-connectomics-python-benchmark-code/)

To produce predictions, open a command line prompt and run:

    python model.py [fluorescence data] [network positions] [true network configuration]
    python model.py ../data/fluorescence_iNet1_Size100_CC01inh.txt ../data/networkPositions_iNet1_Size100_CC03inh.txt ../data/network_iNet1_Size100_CC01inh.txt
    python model.py ../data/normal-4/fluorescence_normal-4.txt ../data/normal-4/networkPositions_normal-4.txt ../data/normal-4/network_normal-4.txt
