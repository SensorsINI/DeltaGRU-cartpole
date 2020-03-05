# DeltaGRU for AMPRO
DeltaGRU training code of the ICRA 2020 paper on controlling AMPRO with EdgeDRNN

# Create Anaconda Environment
Create an environment using the following command:
```
conda create -n pt-ampro python=3.7 matplotlib pandas tqdm pytorch torchvision cudatoolkit=10.1 -c pytorch
```

# Run
Activate the environment before running the script.
```
conda activate pt-ampro
```
Run the script.
```
python main.py
```

# Plot
Run the script test_plot.py to plot the RNN outputs. Please make sure that you have the correct model (.pt file) saved in the 'save' folder.
```
python test_plot.py
```