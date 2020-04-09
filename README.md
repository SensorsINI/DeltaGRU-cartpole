# DeltaGRU for cart-pole robot
DeltaGRU training code for the cart-pole inverted pendulum robot, based on the the ICRA 2020 paper on controlling AMPRO with EdgeDRNN

# Create Anaconda Environment
Create an environment using the following command:
```
conda create -n pt-cartpole python=3.7 matplotlib pandas tqdm pytorch torchvision cudatoolkit=10.1 scipy -c pytorch
```

# Run
Activate the environment before running the script.
```
conda activate pt-cartpole
```
Run the training script.
```
python train.py
```

# Plot
Run the script test_plot.py to plot the RNN outputs. Please make sure that you have the correct model (.pt file) saved in the 'save' folder.
```
python test_plot.py
```
