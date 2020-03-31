import tkinter as tk
from tkinter import filedialog
# from modules.data import load_data, Dataset
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path
# from scipy.interpolate import interp1d
# from scipy.signal import lfilter
from modules.data import filterAndGradients, norm


if __name__ == '__main__':

    root = tk.Tk()
    root.tk.call('tk', 'scaling', 2.0) # doesn't help on hdpi screen
    root.withdraw()

    os.chdir('../data')
    filepath = filedialog.askopenfilename(filetypes=[("CSV data files", "*.csv")])
    if not filepath:
        quit()
    plotfilename = Path(filepath).stem + '.png'
    df = pd.read_csv(filepath)

    # time,deltaTimeMs,angle,position,angleTarget,angleErr,positionTarget,positionErr,angleCmd,positionCmd,motorCmd,actualMotorCmd
    # 172.2985134124756,4.787921905517578,1699,-418,3129,-1428.0,0,-418.0,573146.4030813494,-8360.0,7055,0

    time = df.time.to_numpy()
    deltaTimeMs = df.deltaTimeMs.to_numpy()


    angle = df.angleErr.to_numpy()  # simplify the angle to just angle error, so it is zero centered
    position = df.position.to_numpy()  # same for position
    # compute temporal derivatives from state data
    # averageDeltaT = deltaTimeMs.mean()  # TODO this is approximate derivative since sample rate varied a bit around 5ms
    angle, dAngle, ddAngle = filterAndGradients(angle)
    angle, dAngle, ddAngle = filterAndGradients(angle)
    position, dPosition, ddPosition = filterAndGradients(position)
    actualMotorCmd = norm(df.actualMotorCmd.to_numpy())  # zero-centered motor speed command

    angle=norm(angle)
    dAngle = norm(dAngle)
    ddAngle=norm(ddAngle)
    position=norm(position)
    dPosition=norm(dPosition)
    ddPosition=norm(ddPosition)
    dAngle=norm(dAngle)
    fig, axs = plt.subplots(3, 1, sharex=True) # share x axis so zoom zooms all plots)
    axs[0].plot(time, angle, label='angle')
    axs[0].plot(time, dAngle, label='dAngle')
    axs[0].plot(time, ddAngle, label='ddAngle')
    axs[0].set_ylabel('angle err')
    axs[0].legend(fontsize=14)

    axs[1].plot(time, position, label='position')
    axs[1].plot(time, dPosition, label='dPosition')
    axs[1].plot(time, ddPosition, label='ddPosition')
    axs[1].set_ylabel('position')
    axs[1].legend(fontsize=14)

    axs[2].plot(time, actualMotorCmd,'-o')
    axs[2].set_ylabel('motor cmd')
    axs[2].set_xlabel('time (s)')
    axs[2].legend(fontsize=14)

    plt.title('cart-pole raw data')

    plt.savefig(plotfilename)
    plt.show()
