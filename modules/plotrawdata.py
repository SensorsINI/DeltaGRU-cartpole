import tkinter as tk
from tkinter import filedialog
from data import load_data, Dataset
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path

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
    position = df.positionErr.to_numpy()  # same for position
    # compute temporal derivatives from state data
    averageDeltaT = deltaTimeMs.mean()  # TODO this is approximate derivative since sample rate varied a bit around 5ms
    dAngle = np.gradient(angle, averageDeltaT)
    dPosition = np.gradient(position, averageDeltaT)
    actualMotorCmd = df.actualMotorCmd.to_numpy()  # zero-centered motor speed command

    plt.subplot(3, 1, 1)
    plt.plot(time, angle)
    plt.title('cart-pole raw data')
    plt.ylabel('angle err (ADC)')

    plt.subplot(3, 1, 2)
    plt.plot(time, position)
    plt.ylabel('position err (encoder)')

    plt.subplot(3, 1, 3)
    plt.plot(time, actualMotorCmd)
    plt.xlabel('time (s)')
    plt.ylabel('motor cmd (PWM)')

    plt.savefig(plotfilename)
    plt.show()
