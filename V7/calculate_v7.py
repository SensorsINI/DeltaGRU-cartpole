# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 18:18:52 2020

@author: Marcin
"""

import numpy as np



m=2.0 # mass of pend, kg
M=2.0 # mass of cart, kg
Mfric=1.0 # cart friction, N/m/s
Jfric= 10.0 # friction coefficient on angular velocity, Nm/rad/s
g=9.8 # gravity, m/s^2
L=40.0; # half length of pend, m
J=(m*(2.0*L)**2)/12.0 # moment of inertia around center of pend, kg*m
tracklength=3.0*2.0*L # half of total length of cart track,m
umax=300.0 # max cart force, N
maxv=10.0 # max DC motor speed, m/s, in absense of friction, used for motor back EMF model

controlDisturbance=0.01 # disturbance, as factor of umax
sensorNoise=0.01 # noise, as factor of max values


#Helpers
jml2 = J+m*L**2 #alpja
ml = m*L #beta



def Calculations_real(CartPosition, CartPositionD,
                      angle, angleD,
                      slider,
                      dt,
                      mode):
    
    
    if mode == 1:  # in this case slider is a target position
        kx = 0.5
        kxd = 5.0
        set_angle = ((slider-CartPosition)*kx-CartPositionD*kxd)
        if abs(set_angle)>6.0:
            set_angle = np.sign(set_angle)*6.0
        set_angle = (np.pi/180.0) *set_angle
        kp = 0.5
        kd = 5.0
        Q = (angle-set_angle)*kp  +  angleD*kd
        if Q>1.0:
            Q = 1.0
        elif Q<-1.0:
            Q = -1.0
    elif mode == 0 or mode == 2:  # in this case slider is Q
        Q = slider
                
    
    
    
    
    ueff=umax*(1-abs(CartPositionD)/maxv)*Q # dumb model of EMF of motor, Q is drive -1:1 range
    ueff=ueff+controlDisturbance*np.random.normal()*umax # noise on control
    
    ca=np.cos(angle)
    sa=np.sin(angle)
    A=jml2+(ml**2*(np.cos(angle))**2/(M+m)) # A and B are always >= 0
    B=M+m+(ml**2)/jml2
    CartPositionDD=( ueff    -g*ml**2*sa/jml2    +ml**2*L*angleD**2*ca*sa/jml2    -CartPositionD*Mfric) /B
    angleDD=( -ml/(M+m)*ca*ueff  +m*g*L*sa   -ml*(1+1/(M+m))*angleD**2*ca*sa   -angleD*Jfric) /A
    angleD=angleD+angleDD*dt
    CartPositionD=CartPositionD+CartPositionDD*dt
    angle=angle+angleD*dt
    CartPosition=CartPosition+CartPositionD*dt


    return CartPosition, CartPositionD, CartPositionDD, \
            angle, angleD, angleDD, \
                ueff, slider