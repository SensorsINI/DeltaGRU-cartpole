# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:34:34 2020

@author: Marcin
"""

from matplotlib.patches import FancyBboxPatch
from matplotlib.pyplot import NullLocator, Rectangle, Circle
from matplotlib import transforms
from numpy import arange, around

CartLength = 10.0
WheelRadius = 0.5
WheelToMiddle = 4.0
y_plane = 0.0
MastHight = 10.0
MastThickness = 0.05

def draw_TrackPlane(ax, HalfLength):
    
    
    ax.yaxis.set_major_locator(NullLocator())
    #ax.xaxis.set_major_locator(NullLocator())
    
    ax.set_xlim((-HalfLength,HalfLength))
    ax.set_ylim((-1.0,15.0))
    

    Floor = Rectangle((-HalfLength-CartLength/2.0, -1.0),
                              2*HalfLength+CartLength,
                              1.0,
                              fc='brown')
    ax.add_patch(Floor)
    return ax



def RenderCartPicture (ax, CartPosition):
    
    
    Chassis = FancyBboxPatch((CartPosition-(CartLength/2.0), WheelRadius),
                              CartLength,
                              1*WheelRadius,
                              fc='r')
    ax.add_patch(Chassis)
    
    
    WheelLeft = Circle((CartPosition-WheelToMiddle,y_plane+WheelRadius),
                         radius=WheelRadius,
                         fc='y',
                         ec = 'k',
                         lw = 5)
    WheelRight = Circle((CartPosition+WheelToMiddle, y_plane+WheelRadius),
                         radius=WheelRadius,
                         fc='y',
                         ec = 'k',
                         lw = 5)
    
    ax.add_patch(WheelLeft)
    ax.add_patch(WheelRight)
    
    ax.axis('scaled')
     
    return ax

def draw_slider(ax, slider, slider_max, HalfLength, mode):
    
    ax.clear()
    ax.barh(arange(1), slider, align='center')
    if mode == 0 or mode == 2:
        ax.set_xlabel("Motor power: "+ str(around(slider,2)))
        ax.set(xlim = (-1.1*slider_max,slider_max*1.1))
    elif mode == 1:
        ax.set_xlabel("Target position (m):"+ str(around(slider,2)))
        ax.set(xlim = (-1.1*HalfLength,HalfLength*1.1))
        
    ax.yaxis.set_major_locator(NullLocator())
    ax.set_aspect("auto")
    
def draw_mast(ax, CartPosition, Flag_Arr, angle, mode):
    if mode == 2:
        
        ax.imshow(Flag_Arr,
                         extent = (CartPosition+0.2,CartPosition+4.5, MastHight-3.0,MastHight))
        
        Mast = FancyBboxPatch((CartPosition-(MastThickness/2.0), 1.25*WheelRadius),
                                  MastThickness,
                                  MastHight,
                                  fc='g')
        ax.add_patch(Mast)
        
        
    else:
        
        #invisible Point to keep the hight of the y_axis constant
        #Probably there is a better method to do it
        InvisiblePointUp = FancyBboxPatch((CartPosition-(MastThickness/2.0),MastHight+2.0),
                                  MastThickness,
                                  0.0001,
                                  fc='w',
                                  ec = 'w')
    
        ax.add_patch(InvisiblePointUp)
        
        #invisible Point to keep the hight of the y_axis constant
        #Probably there is a better method to do it
        # InvisiblePointDown = FancyBboxPatch((CartPosition-(MastThickness/2.0),-MastHight),
        #                           MastThickness,
        #                           0.0001,
        #                           fc='w',
        #                           ec = 'w')
    
        # ax.add_patch(InvisiblePointDown)
        
        mast_position = (CartPosition-(MastThickness/2.0))
        
        Mast = FancyBboxPatch((mast_position, 1.25*WheelRadius),
                                  MastThickness,
                                  MastHight,
                                  fc='g')
        
        #Draw rotated mast
        t21 = transforms.Affine2D().translate(-mast_position,-1.25*WheelRadius)
        t22 = transforms.Affine2D().rotate(-angle) 
        t23 = transforms.Affine2D().translate(mast_position,1.25*WheelRadius)
        t2 = t21+t22+t23
        t2 = t2+ ax.transData
        Mast.set_transform(t2)
        
        ax.add_patch(Mast)
        
    

def draw_all(canvas, CartPosition, angle, slider, slider_max, HalfLength, arr_it, mode):
    canvas.AxCart.clear()
    draw_mast(canvas.AxCart, CartPosition, arr_it, angle, mode)
    canvas.AxSlider.clear()
    draw_TrackPlane(canvas.AxCart, HalfLength)
    RenderCartPicture (canvas.AxCart, CartPosition)
    draw_slider(canvas.AxSlider, slider, slider_max, HalfLength, mode)
    canvas.draw()