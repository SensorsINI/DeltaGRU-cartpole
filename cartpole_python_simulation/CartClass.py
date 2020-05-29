# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:27:59 2020

@author: Marcin
"""

# Cart Class
# The file contain the class holding all the parameters and methods
# related to CartPole which are not related to PyQt5 GUI


from numpy import around, random, pi, sin, cos, sign
# Shapes used to draw a Cart and the slider
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
# NullLocator is used to disable ticks on the Figures
from matplotlib.pyplot import NullLocator
# rc sets global parameters for matlibplot; transforms is used to rotate the Mast
from matplotlib import transforms, rc
# Import module to interact with OS
import os
# Import module to save history of the simulation as csv file
import csv
# Import module to get a current time and date used to name the files containing the history of simulations
from datetime import datetime

#Set the font parameters for matplotlib figures
font = {'size'   : 22}
rc('font', **font)


class Cart:                  
    def __init__ (self,             
                  
                  # (Initial) State of the cart
                  # It is only a state after initialization, not after reset
                  # For the later see reset_state function
                  CartPosition = 0.0,
                  CartPositionD = 0.0,
                  CartPositionDD = 0.0,
                  angle = (2.0*random.normal()-1.0)  * pi/180.0,
                  angleD = 0.0,
                  angleDD = 0.0,
                  ueff = 0.0,
                  Q = 0.0,
                  
                  # Other variables controlling flow of the program
                  Q_max = 1,
                  slider_value = 0.0,
                  mode = 0,
                  dt = 0.005,
                  save_history = True,
                  
                  # Variables used for physical simulation
                  m = 2.0, # mass of pend, kg
                  M = 20.0, # mass of cart, kg
                  Mfric = 0.0,#1.0, # cart friction, N/m/s
                  Jfric = 0.0,#10.0, # friction coefficient on angular velocity, Nm/rad/s
                  g = 9.8, # gravity, m/s^2
                  L = 10.0, # half length of pend, m
                  umax = 300.0, # max cart force, N
                  maxv = 10.0, # max DC motor speed, m/s, in absense of friction, used for motor back EMF model
                  controlDisturbance = 0.01, # disturbance, as factor of umax
                  sensorNoise = 0.01, # noise, as factor of max values
                  
                  #Variables for the controller
                  angle_safety_limit = 6.0,
                  kx = 0.5,
                  kxd = 5.0,
                  ka = 0.5,
                  kad = 5.0,
                  
                  #Dimensions of the drawing
                  CartLength = 10.0,
                  WheelRadius = 0.5,
                  WheelToMiddle = 4.0,
                  MastHight = 10.0, # Only drawing, not the one used for physical simulation!
                  MastThickness = 0.05,
                  y_plane = 0.0,
                  HalfLength = 50.0
                  ):
    

        
        
        # State of the cart
        self.CartPosition = CartPosition
        self.CartPositionD = CartPositionD
        self.CartPositionDD = CartPositionDD
        self.angle = angle
        self.angleD = angleD
        self.angleDD = angleDD
        self.ueff = ueff
        self.Q = Q
        
        #Other variables controlling flow of the program
        self.mode = mode
        self.Q_max = Q_max
        # Set the maximal allowed value of the slider dependant on the mode of simulation
        if self.mode == 0:                 
            self.slider_max = self.Q_max
        elif self.mode == 1:
            self.slider_max = self.HalfLength
        self.slider_value = slider_value
        self.dt = dt
        self.time_total = 0.0
        self.dict_history = {}
        self.reset_dict_history()
        self.save_history = save_history
        
        #Variables for the controller
        self.angle_safety_limit = angle_safety_limit
        self.kx = kx
        self.kxd = kxd
        self.ka = ka
        self.kad = kad
        
        #Physical parameters of the cart
        self.m = m # mass of pend, kg
        self.M = M # mass of cart, kg
        self.Mfric = Mfric # cart friction, N/m/s
        self.Jfric = Jfric # friction coefficient on angular velocity, Nm/rad/s
        self.g = g # gravity, m/s^2
        self.L = L # half length of pend, m
        self.umax = umax # max cart force, N
        self.maxv = maxv # max DC motor speed, m/s, in absense of friction, used for motor back EMF model 
        self.controlDisturbance = controlDisturbance # disturbance, as factor of umax
        self.sensorNoise = sensorNoise # noise, as factor of max values
        
        #Helpers
        self.J=(self.m*(2.0*self.L)**2)/12.0 # moment of inertia around center of pend, kg*m
        self.jml2 = self.J+self.m*self.L**2 #alpja
        self.ml = self.m*self.L #beta
        self.B=self.M+self.m+(self.ml**2)/self.jml2
        
        #Dimensions of the drawing
        self.CartLength = CartLength
        self.WheelRadius = WheelRadius
        self.WheelToMiddle = WheelToMiddle
        self.y_plane = y_plane
        self.y_wheel = self.y_plane+self.WheelRadius
        self.MastHight = MastHight # For drowing only. For calculation see L
        self.MastThickness = MastThickness
        self.HalfLength = HalfLength # Length of the track
        
        #Elements of the drawing
        self.Mast = FancyBboxPatch((self.CartPosition-(self.MastThickness/2.0), 1.25*self.WheelRadius),
                              self.MastThickness,
                              self.MastHight,
                              fc='g')
        
        self.Chassis = FancyBboxPatch((self.CartPosition-(self.CartLength/2.0), self.WheelRadius),
                              self.CartLength,
                              1*self.WheelRadius,
                              fc='r')
        
        self.WheelLeft = Circle((self.CartPosition-self.WheelToMiddle,self.y_wheel),
                     radius=self.WheelRadius,
                     fc='y',
                     ec = 'k',
                     lw = 5)
        
        self.WheelRight = Circle((self.CartPosition+self.WheelToMiddle, self.y_wheel),
                             radius=self.WheelRadius,
                             fc='y',
                             ec = 'k',
                             lw = 5)
        
        self.Slider = Rectangle((0.0,0.0), slider_value, 1.0)
        self.t2 = transforms.Affine2D().rotate(0.0) # An abstract container for the transform rotating the mast
        
        
    # This method changes the internal state of the CartPole
    # from a state at time t to a state at t+dt   
    def update_state(self, slider = None, mode = None, dt = None, save_history = True):
        
        # Optionally update slider, mode and dt values
        if slider:
            self.slider_value = slider
        if mode:
         self.mode = mode
        if dt:
            self.dt = dt
        self.save_history = save_history
            
        # In case in the last step the wheel of the cart
        # went beyond the track
        # Bump elastically into an (invisible) boarder
        if (abs(self.CartPosition)+self.WheelToMiddle)>self.HalfLength:
            self.CartPositionD = -self.CartPositionD
        
        # Determine the dimensionales [-1,1] value of the motor power Q
        
        if self.mode == 1:  # in this case slider gives a target position
        
            # We use a PI control on Cart position and speed to determin the angle to which we want to stabilize the Pole
            # The pendulum has to lean in the direction of the movement
            # This causes acceleration in the desired direction
            set_angle = ((self.slider_value-self.CartPosition)*self.kx-self.CartPositionD*self.kxd)
            # But we never want the set angle to be too big - otherwise we loose control irreversibly
            # The following if condition implements an empirically determined boundary for set_angle
            if abs(set_angle)>self.angle_safety_limit:
                set_angle = sign(set_angle)*self.angle_safety_limit
            # We convert angle to radians
            set_angle = (pi/180.0) *set_angle
            # Determine the power of the motor (dimensionless in the -1 to 1 range) with PI controller
            self.Q = (self.angle-set_angle)*self.ka  +  self.angleD*self.kad
            # Implemet condition of saturation of motort power - its magnitude cannot be bigger than 1
            if self.Q > 1.0:
                self.Q = 1.0
            elif self.Q < -1.0:
                self.Q = -1.0
            
        elif self.mode == 0: # in this case slider corresponds already to the power of the motor
            self.Q = self.slider_value
        
        
        
        # Calculate the force created by the motor
        self.ueff = self.umax*(1-abs(self.CartPositionD)/self.maxv)*self.Q # dumb model of EMF of motor, Q is drive -1:1 range
        self.ueff = self.ueff+self.controlDisturbance*(2.0*random.normal()-1.0)*self.umax # noise on control
        
        # Helpers
        ca = cos(self.angle)
        sa = sin(self.angle)
        A = self.jml2+(self.ml**2*(ca)**2/(self.M+self.m)) # A and B are always >= 0
        
        # Calculate new state of the CartPole
        self.CartPositionDD = ( self.ueff    -self.g*self.ml**2*sa/self.jml2    +self.ml**2*self.L*self.angleD**2*ca*sa/self.jml2    -self.CartPositionD*self.Mfric) /self.B
        self.angleDD = ( -self.ml/(self.M+self.m)*ca*self.ueff  +self.m*self.g*self.L*sa   -self.ml*(1+1/(self.M+self.m))*self.angleD**2*ca*sa   -self.angleD*self.Jfric) /A
        self.angleD = self.angleD+self.angleDD*self.dt
        self.CartPositionD = self.CartPositionD+self.CartPositionDD*self.dt
        self.angle = self.angle+self.angleD*self.dt
        self.CartPosition = self.CartPosition+self.CartPositionD*self.dt 
        
        #Update the total time of the simulation
        self.time_total = self.time_total + self.dt
        
        
        # If user chose to save history of the simulation it is saved now
        # It is saved first internally to a dictionary in the Cart instance
        if self.save_history:
            # Saving simulation data
            self.dict_history['time'].append(around(self.time_total, 4))
            self.dict_history['deltaTimeMs'].append(around(self.dt*1000.0,3))
            self.dict_history['position'].append(around(self.CartPosition,3))
            self.dict_history['positionD'].append(around(self.CartPositionD,4))
            self.dict_history['positionDD'].append(around(self.CartPositionDD,4))
            self.dict_history['angleErr'].append(around(self.angle,4))
            self.dict_history['angleD'].append(around(self.angleD,4))
            self.dict_history['angleDD'].append(around(self.angleDD,4))
            self.dict_history['motor'].append(around(self.ueff,4))
            # The PositionTarget is not always meaningful
            # If it is not meaningful all values in this column are set to 0
            if self.mode == 1:
                self.PositionTarget = self.slider_value
            elif self.mode == 0:
                self.PositionTarget = 0.0
            self.dict_history['PositionTarget'].append(around(self.PositionTarget,4))
        
        # Return the state of the CartPole
        return self.CartPosition, self.CartPositionD, self.CartPositionDD, \
                self.angle, self.angleD, self.angleDD, \
                    self.ueff
                    
                    
    # This method only returns the state of the CartPole instance 
    def get_state(self):
        return self.CartPosition, self.CartPositionD, self.CartPositionDD, \
                self.angle, self.angleD, self.angleDD, \
                    self.ueff
    
    
    # This method resets the internal state of the CartPole instance
    def reset_state(self):
        self.CartPosition = 0.0
        self.CartPositionD = 0.0
        self.CartPositionDD = 0.0
        self.angle = (2.0*random.normal()-1.0)  * pi/180.0
        self.angleD = 0.0
        self.angleDD = 0.0
        
        self.ueff = 0.0
        
        self.dt = 0.005
        
        self.slider_value = 0.0
    
    
    # This method draws elements and set properties of the CartPole figure
    # which do not change at every frame of the animation
    def draw_constant_elements(self, fig, AxCart, AxSlider):
        
        # Get the appropriate max of slider depending on the mode of operation
        if self.mode == 0:                 
            self.slider_max = self.Q_max
        elif self.mode == 1:
            self.slider_max = self.HalfLength
        
        # Delete all elements of the Figure
        AxCart.clear()
        AxSlider.clear()
        
        ## Upper chart with Cart Picture
        # Set x and y limits
        AxCart.set_xlim((-self.HalfLength*1.1,self.HalfLength*1.1))
        AxCart.set_ylim((-1.0,15.0))
        # Remove ticks on the y-axes
        AxCart.yaxis.set_major_locator(NullLocator())
        
        # Draw track
        Floor = Rectangle((-self.HalfLength, -1.0),
                                  2*self.HalfLength,
                                  1.0,
                                  fc='brown')
        AxCart.add_patch(Floor)
        
        # Draw an invisible point at constant position
        # Thanks to it the axes is drawn high enough for the mast
        InvisiblePointUp = Rectangle((0,self.MastHight+2.0),
                              self.MastThickness,
                              0.0001,
                              fc='w',
                              ec = 'w')
        
        AxCart.add_patch(InvisiblePointUp)
        # Apply scaling
        AxCart.axis('scaled')
        
        ## Lower Chart with Slider
        # Set y limits
        AxSlider.set(xlim = (-1.1*self.slider_max,self.slider_max*1.1))
        # Remove ticks on the y-axes
        AxSlider.yaxis.set_major_locator(NullLocator())
        # Apply scaling
        AxSlider.set_aspect("auto")
        
        return fig, AxCart, AxSlider
    
    
    # This method accepts the mouse position and updated the slider value accordingly
    # The mouse position has to be captured by a function not included in this class
    def update_slider(self, mouse_position):
        # The if statement formulates a saturation condition
        if mouse_position>self.slider_max:
            self.slider_value = self.slider_max
        elif mouse_position<-self.slider_max:
            self.slider_value = -self.slider_max
        else:
            self.slider_value = mouse_position
    
    
    # This method updates the elements of the Cart Figure which change at every frame.
    # Not that these elements are not ploted directly by this method
    # but rather returned as objects which can be used by another function
    # e.g. animation function from matplotlib package
    def update_drawing(self):
        
        #Draw mast
        mast_position = (self.CartPosition-(self.MastThickness/2.0))
        self.Mast.set_x(mast_position)
        #Draw rotated mast
        t21 = transforms.Affine2D().translate(-mast_position,-1.25*self.WheelRadius)
        t22 = transforms.Affine2D().rotate(-self.angle) 
        t23 = transforms.Affine2D().translate(mast_position,1.25*self.WheelRadius)
        self.t2 = t21+t22+t23
        #Draw Chassis
        self.Chassis.set_x(self.CartPosition-(self.CartLength/2.0)) 
        #Draw Wheels
        self.WheelLeft.center = (self.CartPosition-self.WheelToMiddle,self.y_wheel)
        self.WheelRight.center = (self.CartPosition+self.WheelToMiddle,self.y_wheel)
        #Draw SLider
        self.Slider.set_width(self.slider_value)
        
        return self.Mast, self.t2, self. Chassis, self.WheelRight, self.WheelLeft, self.Slider
    
    
    # This method resets the dictionary keeping the history of simulation
    def reset_dict_history(self):
        self.dict_history = {'time':              [around(self.dt,3)],
                                'deltaTimeMs':    [around(self.dt*1000.0,3)],
                                'position':       [],
                                'positionD':      [],
                                'positionDD':     [],
                                'angleErr':          [],
                                'angleD':         [],
                                'angleDD':        [],
                                'motor':          [],
                                'PositionTarget': []}
        
        
    # This method saves the dictionary keeping the history of simulation to a .csv file
    def save_history_csv(self):
        
        # Make folder to save data (if not yet existing)
        try:
            os.makedirs('save')
        except:
            pass
        
        # Set path where to save the data
        logpath = './save/'+str(datetime.now().strftime('%Y-%m-%d_%H%M%S'))+'.csv'
        # Write the .csv file
        with open(logpath, "w") as outfile:
           writer = csv.writer(outfile)
           writer.writerow(self.dict_history.keys())
           writer.writerows(zip(*self.dict_history.values()))