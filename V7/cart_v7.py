from PyQt5.QtWidgets import QMainWindow, QRadioButton, QApplication, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget, QCheckBox 
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QRunnable, QThreadPool, QTimer, Qt
from PyQt5.QtGui import QIcon
from time import sleep, time


### From CartControl_v3 - not working version

from matplotlib import rc
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.image import imread

from draw_v7 import draw_all
from calculate_v7 import Calculations_real

from numpy import  pi, around, random
import sys, os
import csv

from datetime import datetime

#This piece of code gives a custom ID to our application
# It is essentioal for packaging and e.g. let it be associated with the right icon in the taskbar
try:
    # Include in try/except block if you're also targeting Mac/Linux
    from PyQt5.QtWinExtras import QtWin
    myappid = 'INI.CART.IT.V1'
    QtWin.setCurrentProcessExplicitAppUserModelID(myappid)    
except ImportError:
    pass



font = {'size'   : 22}

rc('font', **font)


class WorkerSignals(QObject):

    result = pyqtSignal(object)


class Worker(QRunnable):


    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()    


    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        
        # Retrieve args/kwargs here; and fire processing using them
        result = self.fn(*self.args, **self.kwargs)
        self.signals.result.emit(result)  # Return the result of the processing

        


class MainWindow(QMainWindow):


    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        
        
        
        
        
        #### initialization of variables
        self.arr_it = imread('IT.png')
        
        self.CartPosition = 0.0
        self.CartPositionD = 0.0
        self.CartPositionDD = 0.0
        self.angle = random.random()  * pi/180.0 #you can give angle in deg
        self.angleD = 0.0
        self.angleDD = 0.0
        self.time_global = 0.0
        self.dt = 0.0
        self.ueff = 0.0
        self.PositionTarget = 0.0
        
        
        self.HalfLength = 50.0

        self.max_engine = 1.0
        self.slider = 00.0
        self.logpath = ''

        self.counter = 0
        self.reset = 0
        self.isrunning = 0
        self.stop = 0
        self.mode = 0
        self.real_time = 1

        




        #### Draw Figure
        self.canvas = FigureCanvas(Figure(figsize = (30,20)))
        self.canvas.AxCart = self.canvas.figure.add_subplot(211)
        self.canvas.AxSlider = self.canvas.figure.add_subplot(212)
        
        draw_all(self.canvas,
                 self.CartPosition,
                 self.angle,
                 self.slider,
                 self.max_engine,
                 self.HalfLength,
                 self.arr_it,
                 self.mode)
    
        self.setGeometry(300, 300, 2500, 1500)
        
        
        layout = QVBoxLayout()
        
        # Create layout for Matplotlib figures only
        lf = QVBoxLayout()
        lf.addWidget(self.canvas)
        
        #Radiobuttons to toggle the mode of operation
        lr = QVBoxLayout()
        self.rb_manual = QRadioButton('Manual Stabilization')
        self.rb_PID = QRadioButton('PID-control with adjustable position')
        self.rb_ItalianCart = QRadioButton('ItalianCart')
        self.rb_manual.toggled.connect(self.onClicked)
        self.rb_PID.toggled.connect(self.onClicked)
        self.rb_ItalianCart.toggled.connect(self.onClicked)
        lr.addStretch(1)
        lr.addWidget(self.rb_manual)
        lr.addWidget(self.rb_PID)
        lr.addWidget(self.rb_ItalianCart)
        lr.addStretch(1)
        self.rb_manual.setChecked(True)
        
        # Create main part of the layout for Figures and radiobuttons
        # And add it to the whole layout
        lm = QHBoxLayout()
        lm.addLayout(lf)
        lm.addLayout(lr)
        layout.addLayout(lm)
        
        #Displays of current relevant values
        ld = QHBoxLayout()
        self.labt = QLabel("Time (s): ")
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()
        self.labs = QLabel('Speed (m/s):')
        self.laba = QLabel('Angle (deg):')
        ld.addWidget(self.labt)
        ld.addWidget(self.labs)
        ld.addWidget(self.laba)
        layout.addLayout(ld)
        
        #Buttons "START/STOP", "RESET", "QUIT"
        bss = QPushButton("START!/STOP!")
        bss.pressed.connect(self.play)
        br = QPushButton("RESET")
        br.pressed.connect(self.reset_button)
        bq = QPushButton("QUIT")
        bq.pressed.connect(QApplication.quit)
        lb = QVBoxLayout() #Layout buttons
        lb.addWidget(bss) 
        lb.addWidget(br) 
        lb.addWidget(bq) 
        layout.addLayout(lb)
        
        #Checkbox to swich between real time and constant dt simulation
        cb = QCheckBox('Real time simulation', self)
        cb.toggle()
        cb.stateChanged.connect(self.real_time_simulation_f)
        layout.addWidget(cb)

        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
         
        self.show()
        self.setWindowTitle('CartPole')
        #This line serves multithreading:
        # or the smooth functioning of the app,
        # the calculations and redrawing of the figures have to be done
        # in a different thread then the one cupturing the mouse position
        self.threadpool = QThreadPool()
        
        #### This line and the following on_move function cupture mouse position
        self.canvas.mpl_connect("motion_notify_event", self.on_move)

    def on_move(self, event):
        if event.xdata == None or event.ydata == None:
            pass
        else:
            if event.inaxes  == self.canvas.AxSlider:
                
                if self.mode == 0 or self.mode == 2:                 
                    if event.xdata>self.max_engine:
                        self.slider = self.max_engine
                    elif event.xdata<-self.max_engine:
                        self.slider = -self.max_engine
                    else:
                        self.slider = event.xdata
                elif self.mode == 1:
                    if event.xdata>self.HalfLength:
                        self.slider = self.HalfLength
                    elif event.xdata<-self.HalfLength:
                        self.slider = -self.HalfLength
                    else:
                        self.slider = event.xdata
    
    
    def reset_variables(self):

        self.CartPosition = 0.0
        self.CartPositionD = 0.0
        self.CartPositionDD = 0.0
        self.angle = random.random()  * pi/180.0 #you can give angle in deg
        self.angleD = 0.0
        self.angleDD = 0.0
        self.time = 0.0
        
        self.counter = 0
        self.slider = 0
    
    def thread_calculations(self):
        
            # Make folders
        try:
            os.makedirs('save')
        except:
            pass
        
        # Training Log
        dict_history = {}
        dict_history['Time'] = [0]
        dict_history['deltaTimeMs'] = [0]
        dict_history['position'] = []
        dict_history['positionD'] = []
        dict_history['positionDD'] = []
        dict_history['angle'] = []
        dict_history['angleD'] = []
        dict_history['angleDD'] = []
        dict_history['motor'] = []
        dict_history['PositionTarget'] = []
        
        start_global = time()
        while(1):

            start = time()
            
            self.CartPosition, self.CartPositionD, self.CartPositionDD, \
                self.angle, self.angleD, self.angleDD,\
                        self.ueff, self.slider, \
                        = Calculations_real(self.CartPosition,
                                                        self.CartPositionD,
                                                        self.angle,
                                                        self.angleD,
                                                        self.slider,
                                                        self.dt,
                                                        self.mode)
            
            if self.mode == 2:
                self.angle = 0
                
            
            sleep(0.001)
            
            stop = time()
            if self.real_time == 1:
                self.dt = stop-start
                self.time_global = stop-start_global
            else:
                self.dt = 0.01
                self.time_global = self.time_global+self.dt
            
            
            
            # Saving simulation data
            dict_history['Time'].append(around(self.time_global, 4))
            dict_history['deltaTimeMs'].append(around(self.dt*1000.0,3))
            dict_history['position'].append(around(self.CartPosition,3))
            dict_history['positionD'].append(around(self.CartPositionD,4))
            dict_history['positionDD'].append(around(self.CartPositionDD,4))
            dict_history['angle'].append(around(self.angle,4))
            dict_history['angleD'].append(around(self.angleD,4))
            dict_history['angleDD'].append(around(self.angleDD,4))
            dict_history['motor'].append(around(self.ueff,4))
            if self.mode == 1:
                self.PositionTarget = self.slider
            elif self.mode == 0 or self.mode == 2:
                self.PositionTarget = 0.0
            dict_history['PositionTarget'].append(around(self.PositionTarget,4))
            
            
            
        
            if abs(self.angle)>pi/2 or self.isrunning == 0:
                self.isrunning = 0
                break
            
            myreset = 0
            if self.reset == 1:
                self.reset_variables()
                self.isrunning = 0
                self.reset = 0
                myreset = 1
            
            if myreset == 1:
                break
        
        
        del dict_history['Time'][-1]
        del dict_history['deltaTimeMs'][-1]
        self.logpath = './save/'+str(datetime.now().strftime('%Y-%m-%d_%H%M%S'))+'.csv'
        # Write Log File
        with open(self.logpath, "w") as outfile:
           writer = csv.writer(outfile)
           writer.writerow(dict_history.keys())
           writer.writerows(zip(*dict_history.values()))
            
        return
        
    
    def thread_drawing(self):
        while(self.reset == 0 and self.isrunning == 1):
                #Draw
                draw_all(self.canvas,
                         self.CartPosition,
                         self.angle,
                         self.slider,
                         self.max_engine,
                         self.HalfLength,
                         self.arr_it,
                         self.mode)
                self.labs.setText("Speed (m/s): " + str(around(self.CartPositionD,2)))
                self.laba.setText("Angle (deg): " + str(around(self.angle*360/(2*pi),2)))
                sleep(0.1)
                
            

    
    def play(self):
        
        
        if self.isrunning == 1:
            self.isrunning = 0
        elif self.isrunning == 0:
            self.isrunning = 1
            # Pass the function to execute
            worker_calculations = Worker(self.thread_calculations)
            worker_drawing = Worker(self.thread_drawing)
            # Execute
            self.threadpool.start(worker_calculations)
            self.threadpool.start(worker_drawing) 

        
    def reset_button(self):
        if self.isrunning == 1:
            self.reset = 1
        elif self.isrunning == 0:
            self.reset_variables()
                        #Draw
            draw_all(self.canvas,
                     self.CartPosition,
                     self.angle,
                     self.slider,
                     self.max_engine,
                     self.HalfLength,
                     self.arr_it,
                     self.mode)
            
            
            self.labs.setText("Speed (m/s): " + str(around(self.CartPositionD,2)))
            self.laba.setText("Angle (deg): " + str(around(-self.angle*360/(2*pi),2)))
            
        
 
        


        
    def recurring_timer(self):
        if self.isrunning == 1: 
            self.counter +=1
            self.labt.setText("Time (s): " + str(float(self.counter)/10.0))
    
    
        
    
    def onClicked(self):
        if self.rb_manual.isChecked():
            self.mode = 0
        elif self.rb_PID.isChecked():
            self.mode = 1
        elif self.rb_ItalianCart.isChecked():
            self.mode = 2
            
        self.slider = 0
        
        draw_all(self.canvas,
                    self.CartPosition,
                    self.angle,
                    self.slider,
                    self.max_engine,
                    self.HalfLength,
                    self.arr_it,
                    self.mode)
        
        
    def real_time_simulation_f(self, state):
        if state == Qt.Checked:
            self.real_time = 1
        else:
            self.real_time = 0


if __name__ == "__main__":
    
    def run_app():
        app = QApplication(sys.argv)
        #set the default icon to use for the windows
        app.setWindowIcon(QIcon('myicon.ico'))
        window = MainWindow()
        window.show()
        app.exec_()
    run_app()