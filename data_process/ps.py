import os
import sys
import time
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.animation as animation


def fileparse(file1):
    labels = ['x','y','z','timestamp']
    with open(file1,"r") as fp:
        tree = ET.parse(fp)
        root = tree.getroot()
        x_list = []
        y_list = []
        z_list = []
        timestamp = []
        freq = []
        for item in root.findall('Dataset'):
            for child in item:
                x_list.append(float(child.attrib.get('x')))
                y_list.append(float(child.attrib.get('y')))
                z_list.append(float(child.attrib.get('z')))
                timestamp.append(int(child.attrib.get('timestamp')))
        #print x_list, y_list, z_list, timestamp
        df = pd.DataFrame({'x':x_list, 'y':y_list, 'z':z_list, 'timestamp':timestamp})
        freq.append(1/((df['timestamp']-df['timestamp'].shift(1))/1000000)*1000)
        return freq, x_list, y_list, z_list, timestamp

def fileparsepose(file1):
    labels = ['qx','qy','qz','qw','timestamp']
    with open(file1,"r") as fp:
        tree = ET.parse(fp)
        root = tree.getroot()
        x_list = []
        y_list = []
        z_list = []
        qx_list = []
        qy_list = []
        qz_list = []
        qw_list = []
        timestamp = []
        freq = []
        for item in root.findall('Dataset'):
            for child in item:
                qx_list.append(float(child.attrib.get('qx')))
                qy_list.append(float(child.attrib.get('qy')))
                qz_list.append(float(child.attrib.get('qz')))
                qw_list.append(float(child.attrib.get('qw')))
                timestamp.append(int(child.attrib.get('timestamp')))
        #print x_list, y_list, z_list, timestamp
        #not conside the roll = 0 condition.
        
        qx = np.array(qx_list)
        qy = np.array(qy_list)
        qz = np.array(qz_list)
        qw = np.array(qw_list)
 
        x_list = np.arctan2(2*(qy*qz+qw*qx),qw*qw-qx*qx-qy*qy+qz*qz)*180/np.pi
        y_list = np.arcsin(2*(qw*qy-qx*qz))*180/np.pi
        z_list = np.arctan2(2*(qx*qy+qw*qz),qw*qw+qx*qx-qy*qy-qz*qz)*180/np.pi
        
        df = pd.DataFrame({'x':x_list, 'y':y_list, 'z':z_list, 'timestamp':timestamp})
        freq.append(1/((df['timestamp']-df['timestamp'].shift(1))/1000000)*1000)
        return freq, x_list, y_list, z_list, timestamp
        
        
def plotgraphs_timestamp(title, data):
    #Plotting the graph
    plt.xlabel("No of Samples")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.plot(data)
    plt.legend()
    plt.draw()
    plt.savefig(title+".png")
    plt.clf()
    #plt.show(block=False)
    return 0
 
def plotgraphs_data(title, x, y, z,ts):
    #Plotting the graph
    plt.title(title)
    plt.xlabel("Time")
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot 
# https://blog.csdn.net/xiaotao_1/article/details/79100163
#    plt.plot(ts,x,'r,', label="x-axis")
#    plt.plot(ts,y,'g,', label="y-axis")
#    plt.plot(ts,z,'b,', label="z-axis")
    plt.plot(ts,x,'r*', label="x-axis")
    plt.plot(ts,y,'gd', label="y-axis")
    plt.plot(ts,z,'bx', label="z-axis")

    plt.legend()
    plt.draw()
    plt.savefig(title+".png")
    plt.clf()
    print "X AVG: {}".format(sum(x)/len(x))
    print "X max: {}".format(max(x))
    print "X min: {}".format(min(x))
    print "X rng: {}".format(max(x)-min(x))
    print "X std: {}".format(np.std(x))
    print ""
    
    print "Y AVG: {}".format(sum(y)/len(y))
    print "Y max: {}".format(max(y))
    print "Y min: {}".format(min(y))
    print "Y rng: {}".format(max(y)-min(y))
    print "Y std: {}".format(np.std(y))
    print ""
    
    print "Z AVG: {}".format(sum(z)/len(z))
    print "Z max: {}".format(max(z))
    print "Z min: {}".format(min(z))
    print "Z rng: {}".format(max(z)-min(z))
    print "Z std: {}".format(np.std(z))
    print ""

    plt.title(title+"_accumulate")
    plt.xlabel("Time")
    x1=np.add.accumulate(x)
    y1=np.add.accumulate(y)
    z1=np.add.accumulate(z)
    plt.plot(ts,x1, label="x-axis")
    plt.plot(ts,y1, label="y-axis")
    plt.plot(ts,z1, label="z-axis")
    plt.legend()
    plt.draw()
    plt.savefig(title+"_accumulate.png")
    plt.clf()

    return 0

def plotgraphs_data_2d(title, x, y):
    #Plotting the graph
    plt.title(title)

    plt.plot(x,y)


    plt.legend()
    plt.draw()
    plt.savefig(title+".png")
    plt.clf()
    return 0


def plotgraphs_datapose(title, x, y, z,ts):
    #Plotting the graph
    plt.title(title)
    plt.xlabel("Time")
    plt.plot(ts,x, label="x-axis")
    plt.plot(ts,y, label="y-axis")
    plt.plot(ts,z, label="z-axis")
    plt.legend()
    plt.draw()
    plt.savefig(title+".png")
    plt.clf()
    print "X AVG: {}".format(sum(x)/len(x))
    print "X max: {}".format(max(x))
    print "X min: {}".format(min(x))
    print "X rng: {}".format(max(x)-min(x))
    print "X std: {}".format(np.std(x))
    print ""
    
    print "Y AVG: {}".format(sum(y)/len(y))
    print "Y max: {}".format(max(y))
    print "Y min: {}".format(min(y))
    print "Y rng: {}".format(max(y)-min(y))
    print "Y std: {}".format(np.std(y))
    print ""
    
    print "Z AVG: {}".format(sum(z)/len(z))
    print "Z max: {}".format(max(z))
    print "Z min: {}".format(min(z))
    print "Z rng: {}".format(max(z)-min(z))
    print "Z std: {}".format(np.std(z))
    print ""

    return 0

xxd,yyd = [],[]

def init(): 
    # creating an empty plot/frame 
    line.set_data([], []) 
    return line,

xsd,ysd = [],[]
def ghostImage(x,y):
    xsd.append(x)
    ysd.append(y)
    if len(xsd)>60:
        del xsd[0]
        del ysd[0]
    return xsd,ysd
    
def animate(i): 
    print "i is ", i
    x = xxd[i*50]
    y = yyd[i*50]
    # appending new points to x, y axes points list 
    line.set_data(ghostImage(x,y)) 
    return line, 

    
if len(sys.argv) <2:
    print "Usage: python ps.py <directory location where all sensor data accelerometer.xml, gyroscope.xml and headTrackingPose.xml exist>"
    exit
elif sys.argv[1] == "-h" or sys.argv[1] == "-help":
    print "Usage: python ps.py <directory location where all sensor data accelerometer.xml, gyroscope.xml and headTrackingPose.xml exist>"
else: 
    directory = sys.argv[1]
    os.chdir(directory)
    files = glob.glob("*.xml")
    for fname in files:
        print fname
        if fname == "cal_gyro.xml":
            data, xdata, ydata, zdata,timestamp = fileparse(fname)
            plotgraphs_data("cal gyro data", xdata, ydata, zdata,timestamp)
            print "\n"
            

        if fname == "uncal_gyro.xml":
            data, xdata, ydata, zdata,timestamp = fileparse(fname)
            plotgraphs_data("uncal Gyroscope data", xdata, ydata, zdata,timestamp)
            print "\n"

        if fname == "headTrackingPose.xml":
            fr, xxd, yyd, zdata, timestamp = fileparsepose(fname)
            #plotgraphs_data_2d("angle",xdata,ydata)
            print "\n"
            plt.style.use('dark_background')
            fig = plt.figure() 
            ax = plt.axes(xlim=(-80, 10), ylim=(-50, 10)) 
            line, = ax.plot([], [], lw=2)
            flen=len(xxd)/50
            anim = animation.FuncAnimation(fig, animate, init_func=init, frames=flen, interval=20, blit=True) 
            # save the animation as gif file 
            anim.save('figure.gif',writer='imagemagick')
        
    
