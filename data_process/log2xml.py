import re
import os
import sys
import time

def proc(gr):
	return '<Data x=\''+ str(("%.12f" % (float(gr.group(1))))) + '\' y=\'' + str(("%.12f" % (float(gr.group(2))))) + '\' z=\'' + str(("%.12f" % (float(gr.group(3))))) + '\' timestamp=\'' + str(gr.group(4)) +'\'/>'

'''
def proc2(gr):
	print(float(gr.group(1)))
	print(float(gr.group(2)))
	print(float(gr.group(3)))
	print(gr.group(4))
	print("%.12f" % (float(gr.group(1))))
	return '<Data x=\''+ str(("%.12f" % (float(gr.group(1))))) + '\' y=\'' + str(("%.12f" % (float(gr.group(2))))) + '\' z=\'' + str(float(gr.group(3))) + '\' timestamp=\'' + str(gr.group(4)) +'\'/>'
	
line = "02-12 09:04:56.965  5302  5333 D QSensorTest: GYRO:[0]:6.1035156E-5,[1]:0.0103302,[2]:-0.0137786865,acc:3,ts:265288617411196"
abcd = "02-12 09:04:56.962  5302  5333 D QSensorTest: GYRO:[0]:-0.014419556,[1]:0.00819397,[2]:0.017105103,acc:3,ts:265288614878237"
x=re.sub(r'^.*QSensorTest: GYRO:\[0\]:(.*),\[1\]:(.*),\[2\]:([\d\.\-E]{1,20}),.*ts:(.*)$', proc2, line)
print x
 

'''
fi = open(sys.argv[1],"r")
fc = open("cal_gyro.xml", "w")
fu = open("uncal_gyro.xml", "w")

fc.write("<?xml version='1.0' encoding='UTF-8'?>\n<Sequence>\n<Dataset>\n")
fu.write("<?xml version='1.0' encoding='UTF-8'?>\n<Sequence>\n<Dataset>\n")

line = fi.readline()

while line:
	matchobj = re.search(r'UNCAL GYRO',line)
	if matchobj:
		x = re.sub(r'^.*UNCAL GYRO:\[0\]:(.*),\[1\]:(.*),\[2\]:([\d\.\-E]*),.*ts:(.*)$', proc, line)
		fu.write(x)
		
	matchobj = re.search(r'QSensorTest: GYRO',line)
	if matchobj:
		x = re.sub(r'^.*QSensorTest: GYRO:\[0\]:(.*),\[1\]:(.*),\[2\]:([\d\.\-E]*),.*ts:(.*)$', proc, line)
		fc.write(x)
	
	line = fi.readline()

fc.write("</Dataset>\n</Sequence>\n")
fu.write("</Dataset>\n</Sequence>\n")
fc.close()
fu.close()
fi.close()
