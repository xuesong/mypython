import numpy as np

x=100


stri= repr(x) +'+'

print(stri)


for i in range(1,3):
	tt = np.random.randint(0,4)
	if tt == 0:
		x = np.random.randint(0,997) + 3
		y = np.random.randint(0,997) + 3
		z = x+y
		strp = repr(x) + ' + ' + repr(y) + ' = '

	elif tt == 1:
		x = np.random.randint(0,1001)
		z = np.random.randint(0,1001)
		y = x+z
		strp= repr(y) + ' - ' + repr(x) + ' = '
	
	elif tt == 2:
		x = np.random.randint(0,9) + 11
		y = np.random.randint(0,18) + 2
		z = x*y
		strp = repr(x) + ' x ' + repr(y) + ' = '

	elif tt == 3:
		x = np.random.randint(0,997) + 3
		y = np.random.randint(0,997) + 3
		z = x+y
		strp=  repr(x) + ' + ' + repr(y) + ' = '


	answ = int(input(strp))

	if answ == z:
		print('bingo')
	else:
		print('answer is %d' %(z))

