import numpy as np
# BP
#debug tips
#python -m pdb myscript.py
#h b n p

# 非线性函数，逻辑回归，激活函数，sigmoid function
def nonlin(x,deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

# 四组样本，每组三个输入 X1 X2 X3
X = np.array([ [1,1,0],
				[1,0,1],
				[0,0,1],
				[1,1,1] ])				
# 输出Y，4个输出结果
y = np.array([[0],[0],[1],[1]])
# 随机种子初始化
np.random.seed(1)
# Y=aX1+bX2+cX3, a,b,c为权重矩阵，随机初始化权重矩阵
syn0 = 2*np.random.random((3,1)) - 1
print syn0
for iter in xrange(10):
    # 输入
	l0 = X
	#点乘 输入X 和权重abc，
	#然后将结果做非线性转换
	l1 = nonlin(np.dot(l0,syn0))
	#计算期望值和l1【现有权重得出的结果】之间的差值
	l1_error = y - l1
    # 根据差值做回归， 跟期望值差的越大，调整越大
	# 求出归一化的调整值
	l1_delta = l1_error * nonlin(l1,True)

    # 更新权重
	#根据原输入和归一化的调整值 计算出
	# 现在的调整值，然后叠加在原权重上，作为下次的权重值
	syn0 += np.dot(l0.T,l1_delta)
	print syn0

print "Output After Training: l1"
print l1
print "weights:"
print syn0

