import numpy as np
from numpy.linalg.linalg import norm 
alist=[1,2,3,4,5]
narray=np.array([1,2,3,4,5])

print(alist)
print(narray)

print(type(alist))
print(type(narray))

print(narray*3)
print(alist*3)

npmatrix1=np.array([narray,narray,narray])
npmatrix2=np.array([alist,alist,alist])
npmatrix3=np.array([narray,[1,1,1,1],narray])

print(npmatrix1)
print(npmatrix2)
print(npmatrix3)

okmatrix=np.array([[1,2],[3,4]])
print(okmatrix)
print(okmatrix*2)

badmatrix=np.array([[1,2],[3,4],[5,6,7]])
print(badmatrix)
print(badmatrix*2)

result= okmatrix*2 + 1
print(result)

result1=okmatrix+okmatrix
print(result1)

result2=okmatrix-okmatrix
print(result2)

result3=okmatrix*okmatrix
print(result3)

matrix3x2=np.array([[1,2],[3,4],[5,6]])
print('Original matrix 3 x 2')
print(matrix3x2)
print('Transporosed matrix 2 x 3')
print(matrix3x2.T)

nparray=np.array([[1,2,3,4]])
print('Original array')
print(nparray)
print('Transposd array')
print(nparray.T)

nparray1=np.array([[1,2,3,4]])
norm1=np.linalg.norm(nparray1)

nparray2=np.array([[1,2],[3,4]])
norm2=np.linalg.norm(nparray2)

print(norm1)
print(norm2)

nparray3=np.array([[1,1],[2,2],[3,3]])
normByCols=np.linalg.norm(nparray3,axis=0)
normByRows=np.linalg.norm(nparray3,axis=1)
print(normByCols)
print(normByRows)

nparray1=np.array([0,1,2,3])
nparray2=np.array([4,5,6,7])

flavor1=np.dot(nparray1,nparray2)
print(flavor1)
flavor2=np.sum(nparray1*nparray2)
print(flavor2)
flavor3=nparray1 @ nparray2
print(flavor3)

flavor4=0
for a,b in zip(nparray1,nparray2):
    flavor4+= a*b

print(flavor4)

norm1=np.dot(np.array([1,2]),np.array([3,4]))
norm2=np.dot([1,2],[3,4])

print(norm1,'=',norm2)

nparray2=np.array([[1,-1],[2,-2],[3,-3]])

sumByCols=np.sum(nparray2,axis=0)
sumByRows=np.sum(nparray2,axis=1)

print('Sum by Cols: ')
print(sumByCols)
print('Sum by Rows: ')
print(sumByRows)

nparray2=np.array([[1,-1],[2,-2],[3,-3]])

mean=np.mean(nparray2)
meanByCols=np.mean(nparray2,axis=0)
meanByRows=np.mean(nparray2,axis=1)

print('Matrix mean: ')
print(mean)
print('Mean by col: ')
print(meanByCols)
print('Mean by rows: ')
print(meanByRows)

nparray2=np.array([[1,1],[2,2],[3,3]])
nparrayCentered=nparray2-np.mean(nparray2,axis=0)
print("Original matrix")
print(nparray2)
print('Centered by columns matrix')
print(nparrayCentered)
print('New mean by col:')
print(nparrayCentered.mean(axis=0))

nparray2=np.array([[1,3],[2,4],[3,5]])

nparrayCentered=nparray2.T-np.mean(nparray2,axis=1)
nparrayCentered=nparrayCentered.T
print('Original matrix')
print(nparray2)
print('Centered by col matrix')
print(nparrayCentered)

print('New mean by rows')
print(nparrayCentered.mean(axis=1))


#excercise

arr1=np.array([[5,2],[3,7],[8,1]])
arr2=np.array([[4,6,2],[3,7,8]])
print('matrix multiplication')
print(np.dot(arr1,arr2))
print('element wise multiplication')
print(np.multiply(arr1,arr2.T))
print('mean')
print(np.mean(arr1))
print(np.mean(arr2))

