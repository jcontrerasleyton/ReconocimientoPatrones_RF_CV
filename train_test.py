import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split

#--------------------------------------------------------------------------------------------------

def folders(path):
	try: os.stat(path)
	except: os.mkdir(path)

#--------------------------------------------------------------------------------------------------

def write(X, y, name, ts):
	#Separar conjunto de datos en train y test
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ts, random_state=42)

	#Union de matriz con vector de etiquetas
	train = np.column_stack((X_train,y_train))
	test = np.column_stack((X_test,y_test))

	if ts == 0.7: fol = folder_1+name
	if ts == 0.6: fol = folder_2+name
	if ts == 0.5: fol = folder_3+name

	#Escribir archivos resultantes en sus respectivas carpetas
	np.savetxt(fol+'-train.csv', train, delimiter=',', fmt='%f')
	np.savetxt(fol+'-test.csv', test, delimiter=',', fmt='%f')

	#Vectores de etiquetas para los train y test
	#np.savetxt(fol+'-gc-train.dat', y_train, delimiter=' ', fmt='%d')
	#np.savetxt(fol+'-gc-test.dat', y_test, delimiter=' ', fmt='%d')

#--------------------------------------------------------------------------------------------------

def get_gender_vector(matrix, name):
	col = int(matrix[0].size-1)

	y = matrix[:,col] #labels
	X = np.delete(matrix, col, 1) #features

	print "\n"+name
	write(X, y, name, 0.7)
	print "--- 70-30"
	write(X, y, name, 0.6)
	print "--- 60-50"
	write(X, y, name, 0.5)
	print "--- 50-50"

#--------------------------------------------------------------------------------------------------

path = sys.argv[1].split("/")[0]
filter_mode = int(sys.argv[2])

if filter_mode == 0:

	folder = path+"/results2/"
	folder_1 = folder+'70-30/'
	folder_2 = folder+'60-40/'
	folder_3 = folder+'50-50/'

	folders(folder_1)
	folders(folder_2)
	folders(folder_3)

	lista = ['int','ulbp','hog','all','edge','sharp','emboss','imp']

	#Cargar matrices
	for line in lista: 
		matrix = np.loadtxt(folder+line+'.csv', delimiter=',')
		print("---Matriz "+line+" cargada")
		get_gender_vector(matrix, line)
		print("---Archivos "+line+" creados\n")

elif filter_mode == 1:

	folder = path+"/results3/"
	folder_1 = folder+'70-30/'
	folder_2 = folder+'60-40/'
	folder_3 = folder+'50-50/'

	folders(folder_1)
	folders(folder_2)
	folders(folder_3)

	#Cargar matrices
	print "\nCargando matrices, expere por favor..."
	file_list = open("./bsif/list.dat", "r") 
	for line in file_list: 
		matrix = np.loadtxt(folder+line.rstrip()+'.csv', delimiter=',')
		print("---Matriz "+line.rstrip()+" cargada")
		get_gender_vector(matrix, line.rstrip())
		print("---Archivos "+line.rstrip()+" creados\n")
		
else:
	print('ERROR ingresar filtro 0 o 1')
	sys.exit()
