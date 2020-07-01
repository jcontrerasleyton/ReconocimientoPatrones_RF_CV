from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import KFold
import numpy as np
import sys
import os

#--------------------------------------------------------------------------------------------------

def folders(path):
	try: os.stat(path)
	except: os.mkdir(path)

#--------------------------------------------------------------------------------------------------

def clss(mode, tipo, filter_mode, tree):
	for fold in lista:
		print('\n'+tipo+'---'+fold)
		print("Trees;Acc(cv);Acc(std);Acc")

		matrix_train = np.loadtxt(folder+fold+tipo+'-train.csv', delimiter=',')

		if mode == 0:
			matrix_test = np.loadtxt(folder+fold+tipo+'-test.csv', delimiter=',')
		else:
			matrix_test = np.loadtxt(folder2+tipo+'.csv', delimiter=',')

		col = int(matrix_train[0].size-1)
		y_train, y_test = matrix_train[:,col], matrix_test[:,col]
		y_train, y_test = y_train.astype(int), y_test.astype(int) 
		X_train, X_test = np.delete(matrix_train, col, 1), np.delete(matrix_test, col, 1)

		kf = KFold(n_splits=5)

		if filter_mode == 0:
			tree = 100
			tree_max = 1001
		else:
			tree_max = tree+1

		for trees in range(tree, tree_max, 100):
			scores = list()

			#Inicializar Random Forest
			clf = RandomForestClassifier(n_estimators=trees, n_jobs=-1)

			#Realizar cross_validation
			for train_index, test_index in kf.split(X_train):
				#Entrenamiento
				clf.fit(X_train[train_index], y_train[train_index])
				#Testing y obtener Accuracy
				scores.append(clf.score(X_train[test_index], y_train[test_index]))
		
			scores = np.asarray(scores)
			#Lista de Accuracy y promedio +- desviacion estandar
			#print(str(scores)+", "+str(scores.mean())+"+-"+str(scores.std())) 

			#Volver a entrenar con conjunto train
			clf.fit(X_train, y_train)

			#Realizar testing con conjunto test, obteniendo su Accuracy
			#print('Trees: '+str(trees)+', Accuracy: '+str(clf.score(X_test,y_test)))

			print(str(trees)+';'+str(scores.mean())+';'+str(scores.std())+';'+str(clf.score(X_test,y_test)))

			#y_pred = clf.predict(X_test)
			#print('Precision-Score = '+str(precision_score(y_test, y_pred, average=None)))
			#print('Precision-Score (mean) = '+str(precision_score(y_test, y_pred, average='macro')))
			#print('Recall-Score = '+str(recall_score(y_test, y_pred, average=None)))
			#print('Recall-Score (mean) = '+str(recall_score(y_test, y_pred, average='macro')))
			#print('F1-Score = '+str(f1_score(y_test, y_pred, average=None)))
			#print('F1-Score (mean) = '+str(f1_score(y_test, y_pred, average='macro')))
			#print('Confusion Matrix = '+str(confusion_matrix(y_test, y_pred)))
			#tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
			#print('TN '+str(tn)+', FP '+str(fp)+', FN '+str(fn)+', TP '+str(tp))

		#print('') 
	#print('----------------------------------') 

#--------------------------------------------------------------------------------------------------

path = sys.argv[1].split("/")[0]
path2 = sys.argv[2].split("/")[0]
filter_mode = int(sys.argv[3])

if filter_mode == 0:

	folder = path+"/results2/"
	folder2 = path2+"/results2/"
	folders(folder+'70-30/')
	folders(folder+'60-40/')
	folders(folder+'50-50/')
	
	lista = ['70-30/','60-40/','50-50/']
	lista2 = ['int','ulbp','hog','all','edge','sharp','emboss','imp']

	for line in lista2: 
		print(line+"\n")
		clss(0,line,0,0)
		clss(1,line,0,0)
		print("")

elif filter_mode == 1:

	folder = path+"/results3/"
	folder2 = path2+"/results3/"

	#part = input("Mejor particion: \n --(1) 70-30 \n --(2) 60-40 \n --(3) 50-50 \n\n :")
	part = 2
	
	if part == 1:
		lista = ['70-30/']
		folders(folder+'70-30/')
	elif part == 2:
		lista = ['60-40/']
		folders(folder+'60-40/')
	elif part == 3:
		lista = ['50-50/']
		folders(folder+'50-50/')
	else:
		print('ERROR al ingresar particion')
		sys.exit()		

	#tree = input("Mejor arbol: ")
	tree = 500

	file_list = open("./bsif/list.dat", "r") 
	for line in file_list: 
		print(line+"\n")
		clss(0,line.rstrip(),1,tree)
		clss(1,line.rstrip(),1,tree)
		print("")

else:
	print('ERROR ingresar filtro 0 o 1')
	sys.exit()
