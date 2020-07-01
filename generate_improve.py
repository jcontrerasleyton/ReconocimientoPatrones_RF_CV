import cv2
import numpy as np
import sys
import os
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tqdm import tqdm
from skimage import feature
from scipy import signal

#--------------------------------------------------------------------------------------------------

def fix(imnpg):
	#vector = intensity(imnpg)
	vector = ulbp(imnpg, radius)
	#vector = hog(imnpg, pcell, cblock)
	
	return vector

#--------------------------------------------------------------------------------------------------

def mejora(imnpg, w_k):
	w_k = np.rot90(w_k, 2)
	im = signal.convolve2d(imnpg, w_k, 'same')
	
	return im

#--------------------------------------------------------------------------------------------------

def intensity(imnpg):
	im = np.asarray(imnpg).reshape(-1)
	(histo, _) = np.histogram(im, bins=256)

	#Normalization	
	histo = histo.astype("float")
	histo /= (histo.sum()+eps)
	
	return np.asarray(histo)

#--------------------------------------------------------------------------------------------------

def ulbp(imnpg, radius):
	lbp = feature.local_binary_pattern(imnpg, radius*8, radius, method="nri_uniform")
	(histo, _) = np.histogram(lbp.ravel(), bins=59)
	
	#Normalization	
	histo = histo.astype("float")
	histo /= (histo.sum()+eps)

	return np.asarray(histo)

#--------------------------------------------------------------------------------------------------

def hog(imnpg, pcell, cblock):
	(H, hogImage) = feature.hog(imnpg, orientations=9, pixels_per_cell=(pcell, pcell), cells_per_block=(cblock, cblock), block_norm='L2-Hys', transform_sqrt=True, visualise=True)

	#Normalization
	H = H.astype("float")
	H /= (H.sum()+eps)
	
	return H

#--------------------------------------------------------------------------------------------------

def cargar_data():
	dirFiles = os.popen("ls "+folder_data).read()
	total_files = int(os.popen("ls "+folder_data+" | wc -l").read())-1
	#print total_files
	first = True
	id_device = 0
	for filename in tqdm(dirFiles.split(), total = total_files):

		#Generar path
		img_path = os.path.join(folder_data, filename)
		infoname = os.path.splitext(filename)[0]+".xml"
		info_path = os.path.join(folder_info, infoname)

		#Lectura de imagen
		img = cv2.imread(img_path)

		if img is None:
			print ('Cannot open file:', img_path)
			continue

		#Obtener imagen en escala de grises
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		#Lectura xml
		tree = ET.parse(info_path)
		root = tree.getroot()
		user = root.find('user')
		gender = user.find('gender').text
		iris = root.find('iris').text

		if(iris == "right"):

			id_gender = 0
			if(gender == "F"): id_gender = 1 #M:0 F:1 
			
			#Redimensiona imagen para su almacenamiento
			gray_resize = cv2.resize(gray, (240, 180))
	
			edge = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])
			sharp = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
			emboss = np.array([[-2, -1, 0],[-1, 1, 1],[0, 1, 2]])
	
			image_edge = fix(mejora(gray_resize, edge))
			image_sharp = fix(mejora(gray_resize, sharp))
			image_emboss = fix(mejora(gray_resize, emboss))

			#Generar vector general
			image_imp = np.append(image_edge, image_sharp)
			image_imp = np.append(image_imp, image_emboss)

			#Agregar genero
			image_edge = np.append(image_edge, id_gender)
			image_sharp = np.append(image_sharp, id_gender)
			image_emboss = np.append(image_emboss, id_gender)
			image_imp = np.append(image_imp, id_gender)
	
			if first:
				m_edge = np.asmatrix(image_edge) #matriz edge
				m_sharp = np.asmatrix(image_sharp) #matriz sharp
				m_emboss = np.asmatrix(image_emboss) #matriz emboss
				m_imp = np.asmatrix(image_imp) #matriz imp
	
				first = False
			else:	
				#matriz edge
				m_edge_aux = np.asmatrix(image_edge) 
				m_edge = np.concatenate((m_edge,m_edge_aux), axis=0) 
	
				#matriz sharp
				m_sharp_aux = np.asmatrix(image_sharp) 
				m_sharp = np.concatenate((m_sharp,m_sharp_aux), axis=0) 
	
				#matriz emboss
				m_emboss_aux = np.asmatrix(image_emboss) 
				m_emboss = np.concatenate((m_emboss,m_emboss_aux), axis=0) 

				#matriz imp
				m_imp_aux = np.asmatrix(image_imp) 
				m_imp = np.concatenate((m_imp,m_imp_aux), axis=0) 
	
	return (m_edge, m_sharp, m_emboss, m_imp)

#--------------------------------------------------------------------------------------------------

def write():
	np.savetxt(folder_result+'edge.csv', m_edge, delimiter=',',fmt='%f')
	np.savetxt(folder_result+'sharp.csv', m_sharp, delimiter=',',fmt='%f')
	np.savetxt(folder_result+'emboss.csv', m_emboss, delimiter=',',fmt='%f')
	np.savetxt(folder_result+'imp.csv', m_imp, delimiter=',',fmt='%f')

	print "Archivo database creado"

#--------------------------------------------------------------------------------------------------

def get_gender_vector():
	print "\nVector Gender\n"
	col = int(m_int[0].size-1)
	np.savetxt(folder_result+'gender_column.dat', m_int[:,col], delimiter=',',fmt='%d')

#--------------------------------------------------------------------------------------------------

radius = 1
pcell = 8
cblock = 4
eps=1e-7
folder_data = sys.argv[1]
folder_info = sys.argv[2]
folder = folder_data.split("/")
folder_result = folder[0]+"/results2/"

try:
    os.stat(folder_result)
except:
    os.mkdir(folder_result)

print "Cargando datos... Espere por favor!"
(m_edge, m_sharp, m_emboss, m_imp) = cargar_data()
print "Carga satisfactoria\n"

write()
get_gender_vector()
print "\nArchivos generados exitosamente\n"
