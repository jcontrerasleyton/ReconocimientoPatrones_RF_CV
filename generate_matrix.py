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

	return np.asarray(H)

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

			#Obtener vectores
			image_int = intensity(gray_resize)
			image_ulbp = ulbp(gray_resize, radius)
			image_hog = hog(gray_resize, pcell, cblock)

			#Generar vector general
			image_all = np.append(image_int, image_ulbp)
			image_all = np.append(image_all, image_hog)

			#Agregar genero
			image_int = np.append(image_int, id_gender)
			image_ulbp = np.append(image_ulbp, id_gender)
			image_hog = np.append(image_hog, id_gender)
			image_all = np.append(image_all, id_gender)

			if first:
				m_int = np.asmatrix(image_int) #matriz intensity
				m_ulbp = np.asmatrix(image_ulbp) #matriz ulbp
				m_hog = np.asmatrix(image_hog) #matriz hog
				m_all = np.asmatrix(image_all) #matriz all

				first = False
			else:
				#matriz intensity
				m_int_aux = np.asmatrix(image_int) 
				m_int = np.concatenate((m_int,m_int_aux), axis=0) 

				#matriz ulbp
				m_ulbp_aux = np.asmatrix(image_ulbp) 
				m_ulbp = np.concatenate((m_ulbp,m_ulbp_aux), axis=0) 

				#matriz hog
				m_hog_aux = np.asmatrix(image_hog) 
				m_hog = np.concatenate((m_hog,m_hog_aux), axis=0) 

				#matriz all
				m_all_aux = np.asmatrix(image_all) 
				m_all = np.concatenate((m_all,m_all_aux), axis=0) 

	return (m_int, m_ulbp, m_hog, m_all)

#--------------------------------------------------------------------------------------------------

def write():
	np.savetxt(folder_result+'int.csv', m_int, delimiter=',',fmt='%f')
	np.savetxt(folder_result+'ulbp.csv', m_ulbp, delimiter=',',fmt='%f')
	np.savetxt(folder_result+'hog.csv', m_hog, delimiter=',',fmt='%f')
	np.savetxt(folder_result+'all.csv', m_all, delimiter=',',fmt='%f')
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
(m_int, m_ulbp, m_hog, m_all) = cargar_data()
print "Carga satisfactoria\n"

write()
get_gender_vector()
print "\nArchivos generados exitosamente\n"
