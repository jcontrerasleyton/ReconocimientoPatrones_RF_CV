## Clasificación de Base de Datos de imágenes perioculares con Random Forest y Cross Validation

Cada uno de los codigos deben estar fuera de las carpetas de bases de datos.

----------------------------------------------------------------------------------------

Para generar los vectores de intensidad, textura y forma (además de vectores de mejora 
edge, sharp, emboss y la concatenación de todos), ejecutar generate_matrix.py

pyhon generate_matrix.py data_path info_path

E.g. pyhon generate_matrix.py Csip/csip_data Csip/csip_info
E.g. pyhon generate_matrix.py Test2/Test2_data Test2/Test2_info

Resultados se guardan en (results2/)

----------------------------------------------------------------------------------------

Para aplicar algoritmo bsif, ejecutar generate_filters.m

Ingresar path

E.g. Csip/csip_data
E.g. Test2/Test2_data

Resultados se guardan en (results3/)

----------------------------------------------------------------------------------------

Para separar los vectores en particiones 70/30 - 60/40 - 50/50, ejecutar train_test.py

pyhon train_test.py path mode

E.g. pyhon train_test.py Csip 0
Dentro de la carpeta results2/ se guardaran los resultados (carpetas 70-30/ 60-40/ 50-50/).

E.g. pyhon train_test.py Csip 1
Dentro de la carpeta results3/ se guardaran los resultados  de bsif (carpetas 70-30/ 60-40/ 50-50/).

----------------------------------------------------------------------------------------

Para evaluar las particiones, ejecutar random_forest.py

pyhon random_forest.py path_a path_b mode

E.g. pyhon random_forest.py Csip Test2 0

E.g. pyhon random_forest.py Csip Test2 1
Ingresar mejor partición y mejor árbol



