## Antes de empezar

Descarga el dataset de BBC que contiene 2225 documentos categorizados por la BBC correspondientes al año 2004-2005 acá: http://mlg.ucd.ie/datasets/bbc.html

Cambia la línea 14 por la ruta del directorio del dataset descargado. 

## Explicación

Categorizador de textos (NLP [Procesamiento de Lenguaje Natural]) para 5 categorías:

* Business
* Entertainment
* Politics 
* Sports
* Tech

Explicación:

1. Creamos el dataset. 
  Introducimos en un solo archivo, el tipo de categoría, el nombre de archivo y el texto del archivo.

2. A partir de ese archivo, creamos una tupla de:
  (SECCIÓN, TEXTO)

3. Limpiamos y tokenizamos. Esto es: limpiar el dataset de palabras que no aportan para la categorización de textos. Palabras, como "El", "La", "Los" y todos los conectores son limpiados del dataset. Una vez hemos limpiado el dataset, lo tokenizamos (palabras => numeros)

4. Obtenemos la frecuencia de distribución (básicamente registrar la frecuencia de cada tipo de palabra en el texto). En la función imprimimos las 20 palabras más comunes para verificar que no existen palabras que deberíamos filtrar antes de crear el embedding. 

5. A partir del dataset, cogemos el 80% del texto para entrenarlo y el 100% para probar que la clasificación sea exitosa. 

6. Creamos un vector de palabras en inglés a partir del 80% dataset que recuenta las palabras.

7. Utilizamos el algoritmo o clasificador de Naive Bayes para crear el modelo. Naives Bayes es un clasificador probabilistico que calcula la probabilidad de que un texto pertenezca a una clase o sección de acuerdo a sus características. En este caso la sección es la categorización del dataset, y sus características son las palabras limpias de cada elemento. 

8. Evaluamos el clasificador de Naives Bayes con los datos de prueba (el 100% de nuestro dataset). 

9. Guardamos el clasificador entrenado y el vectorizado del dataset en archivos pickle separados.

10. Utilizamos el vectorizador y el clasificador para: vectorizar el nuevo texto que queremos clasificar con las mismas reglas que utilizamos para crear el embedding y utilizar el clasificador con ese vectorizado. El clasificador o embedding te da una probabilidad de que ese texto pertenezca a las secciones, normalmente cogemos la primera ya que es la más probable. 




