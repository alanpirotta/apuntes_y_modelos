- variable = [i**2 for i in range(10)]: ejemplo de forma de escribir un loop en una línea (list comprehension)

- max(lista_de_respuestas, key= condicion_generadora_de_respuesta): Sirve para averiguar el índice que tiene el máximo de un diccionario (poniendo por ej: key= lambda k: dict[k] o key= dict.get) o el máximo de ocurrencias en una lista (con key=lista.count)). 

- !wget: Trae datos desde un servidor en la web, sin descargar el archivo. Ej: !wget https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip
- !unzip: Descomprime un archivo. SE puede descomprimir el archivo traído con !wget
- os.path.join(path1, path2): Une dos path, poniendo un / en el medio. Sirve para acceder a una carpeta dentro de una carpeta
- for root, directory, files in os.walk(train_dir): recorre un path y todos los subdirectorios dentro, devolviendo la carpeta root, los directorios dentro (en primer nivel) y los nombre de archivos

- Para pasar/mover archivos de una carpeta a otra:
    import os
    import shutil

    source = 'proyectosCortosPython/a_mover'
    destination = 'proyectosCortosPython/'

    files = os.listdir(source)
    for file in files:
        file_name = os.path.join(source, file)
        shutil.move(file_name, destination)
    print("Files Moved")
- Si no se tiene un subdirectorio, se puede usar el root para poner en from_directory, seleccionando en classes=[] la carpeta que contiene los archivos a testear

- pwd : Muestra el directorio donde está ubicado el notebook

- Instalación de la API de Kaggle
    https://github.com/Kaggle/kaggle-api

    pip install kaggle o conda install -c conda-forge kaggle
    poner "kaggle" en consola luego de instalado, y se generará la carpeta .kaggle, donde se pone el token (archivo .json) sacado del usuario de kaggle.

    ejemplo de uso:
    !kaggle competitions list : Lista todas las competiciones
    Se puede sacar el código de cad dataset o competición para descargar directo.

- Descomprimir archivos descargados:
    import zipfile
    import os
    with zipfile.ZipFile("archivo.zip", "r") as zip_ref:                # La r es de "read"
        zip_ref.extractall(r"ubicación donde descomprimir")
    os.remove("archivo.zip")

- Para comprobar que un dataset ya está descargado:
```python
import pathlib
file = pathlib.Path("folder/file")
if file.exists():
    print('Files already exists')
else:
    print('Creating the file')
```

- %%time : Agregarlo a cada celda al ppio para chequear el tiempo que tarda en correr