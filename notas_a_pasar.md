**Comienzo este archivo en el día 44. Me pareció una buena forma de rastrear el proceso, tomar notas y poder encontrarlas después**

### A averiguar
Tengo que averiguar bien que significa cuando pongo "add ." en git y me tira esto:
*LF will be replaced by CRLF in ...* 
*The file will have its original line endings in your working directory*


### Orden útil de análisis de datos:
- Read_csv o equivalente con todos los datos para tener un buen df. graficos simples, describe e info para ver como se comportan los datos
- Buscar y rellenar/vaciar valores vacíos. Se pueden rellenar con 0, mean, median, ffill o bfill. También pueden pedirse si es muy importante, o averiguar por qué ocurrió si es un conjunto grande de datos. 
- Buscar errores en valores: valores fuera de rango, duplicados. También usar métodos de str, dt y cat segun tipo de columna.
- Graficacion de datos más complejos con los datos limpios.
- Conclusiones.
- Estética del informe.

### Listado de modelos de Machine Learning y su uso:
- Clasificación:
    - LogisticRegression()
    - Support Vector Machine / SVM
    - k-Nearest Neighbor() / KNN
    - Naïve Bayes
    - RandomForestClassifier

- Regresión:
    - LinearRegression()

- Cluster:
    - NearesNeighbor()


- variable = [i**2 for i in range(10)]: ejemplo de forma de escribir un loop en una línea (list comprehension)





### R1D66



### R1D67


- Formas de hacer una regrasión lineal:
    - slope, intercept, r_value, p_value, stderr = linregress(x=, y=) : Desempaqueta (unpack) los resultados de la regresión en las diferentes variables.
    - reg = linregress(x=, y=): Guarda los datos en una sola variable, se pueden acceder por ejemplo con reg.slope.
- plt.plot(x, reg.intercept + reg.slope*x, 'r', label='fitted'): Para graficar la regresión (se puede hacer en la misma figura que un scatter), utilizando la fórmula en el eje Y. el 'r' indica color rojo. 


### R1D73

- max(lista_de_respuestas, key= condicion_generadora_de_respuesta): Sirve para averiguar el índice que tiene el máximo de un diccionario (poniendo por ej: key= lambda k: dict[k] o key= dict.get) o el máximo de ocurrencias en una lista (con key=lista.count)). 

### R1D74

- !wget: Trae datos desde un servidor en la web, sin descargar el archivo. Ej: !wget https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip
- !unzip: Descomprime un archivo. SE puede descomprimir el archivo traído con !wget
- os.path.join(path1, path2): Une dos path, poniendo un / en el medio. Sirve para acceder a una carpeta dentro de una carpeta
- for root, directory, files in os.walk(train_dir): recorre un path y todos los subdirectorios dentro, devolviendo la carpeta root, los directorios dentro (en primer nivel) y los nombre de archivos,
- image_generator = ImageDataGenerator(rescale=1./255): Generador de imágenes que permite hacer image augmentation, con algunos atributos (rescale sirve para que todas las imágenes tenga el mismo tamaño). Tiene el método flow_from_directory, por ejemplo
- Para hacer la modificación de las imágenes y poder dejarlas aptas para nuestro modelo, desde un directorio. Nota: SIEMPRE espera tener al menos una carpeta dentro del directory (no se pueden tirar las imágenes dentro de la carpeta test directo, por ejemplo)
    image_generator.flow_from_directory(
        directory= train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='categorical',
        batch_size= batch_size)


### R1D75

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

- image augmentation: es como se puede modificar las imágenes para que sean más fáciles de modelar, y al mismo tiempo se pueden realizar transformaciones para duplicar la cantidad de train_data.

    ImageDataGenerator(
        rescale=  ,      : reescala todas las imágenes para que estén todas en rango entre 0 y 1 
        fill_mode= ,     : Rellena los pixeles vacíos al realizar transformaciones (en rotación por ejemplo).
        vertical_flip=True,  : invierte la imagen izquierda/derecha.
        horizonta_flip=True, : ídem anterior arriba/abajo.
        rotation_range= ,    : Gira la imagen en el angulo que se pone.
        zoom_range= ,      : Hace zoom a la imagen. si es un float, hace al azar entre [1-float,1+float]. menos de 1 acerca, más de 1 aleja
        width_shift_range= , : mueve la imagen para un costado. Si se pone un float, es porcentaje, y si es un int es la cantidad de pixeles
        height_shift_range= , : Ídem anterior pero de altura
        brightness_range=[0.4,1.5] : cambia la luz de la imagen, en valores dentro del rango marcado. menos de 1 es más oscuro, más de uno más claro.
        )


### R1D76

Ejemplo de proceso para una convolutional network con tf y keras:
**plantear modelo**
model = Sequential([                              
    # First conv layer + subsampling
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Second conv layer + subsampling
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Third layer (flatten)
    tf.keras.layers.Flatten(),
    # Fourth layer (dense)
    tf.keras.layers.Dense(128, activation='relu'),
    # Fifth layer (output)
    tf.keras.layers.Dense(2)
])
**mostrar resumen de layers**
model.summary()
**compilar modelo**
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
**fittear el modelo con los datos**
history = model.fit(train_data_gen,
                    steps_per_epoch= train_steps,
                    epochs= epochs,
                    validation_data= (val_data_gen),
                    validation_steps= val_steps,
                    )
**observación:** Si x es un dataset, generator, o keras.utils.Sequence instance, NO va y (este caso). Sino, hay que ponerlo (ídem ara validation_date)
**Ver la perdida de accuracy**
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
**Evaluar el test data**
probabilities = np.argmax(model.predict(test_data_gen), axis=-1)
test_images = [test_data_gen[0][i] for i in range(50)]

def plotImages(images_arr, probabilities = False):
    fig, axes = plt.subplots(len(images_arr), 1, figsize=(5,len(images_arr) * 3))
    if probabilities is False:
      for img, ax in zip( images_arr, axes):
          ax.imshow(img)
          ax.axis('off')
    else:
      for img, probability, ax in zip( images_arr, probabilities, axes):
          ax.imshow(img)
          ax.axis('off')
          if probability > 0.5:
              ax.set_title("%.2f" % (probability*100) + "% dog")
          else:
              ax.set_title("%.2f" % ((1-probability)*100) + "% cat")
    plt.show()
plotImages(test_images, probabilities=probabilities)


-   tf.keras.layers.Conv2D(cantidad de filtros, (3, 3) -> Tamaño de filtro, activation='relu' u otro, 
        input_shape=(150, 150, 3)) : tamaño de las muestras y canales (en este caso, imágenes de 150x150 RGB, si fuera gris sería 1 el último). Sólo va en la primera
- Usar esta función para calcular los steps_per_epoch y validation_steps, sin generar conflicto con las cantidades inexactas.
def cal_steps(num_images, batch_size):
   # calculates steps for generator
   steps = num_images // batch_size

   # adds 1 to the generator steps if the steps multiplied by
   # the batch size is less than the total training samples
   return steps + 1 if (steps * batch_size) < num_images else steps

### R1D79

- data = df1.merge(right=df2, on='isbn') : Para unir/concatenar dos DF, el on= dice que columna comparar para unir y agregar el resto de columnas.
- df.info o df['columna'].isnull().value_counts(): Se puede obtener si existen valores nulos/vacíos en el df.
- pivot_df = df.pivot(index='columna1', columns='columna2', values='columna3').fillna(0): Crea un df con los datos delas 3 columnas, marcando las relaciones. Sirve para el modelo de nearestNeighbor (requiere un array o un aray like). Se tiene que convertir a array, por ejemplo haciendo pivot_df.values (creo que no es completamente necesario)
- Para modelar con NearestNeighbors,m unsupervised,y encontrar los puntos cercanos a un dato buscado, se puede hacer así. Si lo que se busca es un string, se debe convertir primero a un numero (por ejemplo, buscar el índice donde se encuentra)
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine').fit(pivot_df)     : El cosine hace que las distancias sean menores a 1.
    distances, title_indexes = nbrs.kneighbors(X=np.reshape(fila,(1,-1)), n_neighbors=5)     : La fila es todos los datos del punto que se quiere averiguar, basicamente, buscar la fila con ese índice en el df

### R1D81


- Forma de dividir un dataset en traing/test data:
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(dataset, test_size=0.2)
    y_train = train.pop('columna con resultado/label')
    y_test = test.pop('columna con resultado/label')
*Se puede hacer también con tensorflow*

- Ejemplo de Regresión lineal sin tensorFlow/keras:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    model = LinearRegression()

    lr = model.fit(train_dataset, train_labels)
    train_prediction = model.predict(train_dataset)
    train_mae = mean_absolute_error(train_prediction, train_labels)
    train_mae

    test_pred = model.predict(test_dataset)
    test_mae = mean_absolute_error(test_pred, test_labels)
    test_mae

- Coeficientes de un modelo linearl: model.intercept_, model.coef_


### R1D84

- pwd : Muestra el directorio donde está ubicado el notebook


página útil sobre diferentes formas de tokenization
https://www.analyticsvidhya.com/blog/2019/07/how-get-started-nlp-6-unique-ways-perform-tokenization/#:~:text=Tokenization%20is%20essentially%20splitting%20a,smaller%20units%20are%20called%20tokens.&text=The%20tokens%20could%20be%20words%2C%20numbers%20or%20punctuation%20marks.


### R1D85

- Generar tablas y DataFrames desde bigData con SQL bigquery

from google.cloud import bigquery


client = bigquery.Client() : *Crear un objeto "Client" que contendrá los proyectos y las conexiones con el servicio BigQuery*

**Cada dataset está contenido en un projecto (ej: bigquery-public-data)**
**Construct a reference to the "hacker_news" dataset**
dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")
**API request - fetch the dataset**
dataset = client.get_dataset(dataset_ref)
**List all the tables in the "hacker_news" dataset**
tables = list(client.list_tables(dataset))
**Print names of all tables in the dataset (there are four!)**
for table in tables:  
    print(table.table_id)
**Construct a reference to the "full" table**
table_ref = dataset_ref.table("full")
**API request - fetch the table**
table = client.get_table(table_ref)
**Table Schema: Estructura de la tabla**
**Print information on all the columns in the "full" table in the "hacker_news" dataset**
table.schema
**A las columnas también se les llama campos (fields)**
**Preview the first five lines of the "full" table y las convierte a un DataFrame de pandas**
**Se puede también seleccionar algunas columnas, con selected_fields=table.schema[:1] por ejemplo para la primer columna**
client.list_rows(table, max_results=5).to_dataframe() 

- SQL queries
    - query = """
            SELECT *columna*,*columna2*  : agregar DISTICT si se quiere obtener los valores no repetidos
            FROM *`dirección tabla`*  : ejemplo. `bigquery-public-data.openaq.global_air_quality`
            WHERE *condición*
    - query = """
            SELECT COUNT(*columna*/*1*) AS *nombreColumna*  : o SUM(), AVG(), MIN(), and MAX(). Da el resultado como una nueva columna con alias pasado. Se puede poner COUNT(1) como convención para contar las filas.
            FROM *`dirección tabla`*
            GROUP BY *columna*  : Agrupa por valores únicos la columna para usar una función de aggregate en el SELECT
            HAVING AGGFUNC(*columna*) *condición (ej: >1)*   : Si se agrega esto, da una condición para mostrar parte del resultado GROUP BY
            ORDER BY *columna* : Para ordenar los resultados según alguna columna de forma ascendiente, si se agrega DESC, será de forma descendiente
    - query = """
            SELECT EXTRACT(DAY from *columna con Fechas*) AS Day   : Genera una columna con el día sacado de la fecha con tipo DATE o DATEIME (formato aaaa-mm-dd). (Doc: https://cloud.google.com/bigquery/docs/reference/legacy-sql#datetimefunctions)
            FROM *`dirección tabla`*
    - query = """
            WITH CTE AS
            (
                SELECT EXTRACT(HOUR FROM trip_start_timestamp) AS hour_of_day,
                        trip_miles,
                        trip_seconds
                FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                WHERE trip_start_timestamp > '2017-01-01' AND
                        trip_start_timestamp < '2017-07-01' AND
                        trip_seconds > 0 AND
                        trip_miles > 0
            )
            SELECT hour_of_day,
                    COUNT(1) AS num_trips,
                    3600 * SUM(trip_miles) / SUM(trip_seconds) AS avg_mph
            FROM CTE
            GROUP BY hour_of_day
            ORDER BY hour_of_day
            """

- query_job = client.query(query)  : Para establecer la query.
- df = query_job.to_dataframe()  : Corro la query y genero un df con los resultados
- dry_run_config = bigquery.QueryJobConfig(dry_run=True)
  dry_run_query_job = client.query(query, job_config=dry_run_config)  : Estas dos líneas sirven para saber cuantos bytes procesará la query (cuidado con el límite de 3TB en kaggle!)
- safe_config = bigquery.QueryJobConfig(maximum_bytes_billed={número máximo de bytes}})
  safe_query_job = client.query(query, job_config=safe_config) : Corta la cantidad de bytes a procesar por la query y luego corre la query con esa config (si lo excede, tira error)
- WHERE column_name BETWEEN value1 AND value2 : Para seleccionar sólo los valores que estén dentro de un rango (incluye los bordes)


### R1D86

- CTE: Common table expressions. Expresiones con WITH... AS que contienen una tabal temporal que se retorna dentro de la query
- WHERE *columna* LIKE '%textoABuscar% : Para filtrar sólo las filas cuya columna contenga la parte del texto que estoy poniendo.


### R1D87

Se puede conseguir la sparse matrix para el modelo con un dummy = pd.get_dummies(dataset['columna']), pero tiene varias complicaciones, como que no es tan reproducible.
One hot encoding:

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 

Alternativa 1:

encoder = OneHotEncoder() 
encoder.fit(train_df[['columna']])
matrix = encoder.transform(train_df[['columna']]).todense()
df = pd.DataFrame(matrix, columns=encoder.categories_, index=train_df[['columna']].index)

Alternativa 2:
one_hot_encode = ColumnTransformer ([
                                     (
                                      'one_hot_encode',   #Transformation name
                                      OneHotEncoder(sparse=False),   #transformer to use
                                      ['columna1', 'columna2', 'columna3']    #features to transform
                                     )
                  ])
one_hot_encode.fit(train_df)
matrix = one_hot_encode.transform(train_df)
df = pd.DataFrame(matrix, index=train_df.index)


Scaling:

Alternativa 1:

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
scaler = MaxAbsScaler()
scaler.fit(train_df[['columna']])
matrix = scaler.transform(train_df[['columna']])
pd.DataFrame({'original': train_df['columna'].values, 'scaled': matrix.squeeze()}).describe()   #Para comparar original y escalada
df = pd.DataFrame(matrix, columns=['nombreColEscalada'], index=train_df[['columna']].index)

Alternativa 2:

scaler_encoding = ColumnTransformer([
                            (
                             'MinMaxScaler_encoding',
                             MinMaxScaler(),
                             ['columna1', 'columna2']
                            )
          ])
scaler_encoding.fit(train_df)
matrix = scaler_encoding.transform(train_df)
df = pd.DataFrame(matrix, index=train_df.index)


### R1D88

Más queries de SQL:

- FROM *Primera tabla* AS p
  *X* JOIN *Segunda tabla* AS s 
    ON s.primary_key = p.foreign_key
  La X puede ser (considerar que cuando no hay match, traerán datos con NULL):
        - INNER: Trae sólo los datos que estén relacionados entre ambas tablas.
        - LEFT: Trae todos los datos de la primer tabla y sólo los de la segunda tabla que estén relacionados con la primera.
        - RIGHT: Trae todos los datos de la segunda tabla, y sólo los de la primera tabla que estén relacionados.
        - FULL: Trae todos los datos de las dos tablas, tengan match de key o no.

- SELECT *columnatabla1* FROM *`dirección1`*
  UNION ALL  : Si es UNION DISTICT sólo trae los valores únicos, descarta los duplicados
  SELECT *columnatabla2* FROM *`dirección2`*
Esto une ambas tablas verticalmente, tienen que tener el mismo datatype
- SELECT MIN(TIMESTAMP_DIFF(*segunda columna*, *Primer columna*, SECOND)) as time_passed  : Devuelve los segundos pasados entre las fechas de las dos columnas.

- Analytic Functions:
documentación útil: https://cloud.google.com/bigquery/docs/reference/standard-sql/analytic-function-concepts
- Ej:
    SELECT *,
           SUM(*Columna a realizar suma*) 
                OVER (
                        PARTITION BY *columna sobre la que se divide la función*
                        ORDER BY *Columna a ordenar, por ejemplo una de fechas*
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ) AS *Nombre columna generada*


- Window frame: Ventana de filas donde va a ocurrir lo que pida la función, luego de particionar en grupos y ordenar. Ejemplos:
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW: Todas hasta la actual. Predeterminada.
    ROWS BETWEEN 1 PRECEDING AND CURRENT ROW - La fila anterior de la tabla y la actual.
    ROWS BETWEEN 3 PRECEDING AND 1 FOLLOWING - las últimas 3 filas, la actual, y la siguiente.
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING - Todas las filas de la partición.

Tipos de funciones analíticas:
1) Analytic aggregate functions
MIN() (or MAX()) - Retorna el mínimo o máximo de los inputs
AVG() (or SUM()) - Retorna el promedio o suma de los inputs
COUNT() - Retorna el número de files de los inputs
2) Analytic navigation functions
FIRST_VALUE() (or LAST_VALUE()) - Retorna el primer o último de los inputs
LEAD() (and LAG()) - Retorna el siguiente o anterior de los inputs (fila siguiente o anterior)
3) Analytic numbering functions
ROW_NUMBER() - Retorna el orden en que aparece de los inputs
RANK() - Todas las filas ordenadas con el mismo valor reciven el mismo rango, y luego las siguiente calculan el rango según la cantidad de filas con el rango anterior.

### R1D89

- NESTED columns: Columnas en una tabla de BigQuery que contienen un objeto con más de una característica. Son tipo STRUCT (o tipo RECORD) con más datos dentro con sus propios tipos. Se llama al dato específico seleccionando la columna y el dato (ej: *columna.dato* )
- REPEATED columns: Columna que acepta más de un objeto (dato) dentro de cada fila. El modo de la columna en el schema aparece como REPEATEAD. Cada entrada (fila) en una columna estas es un ARRAY. Para generar una query con esto, se debe poner debajo del FROM, UNNEST(*columna*) AS XXXX.

Ejemplo de uso de nested y repeated:
query = """
        SELECT l.key AS nombre1,   : Key sería el nombre de la variable del objeto buscado
            COUNT(*) AS numbre2
        FROM `bigquery-public-data.github_repos.languages`,
            UNNEST(columnaRepeated) AS l
        GROUP BY nombre1
        ORDER BY nombre2 DESC
        """

- print(table.schema[X]) : Imprime el schema de la columna X de la tabla.

### R1D90

- Función para ver la cantidad de data procesada en una query
def show_amount_of_data_scanned(query):
    # dry_run lets us see how much data the query uses without running it
    dry_run_config = bigquery.QueryJobConfig(dry_run=True)
    query_job = client.query(query, job_config=dry_run_config)
    print('Data processed: {} GB'.format(round(query_job.total_bytes_processed / 10**9, 3)))

- Función para ver el tiempo que tarda una query
from time import time
def show_time_to_run(query):
    time_config = bigquery.QueryJobConfig(use_query_cache=False)
    start = time()
    query_result = client.query(query, job_config=time_config).result()
    end = time()
    print('Time to run: {} seconds'.format(round(end-start, 3)))

### R1D91

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

### R1D92

- Para eliminar las filas que tengan algún valor outlier en alguna de sus columnas. Considero los outliers como valores con zscore >= 3
    from scipy import stats
    train_df_no_outliers = train_df[(np.abs(stats.zscore(train_df)) < 3).all(axis=1)]  

- Para comprobar que un dataset ya está descargado:
    ```python
    import pathlib
    file = pathlib.Path("folder/file")
    if file.exists():
        print('Files already exists')
    else:
        print('Creating the file')
    ```

### R1D93

- Stratified= : Se puede usar en el train_test_split para que la separación la haga manteniendo la relación de la columna que se pase. Esta columna será la que se considere el target. Sirve para datasets desbalanceado (imbalanced)

- Error en regresión logística:
ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

**Respuestas útiles:**
https://stackoverflow.com/questions/62658215/convergencewarning-lbfgs-failed-to-converge-status-1-stop-total-no-of-iter

- Se puede aumentar el número de iteraciones con LogisticRegression(solver='lbfgs', max_iter=XXX) con XXX > 100
- Se puede cambiar el solver del lbfgs predeterminado
    Diferentes optimizadores para regresión logística:
    https://stackoverflow.com/questions/38640109/logistic-regression-python-solvers-definitions/52388406#52388406


### R1D94

- df.isnull().sum().any() : Forma alternativa de ver rápido si existen valores nulos. Se complementa con df.isna().sum().value_counts() si se quiere ver cuantos


### R1D95

- display(): Reemplazo del print, que muestra los DataFrame correctamente. 
- %%time : Agregarlo a cada celda al ppio para chequear el tiempo que tarda en correr
- df.memory_usage() : Devuelve el uso de memoria de cada columna del DF en Bytes.
- np.finfo(): Los límites de la máquina para cada floating point type.

Ejemplo de como reducir el uso de memoria optimizando dtypes para floats:

`display(f'Initial memory usage: {df.memory_usage().sum()/1024**2:.2f}')`

```python
def reduce_memory_usage(df):
    start_mem = df.memory_usage().sum()/1024**2
    datatypes = ['float16', 'float32', 'float64']

    for col in df.columns[:-1]:     #Si se quiere eliminar la última columna (gralmente el target)
        for dtp in datatypes:
            if abs(df[col]).max() <= np.finfo(dtp).max:
                df[col] = df[col].astype(dtp)
                break

    end_mem = df.memory_usage().sum()/1024**2
    reduction = (start_mem - end_mem)*100/start_mem
    print(f'Mem. usage decreased by {reduction:.2f}% to {end_mem:.2f}')
    return df

df = reduce_memory_usage(train)
```

- Cardinality: Número de valores únicos en las features. Por ej: Si tiene muy pocos valores, quizás considerar a esa feature categórica tenga mejores resultados.
- df.nunique(): Cuenta la cantidad de valores unicos en el eje específicado (axis=0 columnas)
- df.T.style.background_gradient(cmap='RdYlGn', subset=[]).bar(subset=[]), color='tomato'): Sirve para colorear las celdas según el cmap por los valores. el .bar es para hacer una barra con el color, y el subset es para marcar en qué features se hará.


- Eliminar el título del eje X:
    fig, ax = plt.subplots()
    ax.set(xlabel=None)
- sns.despine() : Elimina los bordes del gráfico que no tienen ticks.

### R1D97

**Markdown**  

ctrl+k y depsués v para activar el preview en vsCode.
### Usar de 1 a 6 # para títulos (Este es con 3)
Doble barra espaciadora y enter hace salto de línea simple  
Doble enter hace salto de línea con espacio

- Guión (-) genera listas

*Asteriscos simples es cursiva*  
**Asteriscos dobles, negrita**  
***Asteriscos triples es cursiva y negrita***  
`[](link)` para links  
`![alt text](dirección de imagen) para cargar una imagen.`  
<span style='color:#ffd700'>Se puede usar HTML, como  `<span style='color:#ffd700'></span>`</span>  
Encerrar en ` es para mostrar la línea de código como texto  

```python 
    # Encerrar entre tres ` genera bloque de código. Escribir qué lenguaje luego de las primeras tres 
```
> Empezar con > para citas. Se pueden anidar  

Para listas se usa | como separación, con primer renglón encabezado, segundo formato y después los datos:  
encabezado1 | Encabezado 2
------------|-------------
dato1       | dato2

