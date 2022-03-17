**Falta pasar todo ML, Python "base" y generar modelos de notebooks**  
*En las librerías con más apuntes o más visuales, decidir si conviene así o en un notebook con ejemplos*

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

### R1D67

- Formas de hacer una regrasión lineal:
    - slope, intercept, r_value, p_value, stderr = linregress(x=, y=) : Desempaqueta (unpack) los resultados de la regresión en las diferentes variables.
    - reg = linregress(x=, y=): Guarda los datos en una sola variable, se pueden acceder por ejemplo con reg.slope.
- plt.plot(x, reg.intercept + reg.slope*x, 'r', label='fitted'): Para graficar la regresión (se puede hacer en la misma figura que un scatter), utilizando la fórmula en el eje Y. el 'r' indica color rojo. 

### R1D74

- image_generator = ImageDataGenerator(rescale=1./255): Generador de imágenes que permite hacer image augmentation, con algunos atributos (rescale sirve para que todas las imágenes tenga el mismo tamaño). Tiene el método flow_from_directory, por ejemplo
- Para hacer la modificación de las imágenes y poder dejarlas aptas para nuestro modelo, desde un directorio. Nota: SIEMPRE espera tener al menos una carpeta dentro del directory (no se pueden tirar las imágenes dentro de la carpeta test directo, por ejemplo)
    image_generator.flow_from_directory(
        directory= train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='categorical',
        batch_size= batch_size)

### R1D75

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
```python
def cal_steps(num_images, batch_size):
   # calculates steps for generator
   steps = num_images // batch_size

   # adds 1 to the generator steps if the steps multiplied by
   # the batch size is less than the total training samples
   return steps + 1 if (steps * batch_size) < num_images else steps
```
### R1D79

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

página útil sobre diferentes formas de tokenization
https://www.analyticsvidhya.com/blog/2019/07/how-get-started-nlp-6-unique-ways-perform-tokenization/#:~:text=Tokenization%20is%20essentially%20splitting%20a,smaller%20units%20are%20called%20tokens.&text=The%20tokens%20could%20be%20words%2C%20numbers%20or%20punctuation%20marks.

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

### R1D92

- Para eliminar las filas que tengan algún valor outlier en alguna de sus columnas. Considero los outliers como valores con zscore >= 3
    from scipy import stats
    train_df_no_outliers = train_df[(np.abs(stats.zscore(train_df)) < 3).all(axis=1)]  

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