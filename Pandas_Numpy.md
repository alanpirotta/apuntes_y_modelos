```python
    # Importo la librería con la convención de nombre
    import pandas as pd
    import numpy as np
```

## Visualización de datos:

`display()`: Reemplazo del print, que muestra los DataFrame correctamente. Es una función del módulo de IPython  
`df.info()`: Muestra varios datos, sirve para ver la cantidad de valores NaN de cada columna. Alternativa: `df.isna().sum() `
`df.isnull().sum().any()`: Forma alternativa de ver rápido si existen valores nulos. Se complementa con df.isna().sum().value_counts() si se quiere ver cuantos  
`df.memory_usage()`: Devuelve el uso de memoria de cada columna del DF en Bytes.  
`np.finfo()`: Los límites de la máquina para cada floating point type.  
`Series.dtype`  y `df.dtypes`: Muestra los tipos de datos de las tiras de datos  
```python
# Función para mostrar DF uno al lado del otro en el output
from IPython.display import display, HTML

def display_side_by_side(dfs:list, captions:list, tablespacing=5):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    output = ""
    for (caption, df) in zip(captions, dfs):
        output += df.style.set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
        output += tablespacing * "\xa0"
    display(HTML(output))
```

## Slices de df, Series y Arrays:

`df.loc[fila, columna]`: Para extraer datos (se puede con máscaras). Usa índices explícitos e incluye el último valor marcado en el slice.  
`df.iloc[fila, columna]`: Usa índice implíctos (rangeIndex), no incluye el último valor del slice.  

### Juntar Series y DataFrames: 
```python
np.concatenate([Array1, Array2])
df.iloc[np.r_[df.head().index, 10:15, df.tail().index],]
# Sirve para obtener diferentes slices del mismo DataFrame
pd.concat([df1, df2])
# Concatena uno abajo del otro, se puede hacer en columnas o con loc
df.join(): 
#junta dos DF en uno. Las columnas deben tener distintos nombres ( o agregar sufijos)
df.merge(segundo_df, how= 'joins de sql', on= [columnas que se comparan])
# funciona como el concat, pero se peude específicar más cosas
```

### Acceder a **un** valor de una fila y columna:  
- `df.loc[fila][columna]` o `df.loc[fila, columna]`: Busca por el nombre que se le haya asignado a los índices. funcionan con máscaras.
    - `df.at[fila, columna]`: Devuelve sólo un valor 

### Extraer percentiles:
`pd.qcut(df['columna'],num_cortes)` : Genera una Series con las filas separadas en la cantidad de partes que se pida (4 sería los cuatro cuartiles). Luego se puede agrupar el df usando esta Serie.  
`df.quantile(q= 0-1)`: devuelve el cuantil/percentil que se está pidiendo entre 0 y 1. Se pueden comparar y usar como filtros para eliminar extremos  
`df[(df['value'] > df['value'].quantile(0.025)) & (df['value'] < df['value'].quantile(0.975))]` : Forma rápida de eliminar el top y bottom 2.5% de datos según una columna  


## Generación de datos y estructuras:

Random Generator de Nuympy: Provee varias forams de generar datos con distribuciones específicas.  
    `rng = np.random.default_rng(seed)` : Devuelve el generador rng  
    ej dist normal: `rng.normal(punto_central, desviacion, cantidad)`

`pd.Series(data, index= ,name= )`: genera una Series de Pandas  
`pd.DataFrame(data, index= , columns= )`: Genera un DataFrame con los datos pasados
```python
pd.read_csv(data,
           sep='', # separador entre cada columna
           encoding='', #Formato de codificado de texto (UTF8, latin1)
           dtype='', # Se puede definir los tipos de datos directamente, ingresando como un diccionario con las colmnas como keys.
           skip_blank_lines=True # Eliminar las filas que no tienen datos (generalmente al inicio o al final)
```
## Manipulación de datos:

### Máscaras:
`df[df['columna'].isin([lista_de_valores])]` : Para extraer sólo las filas donde los valores de esa columna estén dentro de la lista.  

### groupby:
`df.groupby([columna])`: Agrupa todas las filas con el mismo valor. (símil a tablas dinámicas en excel)  
`df.groupby().size()`: Devuelve una series con la cantidad de entradas de cada agrupación.    
`df.groupby('columna1').describe().loc[indice,'columna2']` : Agrupa, muestra los datos descriptivos de una parte específica.  
*Alternativa:* `df.groupby('columna1').agg(['count', 'sum', 'mean', 'median']).loc[índice,'columna2']`: Otra forma de obtener lo mismo, pero con el método agg donde puedo seleccionar qué quiero poner, y poner otras cosas (como la suma)  
`df.columns = df.columns.map('_'.join).strip('_')`: Para unir los nombres del multiIndex de columnas creado en un groupby.  
*Alternativa:* `df.columns = ['_'.join(col).strip('_') for col in df.columns.values]`

### Pivot
`df.pivot_table(index=['features a agrupar'], aggfunc={'feature a agg':['mean', 'std'], 'feature2 a agg':['mean', 'std']})` : las func de aggregación son las columnas. Se pueden poner que features agregar como "values=", pero ahí quedan agrupadas por la función de agregación en vez del dato original


`array.reshape(1,-1)`: Cambia la forma del array. El -1 sirve para completar lo que se necesite (de forma que mantenga la cantidad de datos constante)  
`df.apply()`: sirve para modificar todos los valores en un DF. el "axis" es para definir si lo hace en cada columna o en cada fila.  
`df['column].sort_values()`: si es una series no necesita "by=", si es un DF requiere que se seleccione la columna.  
`df.rename(columns={'nombreviejo': 'nombrenuevo'})`: Cambiar/renombrar   los nombres de columna
`df.rename_axis('nombre_columnas', axis='columns')`: renombrar los "ejes" de la tabla.   
`serie.name = "nombre_nuevo"` : Cambia el nombre de la pd.Series
`pd.Timedelta(days=, hours=, minutes=, seconds=)`: Crea datos del tipo Timedelta de lapsos de tiempo (si se pasan a np.int64 son nanosegundos). Se puede usar en un apply de una feature (separando correctamente las diferentes partes)


## Notas desordenadas

- astype: sirve para cambiar el tipo de dato (crea nueva serie)
- pd.isnull y pd.notnull
- fillna y replace: Para reemplazar valores de los DF y series.
- df.set_index(): setea los index usando una o más columnas del DF. Sirve para después juntar varios DF usando estas columnas como puntos en común.
- DF.max(axis=1)['row']: Para buscar el máximo valor de una fila (si se cambia el axis sería de una columna)
- DF.idxmin(axis=1)['row']: Para buscar el nombre de la columna donde está el mínimo. se puede cambiar para buscar el indice de la fila 
- DF.max().idxmax(): Devuelve el label de la columna con el valor más alto de un DF
- DF[DF.max().idxmax()].idxmax(): devuelve el index del valor más alto de un DF
- pd.to_numeric(): sirve para convertir una series a int o float. Para convertir parte de un df, se puede usar appliy:
    - df[['column1', 'column2']].apply(pd.to_numeric, errors=, downcast=): el errors sirve para que los no convertibles los ignore o los convierte en NaN. downcast te devuelve el int o float más chico posible
- df.astype(int/float): Convierte el tipo de las columnas al buscado. No acepta NaN, por lo que se debe usar fillna() antes.
- df[['columna1', 'columna2']]= df['columna a dividir'].str.split('caracter que divide', expand=True): Para separar una columna en más de una, según un caracter. el expand=True es para que directamente lo divida en diferentes columnas.
- df.drop('columna/fila', inplace=, axis=): Para eliminar una columna/fila. inplace=True sobreescribe el archivo, es preferible guardarlo de nuevo. 
- df[ df['columna'].isna() ]: Sirve para chequear todas las filas en las que no hay datos para esa columna
- df[una series].str.replace('valorviejo', 'valornuevo'): Sirve para reemplazar una parte de string en una columna
Formas de separar un df por años, cuando la columna es una fecha completa
- df_2017 = df['2017-01-01':'2017-12-31']: forma completamente manual, funciona pero se hace una línea por año
- datos_2018 = datos['{}-01-01'.format('2018'):'{}-12-31'.format('2018')]: Funciona para hacer un ciclo
- Lo siguiente devuelve una lista de dataframespor año:
años=[2017,2018,2019,2020,2021]
años_df=[]
for año in años:
    años_df.append(datos['{}-01-01'.format(año):'{}-12-31'.format(año)])
- datos.index.date[0]: Devuelve datetime.date(año, mes, dia) de la primer fila. Es un tipo de dato "datetime.date"
    Si agrego .year devuelve el año, pero de esa fila específica. Tendría que hacer un loop por todos los indices del objeto datetime
- pd.set_option('display.max_rows', 10): Sirve para el display de un máximo de filas al poner "df"
- df = df.reset_index(): Resetea el index y lo convierte en una columna.
- df = df.set_index('feature'): Poner una columna como index. Sirve para volver a poner luego de resetear
- df['mes_año'] = df['Fecha'].dt.to_period('M'): Extrae el mes y año y los crea en una nueva columna. si se pone 'Y' extrae solo el año.
- df['año'] = pd.DatetimeIndex(df.index).year: Crea una columna con sólo el año de la fecha que está como index
- Error "'DatetimeIndex' object has no attribute 'dt'" lo tira cuando se quiere usar el index para el to_period. Tiene que ser una serie si o si (feature del df)
- df['mes_dia']= df['Fecha'].dt.strftime('%m-%d'): Sirve para extraer como un string el mes y día solamente de una fecha. NO se puede considerar como fecha sin el año.
- df.duplicated().sum() : Para chequear cuantas filas están duplicadas en el DF
Chequeo de duplicados en index. (aplica para columnas que no sean índice)
- df.index.duplicated(): Devuelve un array con True para la segunda aparición de un valor. Si se pone keep='last', el que mantiene (aparece como False) es la última aparición. Si se pone keep=False devuelve con True en todos los valores duplicados. 
- df = df.drop_duplicates(): Descarta todos los valores duplicados.
- ((array1 == array2) == False).sum(): Sirve para chequear que todas las columnas de la fila duplicada (cada array) sean iguales o no. Se puede también comparar directo con un loc, pero si el repetido es el índice, antes se debe resetear el índice.
- df['año_mes'] = df[columna con fecha].dt.to_period('M'): Extrae sólo el año y día. Si la fecha es en el índice, primero resetear
- df_nuevo = df.reset_index().set_index(['columna_1', 'columna_2']): indexar dos columnas del dataframe (o más) como multiIndex.
- df_nuevo.loc[(dato_primer_index , dato_segundo_index)]: Esto sirve para acceder a un dato específico (o conjunto de datos), de un df con multiIndex.
- Genera un df nuevo con datos específicos de cada columna luego de un groupby
  gb = df.groupby(['col1', 'col2'])
  counts = gb.size().to_frame(name='counts')
  (counts
   .join(gb.agg({'col3': 'mean'}).rename(columns={'col3': 'col3_mean'}))
   .join(gb.agg({'col4': 'median'}).rename(columns={'col4': 'col4_median'}))
   .join(gb.agg({'col4': 'min'}).rename(columns={'col4': 'col4_min'}))
   .reset_index()
  )
- df.groupby(['A', 'B'])['C'].describe(): Alternativa y te muestra las de una columna sola
- Para agregar una columna de suma de datos en un multiIndex de columnas (el condicional es para eliminar las columnas que no sean numéricas, que NO se quieran agregar):
    for columna in por_mes.columns:
        if columna == 'Fecha':
            continue
        agrupado_por_mes[columna,'sum'] = por_mes.groupby(['año','mes'])[columna].sum()

- df.loc[(indice_1_nivel,indice_2_nivel),'columna'] = value : Esto sirve para cambiar un valor. (puede ser que cambie varios si comparten esos índices)
- df.index.get_level_values(nivel de indice(0,1,2...)).unique(): Para extraer los valoresúnicos de un índice específico. Forma extraña, hay más fáciles.

- Df.fillna(method='ffill') o bfill: para que conplete los NaN con el valor de la fila anterior.

- array.tolist(): Convierte un array a una lista.
- df.drop(df[<some boolean condition>].index): Eliminar filas según una condición o valor. La condición puede ser un df filtrado
- Formas de medir la cantidad de filas:
    - df.count()
    - df.shape[0]
    - len (df)
    - np.around('numero',cantidaddedecimales): Redondea el numero a la cantidad de decimales pedidos (me sirvió mucho al calcular la media de columnas)
- pd.DataFrame(data): Para crear un DataFrame, primero crear un diccionario con las diferentes columnas (los valores pueden ser series, o listas por ejemplo)
- df.sort_values(by=['columna'], ascending=False): Ordena de mayor a menor (crea nuevo df)
- df['columna'].replace({valoranterior:valornuevo, valoranterior2:valornuevo2}): Sintaxis que sirve para modificar/reemplazar varios valores por otros en una columna de un df. Genera un nuevo df
- pd.melt(data, id_vars= ['columna que se quiere mantener], value_vars=['columnas que se quieren ubicar una encima de la otra]): Converte todas las columnas a sólo dos: variable ( nombre de columna) y value ( valor ). Sirve para graficar más de dos columnas facilmente por ejemplo

- df['year'] = [d.year for d in df_box.date] : Otra alternativa para loopear por todo la columna de fecha del dataFrame y agrega una sólo con el año
- df['month'] = [d.strftime('%b') for d in df_box.date]: Forma de crear columna con el mes, desde una columna con la fecha. el '%b' muestra el nombre del mes, en vez del número
- np.where(condicion, valor si True, valor si False): Sirve para convertir texto de un dataset a valores binarios por ejemplo. Es el equivalente por ejemplo a extraer parte de un dataset cuando el texto == a algo, y coenvertir todos esos valores a 1, y el resto a 0
- pd_read_table : para leer archivos .tsv (tab separated values). names=() define los nombre de columnas (ídem a pd_read_csv)
- array.squeeze() : Sirve para eliminar dimensiones vacías es un array. Por ej: [[[2]]] lo convierte a [2]
- Para contar ocurrencias de cada valor en un np.array, como un diccionario:
    unique, counts = np.unique(array, return_counts=True)
    dict(zip(unique, counts))



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


- df.info o df['columna'].isnull().value_counts(): Se puede obtener si existen valores nulos/vacíos en el df.
`pivot_df = df.pivot(index='columna1', columns='columna2', values='columna3').fillna(0)`: Crea un df con los datos de las 3 columnas, marcando las relaciones. Sirve para el modelo de **nearestNeighbor** (requiere un array o un aray like). Se tiene que convertir a array, por ejemplo haciendo pivot_df.values (creo que no es completamente necesario)

