
## Data Types y conversiones

`SELECT CAST("valor" AS datatype)`  
Para convertir un tipo de dato a otro al traer la info. Se puede usar SAFE_CAST para que convierta en NULL si diera error (convertir un texto a INT, por ej)  

`SELECT FORMAT_DATE("%M/%D/%Y", DATE "2022-05-27")`  
Convierte como muestra las fechas en el output

`SELECT MIN(TIMESTAMP_DIFF(*segunda columna*, *Primer columna*, SECOND)) as time_passed`  : Devuelve los segundos pasados entre las fechas de las dos columnas.

`CURRENT_DATE()` : Para comparar contra la fecha de hoy, se le puede restar o sumar días.

`columna BETWEEN DATE '2022-05-29' AND DATE '2022-06-01'`: Filtra entre determinadas fechas

## Operaciones de String

`concat(columna1, columna2)`: Concentate (une) dos columnas en una sola  
`length(columna)` : Devuelve el largo del string  
`lower/upper)columna)` : Cambia todo a minúsculas o mayúscules  
`lpad/rpad(columna, size)`: Agrega caracteres para llegar a la cantidad de cartacteres pedido  
`ltrim/rtrim(columna,value)`: Elimina espacios en blanco o el valor pasado en la función.  
`replace(columna, stringABuscar, stringNuevo)`: Reemplaza el string buscado por el stirng nuevo  
`reverse(columna)` : Invierte el string  
`split(columna, delimiter)` : divide el string y devuelve una matriz  
`translate(columna, from, to)` : Es muy similar a replace, pero reemplaza caracter a caracter, no necesariamente continuos.  
`strpos(columna, substring)` : Devuelve la posición donde comienza el substring dentro del string. Empieza en 1  
`substr(columna, start, length)`: Obtiene un substring, pasando la posición inicial (comienza en 1) y la cantidad
`trim(columna)`: Remueve todos los espacios en blanco a derecha e izquierda

## Expresiones condicionales:

`IFNULL(columna, string)`: Si hay valor, devuelve el valor, sino el string  
`COALESCE(columna1, columna2,...,columnaN, string)`: Revisa una por una las columnas, y devuelve el primr valor no nulo, o el string en caso de no encontrar
```sql
SELECT
case when condicion1 then string1
     when condicion2 then string2
     ...
     when condicionN then stringN
ELSE stringElse 
END    
```
La query anterior evalúa cada when hasta encontrar la que cumple.  
`IF(condicion, valorSiverdades, valorSiFalso)`


## GROUP BY Y ORDER BY

`GROUP BY 1` : Depende el sql, pero normalmente significa que agrupa por la primer colmna del select. Funciona para ORDER BY también

## WHERE

`WHERE column_name BETWEEN value1 AND value2` : Para seleccionar sólo los valores que estén dentro de un rango (incluye los bordes)  
`WHERE *columna* LIKE '%textoABuscar%` : Para filtrar sólo las filas cuya columna contenga la parte del texto que estoy poniendo.

## JOIN

```sql
FROM *Primera tabla* AS p
  *X* JOIN *Segunda tabla* AS s 
    ON s.primary_key = p.foreign_key
```
La X puede ser (considerar que cuando no hay match, traerán datos con NULL):  
        - INNER: Trae sólo los datos que estén relacionados entre ambas tablas.  
        - LEFT: Trae todos los datos de la primer tabla y sólo los de la segunda tabla que estén relacionados con la primera.  
        - RIGHT: Trae todos los datos de la segunda tabla, y sólo los de la primera tabla que estén relacionados.  
        - FULL: Trae todos los datos de las dos tablas, tengan match de key o no.  

```sql
SELECT *columnatabla1*  
FROM *`dirección1`*
UNION ALL   -- Si es UNION DISTICT sólo trae los valores únicos, descarta los duplicados
SELECT *columnatabla2* 
FROM *`dirección2`*
```
Unir dos tablas verticalmente, tienen que tener el mismo datatype

## Nested columns

Columnas en una tabla de BigQuery que contienen un objeto con más de una característica. Son tipo STRUCT (o tipo RECORD) con más datos dentro con sus propios tipos de datos. Se llama al dato específico seleccionando la columna y el dato (ej: *tabla.columna.dato* )
- REPEATED columns: Columna que acepta más de un objeto (dato) dentro de cada fila. El modo de la columna en el schema aparece como REPEATEAD. Cada entrada (fila) en una de estas columna es un ARRAY. Para generar una query con esto, se debe poner FROM XXXX, UNNEST(*columna*) AS YYYY.

Ejemplo de uso de nested y repeated:  
```sql
        SELECT l.key AS nombre1,   
        --: Key sería el nombre de la variable del objeto buscado
                COUNT(*) AS nombre2
        FROM `proyecto.dataset.tabla`,
                UNNEST(columnaRepeated) AS l
        GROUP BY nombre1
        ORDER BY nombre2 DESC
```

## Trabajar desde notebooks con SQL bigquery

```python
from google.cloud import bigquery
```

`client = bigquery.Client()` : *Crear un objeto "Client" que contendrá los proyectos y las conexiones con el servicio BigQuery*

`dataset_ref = client.dataset("dataset1", project="bigquery-public-data")` : Cada dataset está contenido en un projecto (ej: bigquery-public-data)  
`dataset = client.get_dataset(dataset_ref)` : API request - fetch the dataset  

`tables = list(client.list_tables(dataset))`: List all the tables in the "hacker_news" dataset  
`for table in tables: print(table.table_id)`: Print names of all tables in the dataset  
`table_ref = dataset_ref.table("tableName")`: Construct a reference to the table  
`table = client.get_table(table_ref)`: API request - fetch the table  

`table.schema`: Estructura de la tabla. Print information on all the columns in the "tableName" table in the "dataset1" dataset  
`selected_fields=table.schema[:1]`: Se puede también seleccionar algunas columna. Este ejemplo es para la primer columna. A las columnas también se les llama campos (fields).  
`print(table.schema[X])` : Imprime el schema de la columna X de la tabla.

`client.list_rows(table, max_results=5).to_dataframe()`: Preview the first five lines of the "full" table y las convierte a un DataFrame de pandas.  

- Ejemplos SQL queries
    - query = """  
            SELECT *columna*,*columna2*  : agregar DISTICT si se quiere obtener los valores no repetidos  
            FROM *`dirección tabla`*  : ejemplo `bigquery-public-data.openaq.global_air_quality`  
            WHERE *condición*  
            """
    - query = """  
            SELECT COUNT(*columna*/*1*) AS *nombreColumna*  : o SUM(), AVG(), MIN(), and MAX(). Da el resultado como una nueva columna con alias pasado. Se puede poner COUNT(1) como convención para contar las filas.  
            FROM *`dirección tabla`*  
            GROUP BY *columna*  : Agrupa por valores únicos la columna para usar una función de aggregate en el SELECT  
            HAVING AGGFUNC(*columna*) *condición (ej: >1)*   : Si se agrega esto, da una condición para mostrar parte del resultado GROUP BY  
            ORDER BY *columna* : Para ordenar los resultados según alguna columna de forma ascendiente, si se agrega DESC, será de forma descendiente  
            """
    - query = """
            SELECT EXTRACT(DAY from *columna con Fechas*) AS Day   : Genera una columna con el día sacado de la fecha con tipo DATE o DATETIME (formato aaaa-mm-dd). (Doc: https://cloud.google.com/bigquery/docs/reference/legacy-sql#datetimefunctions)  
            FROM *`dirección tabla`*  
            """


`query_job = client.query(query)`  : Para establecer la query.
`df = query_job.to_dataframe()`  : Corro la query y genero un df con los resultados

## Chequeo y optimización de datos extraídos

`dry_run_config = bigquery.QueryJobConfig(dry_run=True)`  
`dry_run_query_job = client.query(query, job_config=dry_run_config)`  : Estas dos líneas sirven para saber cuantos bytes procesará la query (cuidado con el límite de 3TB en kaggle!) 

`safe_config = bigquery.QueryJobConfig(maximum_bytes_billed={número máximo de bytes}})`
`safe_query_job = client.query(query, job_config=safe_config)` : Estas dos línesas cortan la cantidad de bytes a procesar por la query y luego corre la query con esa config (si lo excede, tira error)

Función para ver la cantidad de data procesada en una query:
```python
def show_amount_of_data_scanned(query):
    # dry_run lets us see how much data the query uses without running it
    dry_run_config = bigquery.QueryJobConfig(dry_run=True)
    query_job = client.query(query, job_config=dry_run_config)
    print('Data processed: {} GB'.format(round(query_job.total_bytes_processed / 10**9, 3)))
```

Función para ver el tiempo que tarda una query
```python
from time import time
def show_time_to_run(query):
    time_config = bigquery.QueryJobConfig(use_query_cache=False)
    start = time()
    query_result = client.query(query, job_config=time_config).result()
    end = time()
    print('Time to run: {} seconds'.format(round(end-start, 3)))
```

## CTE (common table expressions)

Expresiones con WITH... AS que contienen una tabla temporal que se retorna dentro de la query

- Ejemplo de Query
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

## Analytic Functions/ funciones análiticas

documentación útil: https://cloud.google.com/bigquery/docs/reference/standard-sql/analytic-function-concepts

- Ej:
    ```sql
    SELECT *,  
           SUM(*Columna a realizar suma*) 
                OVER (
                        PARTITION BY columna sobre la que se divide la función
                        ORDER BY Columna a ordenar --por ejemplo una de fechas
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ) AS Nombre columna generada
    ```

**Window frame**: Ventana de filas donde va a ocurrir lo que pida la función, luego de particionar en grupos y ordenar.  
Ejemplos:  
- `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`: Todas hasta la actual. Predeterminada.  
- `ROWS BETWEEN 1 PRECEDING AND CURRENT ROW` - La fila anterior de la tabla y la actual.  
- `ROWS BETWEEN 3 PRECEDING AND 1 FOLLOWING` - las últimas 3 filas, la actual, y la siguiente.  
- `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` - Todas las filas de la partición.  

Tipos de funciones analíticas:
1) Analytic aggregate functions  
        - MIN() (or MAX()) - Retorna el mínimo o máximo de los inputs  
        - AVG() (or SUM()) - Retorna el promedio o suma de los inputs  
        - COUNT() - Retorna el número de files de los inputs  
2) Analytic navigation functions  
        - FIRST_VALUE() (or LAST_VALUE()) - Retorna el primer o último de los inputs  
        - LEAD() (and LAG()) - Retorna el siguiente o anterior de los inputs (fila siguiente o anterior)  
3) Analytic numbering functions  
        - ROW_NUMBER() - Retorna el orden en que aparece de los inputs  
        - RANK() - Todas las filas ordenadas con el mismo valor reciven el mismo rango, y luego las siguiente calculan el rango según la cantidad de filas con el rango anterior.  





 