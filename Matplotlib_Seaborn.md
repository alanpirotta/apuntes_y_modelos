
### PairGrid

Se debe primero asignar los datos y generar la grilla. Luego, se le generan los gráficos en cada posición, según lo que se necesite.
ej:
```python
    g = sns.PairGrid(df, hue='columna', vars=['lista_de_features'], diag_sharey=False)
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)
    g.add_legend()
```

### Boxplot
Utilidad principal, ver outliers, secundaria, ver distribución.  
```python
fig, axs = plt.subplots(2,1,figsize=(20,5))
bp_ars = sns.boxplot(x='feature', data= df1, ax=axs[0])
bp_usd = sns.boxplot(x='feature', data=df2, ax=axs[1])
bp_ars.set_title('titulo1')
bp_usd.set_title('titulo2')
plt.show()
```

Matplotlib(plt): Sirve para definir las características del gráfico (no el contenico?) 
- plt.figure(figsize=(ancho,alto)): Define el tamaño del gráfico
- plt.title("Título del gráfico")
- plt.xlabel("etiqueta eje X"): Equivalente a plt.ylabel

- sns.lineplot(data=DataFrame): gráfico de linea. Puedo definir varias columnas sólas en diferentes lineas, seleciconando DataFrama['columna'] (agregar atributo "label=")
- plt.xticks(rotation=90): Sirve para rotar 90 grados las label del eje X y dejarlas en vertical (45 también queda bien) . busqué si podía hacerlo con plt.xlabel pero me parece que no existe


- sns.barplot(x= ,y= ): Gráfico de barras
- sns.heatmap(data= DF, annot=True): Gráfico de calor. el annot es para que muestre los valores de cada comparación
- sns.scatterplot(x= , y= ): Gráfico scatter. Sirve para ver correlación. Se puede hacer con una regresión
- sns.regplot(x= , y= ): gráfico scatter con regresión lineal.
- sns.lmplot(x= "nombre columna", y= "nombre columna", hue= "nombre columna" , data= df): Gráfico de regresión con dos rectas, uno por cada set (color-coded con el "hue")
- sns.swarmplot(x= , y= ): categorical scatter plot. Sirve cuando el X es un booleano por ejemplo
- plt.legend(): Llamarlo antes del plt.show() para que muestre los label de los gráficos (si se definieron dentro del gráfico, el () queda vacío acá)

- sns.distplot(a= "columna del DF para eje x", kde=False, label=): el kde genera un histograma de densidad en vez de valores (fuera del valor del eje y, no estoy 100% seguro de en que se diferencia) 
*Deprecado*
- sns.histplot o snsdisplot se usan ahora (los datos se cargan con data=)
- sns.kdeplot(data= "columna del DF para eje x", shade= (boolean colorear area debajo), label=): Es un histograma donde te suaviza todo y unifica, en vez de ser barras por valor.
- sns.jointplot(x= , y= , kind="kde"): Te une dos gráficos kde, y los muestra "desde arriba"
- sns.set_style("theme"): Cambia el estilo de los gráficos a uno de los 5 temas: (1)"darkgrid", (2)"whitegrid", (3)"dark", (4)"white", and (5)"ticks"
- Para graficar varios graficos con una celda, se puede poner el código del primer gráfico, después un plt.show(), después el siguiente código, otro plt.show()... etc
- fig, axs = plt.subplots(nfilas, n°col, figsize=(16,10)): Para crear varios gráfico en la misma celda, sin separación. **no logré averiguar bien como rotar 45° las label del eje x**
- https://www.colourlovers.com/palettes/ :  Esta página sirve para buscar paletas de colores (útiles para los gráficos)
- plt.legend(fontsize= 'large'): tamaño de la letra en la leyenda
- sns.catplot(kind='count', hue='', col=''): el kinddefine que uno de los ejes será la suma de ocurrencias del otro. Con hue='' se puede dividir más y en colores. con col=, genera dos o más subplots. **este es un gráfico que funciona a nivel figura, porque lo que no se puede hacer subplot con mpl**
- sns.heatmap(data, annot=True, fmt='', vmin=, vmax=): el fmt es el formato de las anotaciones, puede ser por ejemplo '.1f' para mostrar sólo un punto decimal (un punto flotante). vmin y vmax es para dar el rango en el que estarán los colores (por ejemplo, de -1 a +1)
- df.corr(): Genera un DF de correlación útil para realizar el heatmap por ejemplo.
- Para enmascarar el triángulo superior derecho de un DF/heatmap: 
        - mask = np.zeros_like(corr) : Genera un array con la misma forma que el df 
        - mask[np.triu_indices_from(mask)] = True : Guarda sólo el triángulo superior derecho
        - sns.heatmap(... , mask=mask)
- Alternativa: mask = np.triu(corr), es más simple. O mask = np.triu(np.ones_like(features_corr, dtype=bool), k=0) 
- fig=nombrecatplot.fig: Para que acceda al Facetgrid de matplotlib el catplot. Deprecada, se usa figure ahora
- nombrecatplot.set_ylabels('nombre'): Cambiar el nombre del eje en un gráfico catplot
- Ejemplo de como marcar una annotation de un outlier en un gráfico
    ```python 
    plt.annotate('Possible outlier', xy=(46,0.030), xytext=(189,0.0070), fontsize=12, arrowprops=dict(arrowstyle='->', ec='grey', lw=2), bbox = dict(boxstyle="round", fc="0.8"))
    ```

- Eliminar el título del eje X:
    fig, ax = plt.subplots()
    ax.set(xlabel=None)
- sns.despine() : Elimina los bordes del gráfico que no tienen ticks.


- Para hacer un ciclo de gráficos de distribución sobre subplots:
    ```python
    def distribution_plotter(data, label, ax, bin_width=500):
        dist = sns.histplot(data, stat = 'count', kde = False, 
                                        line_kws = {'linewidth':5}, 
                                        binwidth = bin_width,
                                        ax = ax)    
                    dist.set_title('Distribucion ' + label + '\n', fontsize = 16)

    fig, ax = plt.subplots(nrows=10, ncols=2, figsize=(25,40), tight_layout=True)
    # tight_layout hace que se separen correctamente
    lista = ['filtro1', 'filtro2'] #Lista con los filtros a usar
    num_lista=0 # número para loopear sobre la lista
    for i in range(10):
        for j in range(2):
            if i==9 and j==1:
                # Creo este condicional para completar el último gráfico, si la cantidad de valores en la lista son impares
                distribution_plotter(columnaBuscada, 'Gráfico del total', ax[9][1])
                break
            distribution_plotter(columnaBuscada[data[columna a filtrar] == lista[num_lista]], 'gráfico de {}'.format(lista[num_lista]), ax[i][j])
            num_lista += 1
    ```
