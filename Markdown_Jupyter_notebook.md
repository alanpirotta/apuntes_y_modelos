
## hotkeys jupyter labs
* esc/enter: entrar y salir de modos
* Y/M: convertir celda en code/markdown
* A/B: insertar celda arriba/abajo
* D D: borrar una celda
* X/C/V: Cortar, copiar y pegar celdas seleccionadas
* ctrl+/ o ctrl+}: Comentar una línea

## Google collab
* A/B o ctrl+m+a/b: insertar celda de código arriba/abajo. Con control se peude hacer desde dentro de la celda
* ctrl+m+d : Borra la celda actual

## Markdown

https://programmerclick.com/article/9139292621/
Útl para encontrar códigos menos usados

ctrl+k y depsués v para activar el preview en **VsCode**.
### Usar de 1 a 6 # para títulos (Este es con 3)
Doble barra espaciadora y enter hace salto de línea simple  
Doble enter hace salto de línea con espacio

- Guión (-) genera listas
* Asterisco (*) también
+ Lo mismo la suma (+)

*Asteriscos simples es cursiva*  
**Asteriscos dobles, negrita**  
***Asteriscos triples es cursiva y negrita***  
`[](link)` para links  
`![alt text](dirección de imagen)` para cargar una imagen.   
<span style='color:#ffd700'>Se puede etiquetas de usar HTML con estilo, como  `<span style='color:#ffd700'></span>`</span>  
Encerrar en ` es para mostrar la línea de código como texto  

---
Tres guiones o asteriscos entre renglones vacíos, genera una línea completa  

***

```python 
    # Encerrar dentro de un bloque rodeado de tres ` genera bloques de código. Escribir qué lenguaje luego de las primeras tres lo reconoce como tal.
```
> Empezar con > para citas. Se pueden anidar  

Para listas se usa | como separación, con primer renglón encabezado, segundo formato y después los datos:  
encabezado1 | Encabezado 2
------------|-------------
dato1       | dato2

$\hspace{.5cm}$ `$\hspace{.5cm}$` Genera un espacio en blanco/tabulación. Sirve para sangría.  

### Generar un índice:

Utilizar la etiqueta `<a id="nombreSeccion"></a>` Para identificar cada título de sección/subsección.  
Generar un link a la etiqueta, con `[texto que aparece en la vista](#nombreSeccion)`  
$\hspace{.5cm}$ `$\hspace{.5cm}$` genera una tabulación 
