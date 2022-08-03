## Configuración de git

- git config -l : Lista toda la config del git
- git config --global user.email johndoe@example.com : Configura el mail de usuario
- git config --global user.name "John Doe": Configura el nombre de usuario

## Configuración de repo localmente

Si la carpeta es un repositorio de git, tendrá una carpeta oculta llamada .git
- git clone url : Clona el repo en una carpeta nueva, genera, la carpeta oculta .git
- git init : Inicia la carpeta como repo
- git remote add origin url : Conecta la carpeta a un repo en github (requiere el git init previo).
- git remote set-url origin url : Para modificar la conexión a otro repo
- git remote -v : Chequea con que repo (url) está conectado


## Descargar cambios del repo online al local (pull)

En caso de que el repo local esté atrasado frente al online. 
- git checkout codCommit : Trae la versión que se le pida del repo. Si se pone master/main trae la actual
- git pull origin master: Trae los cambios del repo en github que no estén en local
- git fecth : modifica los metadatos, y trae también las ramas nuevas

## Branches

git branch: Muestra todas las ramas, marcando la activa
git branch --show-current : Muestra sólo la rama activa
git branch feature/(nombreX) : Crea una rama
git checkout -b feature/(nombreX) : Crea y se ubica en la rama "feature/(nombreX)"
git checkout develop : Vuelve a la rama develop
git log --oneline --decorate --graph --all : Muestra el historial de todas las commit
git push --set-upstream origin feature/(nombreX) : Pushea a la rama esa



## Realizar una subida al repo online (commit)

- git add .  : Agregar/stage todas las modificaciones
- git commit -m "mensaje"  : Generar un commit
- git log origin/main..HEAD : Muestra cuales son los commit hechos localmente por delante del repo en gihthub
- git reset --soft HEAD~X : Resetea los últimos commit locales no pusheados (x es el número de últimos commits)

- git push origin main/master : Pushear/enviar todos los commits creados, a la rama ppal. 

## Misceláneos

- git status : Da el estado del repo local
- git log: Te muestra el log del local  (tiene varias extras como --oneline o --graph)

## .gitignore

- touch .gitignore : Desde consola ubicandote en el root del repo. Crea el archivo .gitignore
- git rm --cached *.csv : Elimina del cache de commits los archivos .csv (cambiar por otro tipo de archivo). *Esto sirve se se realizó un commit previo y luego se generó el .gitignore*

- Documentación interesante:
    - https://docs.github.com/es/get-started/getting-started-with-git/ignoring-files
    - https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository#_ignoring  
    - https://github.com/github/gitignore listado de ejemplos





