# Reconocimiento Facial

Reconoce y manipula rostros desde Python o desde la línea de comandos con
la biblioteca de reconocimiento facial más simple del mundo.

Construido usando el reconocimiento facial de última generación de [dlib](http://dlib.net/)
con aprendizaje profundo. El modelo tiene una precisión del 99.38% en el
benchmark [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/).

Esto también proporciona una herramienta de línea de comandos simple `face_recognition` que te permite
hacer reconocimiento facial en una carpeta de imágenes desde la línea de comandos.


[![PyPI](https://img.shields.io/pypi/v/face_recognition.svg)](https://pypi.python.org/pypi/face_recognition)
[![Build Status](https://github.com/ageitgey/face_recognition/workflows/CI/badge.svg?branch=master&event=push)](https://github.com/ageitgey/face_recognition/actions?query=workflow%3ACI)
[![Documentation Status](https://readthedocs.org/projects/face-recognition/badge/?version=latest)](http://face-recognition.readthedocs.io/en/latest/?badge=latest)

## Características

#### Encontrar rostros en imágenes

Encuentra todos los rostros que aparecen en una imagen:

![](https://cloud.githubusercontent.com/assets/896692/23625227/42c65360-025d-11e7-94ea-b12f28cb34b4.png)

```python
import face_recognition
image = face_recognition.load_image_file("tu_archivo.jpg")
face_locations = face_recognition.face_locations(image)
```

#### Encontrar y manipular características faciales en imágenes

Obtén las ubicaciones y contornos de los ojos, nariz, boca y mentón de cada persona.

![](https://cloud.githubusercontent.com/assets/896692/23625282/7f2d79dc-025d-11e7-8728-d8924596f8fa.png)

```python
import face_recognition
image = face_recognition.load_image_file("tu_archivo.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)
```

Encontrar características faciales es súper útil para muchas cosas importantes. Pero también puedes usarlo para cosas divertidas
como aplicar [maquillaje digital](https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py) (piensa en 'Meitu'):

![](https://cloud.githubusercontent.com/assets/896692/23625283/80638760-025d-11e7-80a2-1d2779f7ccab.png)

#### Identificar rostros en imágenes

Reconoce quién aparece en cada foto.

![](https://cloud.githubusercontent.com/assets/896692/23625229/45e049b6-025d-11e7-89cc-8a71cf89e713.png)

```python
import face_recognition
known_image = face_recognition.load_image_file("biden.jpg")
unknown_image = face_recognition.load_image_file("desconocido.jpg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
```

Incluso puedes usar esta biblioteca con otras bibliotecas de Python para hacer reconocimiento facial en tiempo real:

![](https://cloud.githubusercontent.com/assets/896692/24430398/36f0e3f0-13cb-11e7-8258-4d0c9ce1e419.gif)

Ve [este ejemplo](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py) para el código.

## Demostraciones en Línea

Demostración de Jupyter notebook contribuida por usuarios (no soportada oficialmente): [![Deepnote](https://beta.deepnote.org/buttons/try-in-a-jupyter-notebook.svg)](https://beta.deepnote.org/launch?template=face_recognition)

## Instalación

### Requisitos

  * Python 3.3+ o Python 2.7
  * macOS o Linux (Windows no está oficialmente soportado, pero podría funcionar)

### Opciones de Instalación:

#### Instalando en Mac o Linux

Primero, asegúrate de tener dlib ya instalado con enlaces de Python:

  * [Cómo instalar dlib desde el código fuente en macOS o Ubuntu](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)

Luego, asegúrate de tener cmake instalado:

```brew install cmake```

Finalmente, instala este módulo desde pypi usando `pip3` (o `pip2` para Python 2):

```bash
pip3 install face_recognition
```

Alternativamente, puedes probar esta biblioteca con [Docker](https://www.docker.com/), ve [esta sección](#despliegue).

Si tienes problemas con la instalación, también puedes probar una
[VM preconfigurada](https://medium.com/@ageitgey/try-deep-learning-in-python-now-with-a-fully-pre-configured-vm-1d97d4c3e9b).

#### Instalando en una placa Nvidia Jetson Nano

 * [Instrucciones de instalación para Jetson Nano](https://medium.com/@ageitgey/build-a-hardware-based-face-recognition-system-for-150-with-the-nvidia-jetson-nano-and-python-a25cb8c891fd)
   * Por favor sigue las instrucciones del artículo cuidadosamente. Actualmente hay un error en las bibliotecas CUDA del Jetson Nano que causará que esta biblioteca falle silenciosamente si no sigues las instrucciones del artículo para comentar una línea en dlib y recompilarlo.

#### Instalando en Raspberry Pi 2+

  * [Instrucciones de instalación para Raspberry Pi 2+](https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65)

#### Instalando en FreeBSD

```bash
pkg install graphics/py-face_recognition
```

#### Instalando en Windows

Aunque Windows no está oficialmente soportado, usuarios útiles han publicado instrucciones sobre cómo instalar esta biblioteca:

  * [Guía de instalación de @masoudr para Windows 10 (dlib + face_recognition)](https://github.com/ageitgey/face_recognition/issues/175#issue-257710508)

#### Instalando una imagen de Máquina Virtual preconfigurada

  * [Descargar la imagen de VM preconfigurada](https://medium.com/@ageitgey/try-deep-learning-in-python-now-with-a-fully-pre-configured-vm-1d97d4c3e9b) (para VMware Player o VirtualBox).

## Uso

### Interfaz de Línea de Comandos

Cuando instalas `face_recognition`, obtienes dos programas de línea de comandos simples:

* `face_recognition` - Reconoce rostros en una fotografía o carpeta llena de fotografías.
* `face_detection` - Encuentra rostros en una fotografía o carpeta llena de fotografías.

#### Herramienta de línea de comandos `face_recognition`

El comando `face_recognition` te permite reconocer rostros en una fotografía o
carpeta llena de fotografías.

Primero, necesitas proporcionar una carpeta con una imagen de cada persona que
ya conoces. Debe haber un archivo de imagen para cada persona con los
archivos nombrados según quién está en la imagen:

![known](https://cloud.githubusercontent.com/assets/896692/23582466/8324810e-00df-11e7-82cf-41515eba704d.png)

Luego, necesitas una segunda carpeta con los archivos que quieres identificar:

![unknown](https://cloud.githubusercontent.com/assets/896692/23582465/81f422f8-00df-11e7-8b0d-75364f641f58.png)

Entonces simplemente ejecutas el comando `face_recognition`, pasando la
carpeta de personas conocidas y la carpeta (o imagen individual) con personas
desconocidas y te dice quién está en cada imagen:

```bash
$ face_recognition ./imagenes_de_personas_que_conozco/ ./imagenes_desconocidas/

/imagenes_desconocidas/desconocido.jpg,Barack Obama
/imagenes_desconocidas/desconocido.jpg,persona_desconocida
```

Hay una línea en la salida para cada rostro. Los datos están separados por comas
con el nombre del archivo y el nombre de la persona encontrada.

Una `persona_desconocida` es un rostro en la imagen que no coincidió con nadie en
tu carpeta de personas conocidas.

#### Herramienta de línea de comandos `face_detection`

El comando `face_detection` te permite encontrar la ubicación (coordenadas de píxeles)
de cualquier rostro en una imagen.

Simplemente ejecuta el comando `face_detection`, pasando una carpeta de imágenes
para verificar (o una sola imagen):

```bash
$ face_detection ./carpeta_con_imagenes/

examples/imagen1.jpg,65,215,169,112
examples/imagen2.jpg,62,394,211,244
examples/imagen2.jpg,95,941,244,792
```

Imprime una línea para cada rostro detectado. Las coordenadas
reportadas son las coordenadas superior, derecha, inferior e izquierda del rostro (en píxeles).

##### Ajustando Tolerancia / Sensibilidad

Si estás obteniendo múltiples coincidencias para la misma persona, podría ser que
las personas en tus fotos se ven muy similares y se necesita un valor de tolerancia más bajo
para hacer las comparaciones de rostros más estrictas.

Puedes hacer eso con el parámetro `--tolerance`. El valor de tolerancia
predeterminado es 0.6 y números más bajos hacen las comparaciones de rostros más estrictas:

```bash
$ face_recognition --tolerance 0.54 ./imagenes_de_personas_que_conozco/ ./imagenes_desconocidas/

/imagenes_desconocidas/desconocido.jpg,Barack Obama
/imagenes_desconocidas/desconocido.jpg,persona_desconocida
```

Si quieres ver la distancia facial calculada para cada coincidencia para
ajustar la configuración de tolerancia, puedes usar `--show-distance true`:

```bash
$ face_recognition --show-distance true ./imagenes_de_personas_que_conozco/ ./imagenes_desconocidas/

/imagenes_desconocidas/desconocido.jpg,Barack Obama,0.378542298956785
/imagenes_desconocidas/desconocido.jpg,persona_desconocida,None
```

##### Más Ejemplos

Si simplemente quieres saber los nombres de las personas en cada fotografía pero no te
importan los nombres de archivo, podrías hacer esto:

```bash
$ face_recognition ./imagenes_de_personas_que_conozco/ ./imagenes_desconocidas/ | cut -d ',' -f2

Barack Obama
persona_desconocida
```

##### Acelerando el Reconocimiento Facial

El reconocimiento facial se puede hacer en paralelo si tienes una computadora con
múltiples núcleos de CPU. Por ejemplo, si tu sistema tiene 4 núcleos de CPU, puedes
procesar aproximadamente 4 veces más imágenes en la misma cantidad de tiempo usando
todos tus núcleos de CPU en paralelo.

Si estás usando Python 3.4 o más reciente, pasa un parámetro `--cpus <numero_de_nucleos_de_cpu_a_usar>`:

```bash
$ face_recognition --cpus 4 ./imagenes_de_personas_que_conozco/ ./imagenes_desconocidas/
```

También puedes pasar `--cpus -1` para usar todos los núcleos de CPU en tu sistema.

#### Módulo de Python

Puedes importar el módulo `face_recognition` y luego manipular fácilmente
rostros con solo un par de líneas de código. ¡Es súper fácil!

Documentación de API: [https://face-recognition.readthedocs.io](https://face-recognition.readthedocs.io/en/latest/face_recognition.html).

##### Encontrar automáticamente todos los rostros en una imagen

```python
import face_recognition

image = face_recognition.load_image_file("mi_imagen.jpg")
face_locations = face_recognition.face_locations(image)

# face_locations ahora es un array listando las coordenadas de cada rostro!
```

Ve [este ejemplo](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py)
 para probarlo.

También puedes optar por un modelo de detección de rostros basado en aprendizaje profundo algo más preciso.

Nota: Se requiere aceleración GPU (a través de la biblioteca CUDA de NVidia) para un buen
rendimiento con este modelo. También querrás habilitar el soporte CUDA
al compilar `dlib`.

```python
import face_recognition

image = face_recognition.load_image_file("mi_imagen.jpg")
face_locations = face_recognition.face_locations(image, model="cnn")

# face_locations ahora es un array listando las coordenadas de cada rostro!
```

Ve [este ejemplo](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture_cnn.py)
 para probarlo.

Si tienes muchas imágenes y una GPU, también puedes
[encontrar rostros en lotes](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_batches.py).

##### Localizar automáticamente las características faciales de una persona en una imagen

```python
import face_recognition

image = face_recognition.load_image_file("mi_imagen.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)

# face_landmarks_list ahora es un array con las ubicaciones de cada característica facial en cada rostro.
# face_landmarks_list[0]['left_eye'] sería la ubicación y contorno del ojo izquierdo de la primera persona.
```

Ve [este ejemplo](https://github.com/ageitgey/face_recognition/blob/master/examples/find_facial_features_in_picture.py)
 para probarlo.

##### Reconocer rostros en imágenes e identificar quiénes son

```python
import face_recognition

picture_of_me = face_recognition.load_image_file("yo.jpg")
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

# my_face_encoding ahora contiene una 'codificación' universal de mis características faciales que se puede comparar con cualquier otra imagen de un rostro!

unknown_picture = face_recognition.load_image_file("desconocido.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

# Ahora podemos ver si las dos codificaciones faciales son de la misma persona con `compare_faces`!

results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

if results[0] == True:
    print("¡Es una foto mía!")
else:
    print("¡No es una foto mía!")
```

Ve [este ejemplo](https://github.com/ageitgey/face_recognition/blob/master/examples/recognize_faces_in_pictures.py)
 para probarlo.

## Ejemplos de Código Python

Todos los ejemplos están disponibles [aquí](https://github.com/ageitgey/face_recognition/tree/master/examples).


#### Detección de Rostros

* [Encontrar rostros en una fotografía](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py)
* [Encontrar rostros en una fotografía (usando aprendizaje profundo)](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture_cnn.py)
* [Encontrar rostros en lotes de imágenes con GPU (usando aprendizaje profundo)](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_batches.py)
* [Desenfocar todos los rostros en un video en vivo usando tu webcam (Requiere OpenCV instalado)](https://github.com/ageitgey/face_recognition/blob/master/examples/blur_faces_on_webcam.py)

#### Características Faciales

* [Identificar características faciales específicas en una fotografía](https://github.com/ageitgey/face_recognition/blob/master/examples/find_facial_features_in_picture.py)
* [Aplicar maquillaje digital (horriblemente feo)](https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py)

#### Reconocimiento Facial

* [Encontrar y reconocer rostros desconocidos en una fotografía basándose en fotografías de personas conocidas](https://github.com/ageitgey/face_recognition/blob/master/examples/recognize_faces_in_pictures.py)
* [Identificar y dibujar recuadros alrededor de cada persona en una foto](https://github.com/ageitgey/face_recognition/blob/master/examples/identify_and_draw_boxes_on_faces.py)
* [Comparar rostros por distancia facial numérica en lugar de solo coincidencias Verdadero/Falso](https://github.com/ageitgey/face_recognition/blob/master/examples/face_distance.py)
* [Reconocer rostros en video en vivo usando tu webcam - Versión Simple / Más Lenta (Requiere OpenCV instalado)](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam.py)
* [Reconocer rostros en video en vivo usando tu webcam - Versión Más Rápida (Requiere OpenCV instalado)](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py)
* [Reconocer rostros en un archivo de video y escribir un nuevo archivo de video (Requiere OpenCV instalado)](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_video_file.py)
* [Reconocer rostros en una Raspberry Pi con cámara](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_on_raspberry_pi.py)
* [Ejecutar un servicio web para reconocer rostros vía HTTP (Requiere Flask instalado)](https://github.com/ageitgey/face_recognition/blob/master/examples/web_service_example.py)
* [Reconocer rostros con un clasificador K-vecinos más cercanos](https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_knn.py)
* [Entrenar múltiples imágenes por persona y luego reconocer rostros usando un SVM](https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_svm.py)

## Creando un Ejecutable Independiente
Si quieres crear un ejecutable independiente que pueda ejecutarse sin necesidad de instalar `python` o `face_recognition`, puedes usar [PyInstaller](https://github.com/pyinstaller/pyinstaller). Sin embargo, requiere alguna configuración personalizada para funcionar con esta biblioteca. Ve [este issue](https://github.com/ageitgey/face_recognition/issues/357) para saber cómo hacerlo.

## Artículos y Guías que cubren `face_recognition`

- Mi artículo sobre cómo funciona el Reconocimiento Facial: [Reconocimiento Facial Moderno con Aprendizaje Profundo](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)
  - Cubre los algoritmos y cómo funcionan en general
- [Reconocimiento facial con OpenCV, Python y aprendizaje profundo](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/) por Adrian Rosebrock
  - Cubre cómo usar el reconocimiento facial en la práctica
- [Reconocimiento Facial en Raspberry Pi](https://www.pyimagesearch.com/2018/06/25/raspberry-pi-face-recognition/) por Adrian Rosebrock
  - Cubre cómo usar esto en una Raspberry Pi
- [Agrupación de rostros con Python](https://www.pyimagesearch.com/2018/07/09/face-clustering-with-python/) por Adrian Rosebrock
  - Cubre cómo agrupar automáticamente fotos basándose en quién aparece en cada foto usando aprendizaje no supervisado

## Cómo Funciona el Reconocimiento Facial

Si quieres aprender cómo funcionan la localización y el reconocimiento de rostros en lugar de
depender de una biblioteca de caja negra, [lee mi artículo](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78).

## Advertencias

* El modelo de reconocimiento facial está entrenado en adultos y no funciona muy bien en niños. Tiende a mezclar
  niños bastante fácilmente usando el umbral de comparación predeterminado de 0.6.
* La precisión puede variar entre grupos étnicos. Por favor ve [esta página wiki](https://github.com/ageitgey/face_recognition/wiki/Face-Recognition-Accuracy-Problems#question-face-recognition-works-well-with-european-individuals-but-overall-accuracy-is-lower-with-asian-individuals) para más detalles.

## <a name="despliegue">Despliegue en Hosts en la Nube (Heroku, AWS, etc)</a>

Dado que `face_recognition` depende de `dlib` que está escrito en C++, puede ser complicado desplegar una aplicación
que lo use en un proveedor de hosting en la nube como Heroku o AWS.

Para facilitar las cosas, hay un Dockerfile de ejemplo en este repositorio que muestra cómo ejecutar una aplicación construida con
`face_recognition` en un contenedor [Docker](https://www.docker.com/). Con eso, deberías poder desplegarlo
en cualquier servicio que soporte imágenes Docker.

Puedes probar la imagen Docker localmente ejecutando: `docker-compose up --build`

También hay [varias imágenes Docker preconstruidas.](docker/README.md)

Los usuarios de Linux con una GPU (drivers >= 384.81) y [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker) instalado pueden ejecutar el ejemplo en la GPU: Abre el archivo [docker-compose.yml](docker-compose.yml) y descomenta las líneas `dockerfile: Dockerfile.gpu` y `runtime: nvidia`.

## ¿Tienes problemas?

Si encuentras problemas, por favor lee la sección [Errores Comunes](https://github.com/ageitgey/face_recognition/wiki/Common-Errors) del wiki antes de crear un issue en github.

## Agradecimientos

* Muchas, muchas gracias a [Davis King](https://github.com/davisking) ([@nulhom](https://twitter.com/nulhom))
  por crear dlib y por proporcionar los modelos de detección de características faciales y codificación de rostros entrenados
  usados en esta biblioteca. Para más información sobre el ResNet que impulsa las codificaciones faciales, revisa
  su [publicación de blog](http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html).
* Gracias a todos los que trabajan en todas las increíbles bibliotecas de ciencia de datos de Python como numpy, scipy, scikit-image,
  pillow, etc, etc que hacen este tipo de cosas tan fáciles y divertidas en Python.
* Gracias a [Cookiecutter](https://github.com/audreyr/cookiecutter) y la plantilla de proyecto
  [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
  por hacer el empaquetado de proyectos Python mucho más tolerable.
