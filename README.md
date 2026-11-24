# SMARTPARK AI

Análisis local de ocupación de parqueaderos. El proyecto entrena dos CNN desde una inicialización aleatoria: una convolucional estándar y una con convoluciones separables en los últimos bloques. Ambas clasifican recortes de plazas como libres u ocupadas. La comparación de fondo queda disponible como línea base opcional.

## Reproducir la entrega

Ejecuta estos comandos desde la raíz con el entorno de **Instalación y pruebas** activado. MetaPKLot distribuye los JPG de PKLot y publica cajas de vehículos y polígonos de plazas en archivos COCO separados. La revisión fijada a continuación permite reconstruir la muestra. [Edición MetaPKLot](https://huggingface.co/datasets/DSBD-Research/MetaPKLot-Dataset) · [Fuente original de imágenes y cita de Almeida et al.](https://web.inf.ufpr.br/vri/databases/parking-lot-database/)

```powershell
. .\.venv\Scripts\Activate.ps1
python prepare_metapklot.py --download --revision b2b20b94053476ee6e51fad2b87f90679f31b30a
python train_models.py data/meta/plazas.csv --epochs 8 --seed 42
python train_vehicles.py data/meta/vehiculos.json --epochs 12 --seed 42
python train_assistant.py --epochs 45 --seed 42
python prepare_demo.py
python verify_demo.py
python -m streamlit run dashboard.py --server.address 127.0.0.1
```

Para reproducir con la muestra ya descargada, omite `--download`. El preparador verifica cada JPG contra las anotaciones. Entrenamiento y validación usan días distintos de UFPR04. La prueba usa UFPR05. ImageNet y otros pesos externos quedan deshabilitados mediante `weights=None` y `weights_backbone=None` en SSDLite. El detector elige época y umbral por F1 de validación con IoU ≥ 0,5 y registra TP, FP y FN en prueba con asignación uno a uno. El área anotada de las cámaras seleccionadas cubre el fotograma completo.

Para usar anotaciones propias, crea un manifiesto JSON con una lista de entradas como `{"image":"fotos/f01.jpg","group":"camara1/sesion1","split":"train","boxes":[[34,62,91,125]]}`. Cada caja está en píxeles `[x1,y1,x2,y2]`. Incluye una entrada por imagen, anota todos los vehículos visibles de la clase objetivo y usa `boxes: []` en imágenes realmente vacías. Usa `train`, `val` y `test` con cámaras o sesiones exclusivas y cajas en cada partición.

En **Monitoreo** selecciona `cnn`, activa **Detectar vehículos con cajas** y carga `data/demo-meta/entrada.mp4` junto a `data/demo-meta/plazas.json`. Consulta **Dashboard** y luego **Disponibilidad** con «¿Cuántas plazas hay libres?», «¿Dónde puedo estacionar?», «¿Qué porcentaje está ocupado?» y «¿Está lleno?». El MP4 de prueba presenta cuatro capturas reales espaciadas de UFPR05, repetidas tres veces cada una para aplicar la confirmación. Sus saltos de tiempo impiden evaluar movimiento continuo. Sustituye ambos archivos por un video continuo de cámara fija y su configuración para evaluar llegadas, salidas y tránsito por el pasillo.

```powershell
python app.py data/demo-meta/entrada.mp4 --checkpoint models/ocupacion/cnn.pt --vehicle-checkpoint models/vehiculos/vehicles.pt --spaces data/demo-meta/plazas.json --output results/terminal-meta.mp4
```

El contrato compartido sigue siendo `Occupancy`. `total` es el número de polígonos configurados. `available`, `occupied` y `pending` contienen los identificadores y sirven para obtener los conteos. `occupancy_percent` vale `100 × ocupadas / total` cuando hay confirmación y queda vacío mientras existen plazas pendientes. Ambos paneles leen la misma instantánea del último fotograma del trabajo activo. Las cajas vienen del detector y los estados de cada plaza vienen de la CNN y su filtro temporal.

El Transformer clasifica cinco intenciones a partir de texto en español. Las cifras y los identificadores los recupera la aplicación mediante el contrato anterior. La clase fuera de alcance y un umbral calibrado en validación permiten rechazar algunas consultas. `config/intenciones.json` fija las particiones, y `models/asistente/evaluation.json` guarda las respuestas de clasificación por pregunta y la matriz de confusión. Los informes locales de los modelos se generan en `models`, y la validación del MP4 en `results`.

## Entrenar los dos modelos

Si no puedes descargar PKLot, puedes ejecutar un piloto sin internet con el video de Sergey Rudakov ya disponible en `data/pexels-5957716.mp4`.

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python prepare_local_pilot.py
if ($LASTEXITCODE -eq 0) {
    & $python train_models.py "data/piloto-local/manifest.csv" --epochs 8 --seed 42
}
```

Este piloto usa 64 recortes anotados manualmente de ocho plazas de un solo video. Las particiones separan plazas físicas, pero comparten la misma toma. Sirve para comprobar el entrenamiento y la aplicación. Para conclusiones sobre nuevos parqueaderos, usa PKLot o videos anotados de otras cámaras. [Video y autor en Pexels](https://www.pexels.com/video/aerial-footage-of-a-parking-lot-5957716/)

Descarga el conjunto original desde la [Universidade Federal do Paraná](https://web.inf.ufpr.br/vri/databases/parking-lot-database/) y extrae el archivo en `data/PKLot`. El archivo original ocupa aproximadamente 4,6 GB y contiene imágenes y XML con etiquetas por plaza. El preparador agrupa por cámara y fecha para que los recortes del mismo grupo queden juntos.

Si el dominio de la universidad no abre, la [copia pública de PKLot en Hugging Face](https://huggingface.co/datasets/teenygrad/pklot/tree/main) documenta que conserva los JPG y XML originales. Descarga el archivo del conjunto de datos, extrae su contenido y apunta el preparador a la carpeta extraída.

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python prepare_pklot.py "data/PKLot" --output "data/pklot-preparado"
if ($LASTEXITCODE -eq 0) {
    & $python train_models.py "data/pklot-preparado/manifest.csv" --epochs 8 --seed 42
}
```

Para imágenes propias, se puede omitir el preparador y escribir un CSV con columnas `path,label,split,group`. Las rutas se interpretan desde la carpeta del CSV. Usa etiqueta 0 para libre y 1 para ocupada, y particiones `train`, `val` y `test`. Cada partición requiere ambas clases. Usa una cámara o sesión de captura como grupo y evita repartir imágenes consecutivas entre particiones.

El entrenamiento usa AdamW y entropía cruzada. Se conserva para cada arquitectura la época con mayor F1 de validación y se mide en la partición de prueba. Los archivos `cnn.pt`, `cnn-separable.pt` y `evaluation.json` se generan en `models/ocupacion`. La evaluación registra precisión, recall, F1, accuracy, matriz de confusión, parámetros y tiempo de entrenamiento. Los checkpoints se generan aquí desde cero.

## Ejecutar la aplicación web

Desde PowerShell en la carpeta del proyecto:

```powershell
& .\.venv\Scripts\python.exe -m streamlit run dashboard.py --server.address 127.0.0.1
```

En **Monitoreo**, selecciona uno de los modelos entrenados y carga el MP4 de cámara fija y el JSON de polígonos. Si eliges **Comparación de fondo**, carga también una imagen de las plazas vacías, con el mismo encuadre y resolución.

Inicia el análisis. El Dashboard muestra indicadores, cambios de ocupación e historial CSV. **Evaluación** presenta la comparación cuando existe un informe de entrenamiento. La consulta de disponibilidad usa los resultados del último fotograma.

## Delimitar plazas

Con un video propio guardado en `data`:

```powershell
& .\.venv\Scripts\python.exe edit_spaces.py "data/mi-video.mp4" "config/mis-plazas.json"
```

Clic izquierdo agrega un vértice. Enter confirma el polígono. Clic derecho selecciona una plaza existente. D la elimina, S guarda y Esc cierra. Usa polígonos convexos sin superposición. El JSON almacena coordenadas normalizadas.

## Ejecutar por terminal

```powershell
$salida = "results/ocupacion-$(Get-Date -Format 'yyyyMMdd-HHmmss').mp4"
& .\.venv\Scripts\python.exe app.py "data/mi-video.mp4" --checkpoint "models/ocupacion/cnn.pt" --spaces "config/mis-plazas.json" --output $salida
if ($LASTEXITCODE -eq 0) { Invoke-Item $salida }
```

Los videos previos de dron y los MP4 ya exportados corresponden a ensayos históricos. Los polígonos de las plazas suponen una cámara con posición, zoom y resolución estables.

## Algoritmo y alcance

Las CNN analizan cada recorte. Sus polígonos muestran el estado de las plazas configuradas. SSDLite analiza el fotograma completo y dibuja cajas de vehículos, incluidas detecciones fuera de las plazas.

La línea base agrupa cambios visuales respecto al fondo y los asocia con las plazas. Las CNN y la línea base comparten el filtro temporal. Se confirman los estados después de tres fotogramas consecutivos y se muestran polígonos rojos, verdes o grises. La ocupación se calcula como plazas ocupadas divididas por plazas configuradas, por cien.

Para evaluar un video continuo de cámara fija, registra la ocupación manual y las cajas de varios fotogramas antes de comparar con el MP4 anotado.

## Instalación y pruebas

En un entorno nuevo con Python 3.12 de 64 bits:

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
& .\.venv\Scripts\python.exe -m pytest -q
```

## Informe y guion de sustentación

### Resultados de ocupación

La muestra contiene 104 JPG, 3.125 cajas y 3.928 recortes, repartidos en 2.608 de entrenamiento, 584 de validación y 736 de prueba. La matriz usa filas reales y columnas predichas en el orden libre, ocupada. La clase positiva es ocupada. Tiempos de esta ejecución en CPU.

| Modelo | Accuracy | Precisión | Recall | F1 | Matriz | Tiempo | Predicciones libre / ocupada |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| cnn | 66,17 % | 55,61 % | 100 % | 71,48 % | `[[175,249],[0,312]]` | 34,3 s | 175 / 561 |
| cnn-separable | 67,26 % | 56,42 % | 100 % | 72,14 % | `[[183,241],[0,312]]` | 35,7 s | 183 / 553 |

Las dos CNN predijeron ambas clases. Validación favoreció la época 8 en ambos casos. Los falsos positivos predominan en UFPR05. El piloto original se reprodujo aparte con 16 recortes de prueba. Dio 50 % en ambos modelos, con matriz `[[0,8],[0,8]]` para cnn y `[[8,0],[8,0]]` para cnn-separable.

El Transformer obtuvo 11/22 aciertos en preguntas de prueba y 11/16 en validación, con 82 ejemplos de entrenamiento y 14.725 parámetros. Acertó las cuatro consultas de la demostración. Algunas paráfrasis y preguntas ajenas al parqueadero se confundieron con intenciones válidas.

SSDLite obtuvo **8,99 % de precisión, 11,89 % de recall y 10,24 % de F1** en los 16 JPG de prueba (IoU ≥ 0,5, 54 TP, 547 FP y 400 FN). El MP4 contiene cajas dibujadas por el modelo, incluidas cajas que no corresponden a vehículos. La detección necesita mayor cobertura de entrenamiento y otra evaluación.

La revisión visual de las 12 plazas y las etiquetas de referencia dio estos conteos. La sesión de Streamlit de prueba recibió los bytes del MP4 y del JSON, procesó el video, mostró el reproductor y contestó las cuatro consultas usando el resultado final. El asistente devolvió P004 y P012 como libres y 83,3 % ocupado, igual que el Dashboard. La comprobación independiente mostró que seis plazas estaban ocupadas al final.

| Fotograma | Conteo visual y referencia | Predicción de ocupadas | Cajas predichas |
| --- | ---: | ---: | ---: |
| 3 | 1/12 (8,3 %) | 9/12 (75,0 %) | 36 |
| 6 | 12/12 (100 %) | 12/12 (100 %) | 52 |
| 9 | 12/12 (100 %) | 12/12 (100 %) | 49 |
| 12 | 6/12 (50,0 %) | 10/12 (83,3 %) | 40 |

En el caso controlado, Dashboard y Disponibilidad mostraron 10 plazas libres, dos ocupadas, 16,7 % y los mismos identificadores. La caja de prueba en el pasillo no alteró esos estados. También se verificaron el caso sin resultados en Streamlit y los casos lleno, pendiente y fuera de alcance en las pruebas de respuestas.

### Guion de 10 minutos

| Tiempo | Qué mostrar |
| --- | --- |
| 0:00–1:00 | Presentar el problema y mostrar los polígonos y los tres estados. |
| 1:00–2:30 | Mostrar fuente, anotaciones de cajas y plazas, y separación de cámaras y días. |
| 2:30–4:00 | Comparar las dos CNN y el piloto. Explicar los falsos positivos en la cámara de prueba. |
| 4:00–6:00 | Cargar el MP4 y los polígonos en Monitoreo. Activar las cajas. Mostrar el conteo visual frente al porcentaje predicho. |
| 6:00–7:30 | Abrir Dashboard y consultar Disponibilidad en la misma sesión. Verificar cifras e identificadores. |
| 7:30–8:30 | Mostrar matriz del Transformer y confusiones con consultas fuera de alcance. |
| 8:30–10:00 | Explicar límites de la muestra, del montaje de cámara fija y de la validación en video continuo. |

### Preguntas de sustentación

**¿Por qué hay dos CNN con los mismos datos?** Usamos una CNN convencional y una separable con las mismas particiones para que la comparación mida la diferencia entre arquitecturas bajo el mismo protocolo. En UFPR05 la separable logró 67,26 % y la convencional 66,17 %. Una sola cámara de prueba limita la generalización de esa diferencia.

**¿Qué demuestra el 100 % que salió en el piloto?** La aplicación divide ocupadas predichas por plazas configuradas. Predijo 5 de 5, por lo que el 100 % era matemáticamente correcto, pero en ese ejemplo la predicción de las cinco plazas era incorrecta. El conteo manual permite verificar las etiquetas, y el piloto repetido mostró que ambos modelos colapsaban a una clase.

**¿Por qué las cajas y los estados tienen modelos distintos?** El detector localiza vehículos en cualquier parte del encuadre. Las CNN clasifican cada plaza y el filtro exige tres fotogramas consecutivos para actualizar su estado. La prueba controlada introduce una caja en el pasillo y conserva el conteo determinado por las plazas. La CNN aún puede confundirse si un vehículo tapa parte del recorte.

**¿Cómo se reduce la fuga de datos?** Entrenamiento y validación usan días distintos de UFPR04. Prueba usa UFPR05, una cámara ausente del entrenamiento. El manifiesto y las comprobaciones de grupo permiten repetir la separación. Las sesiones de una misma cámara pueden compartir vehículos y configuración, y la muestra abarca solo dos cámaras.

**¿De dónde salen las cifras del asistente?** El Transformer clasifica la pregunta. La aplicación lee el estado del último fotograma que también usa el Dashboard. La verificación web compara conteos, porcentaje e identificadores. Las 11 respuestas correctas de 22 preguntas nuevas muestran que todavía se confunde con vocabulario y formulaciones distintas.
