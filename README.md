# SMARTPARK AI

Análisis local de ocupación de parqueaderos. El proyecto entrena dos CNN desde una inicialización aleatoria: una convolucional estándar y una con convoluciones separables en los últimos bloques. Ambas clasifican recortes de plazas como libres u ocupadas. La comparación de fondo queda disponible como línea base opcional.

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

Las CNN analizan directamente cada recorte. Sus polígonos muestran el estado de las plazas configuradas. La localización independiente de vehículos en pasillos o fuera de las plazas exige un detector adicional entrenado con cajas anotadas.

La línea base agrupa cambios visuales respecto al fondo y los asocia con las plazas. Las CNN y la línea base comparten el filtro temporal. Se confirman los estados después de tres fotogramas consecutivos y se muestran polígonos rojos, verdes o grises. La ocupación se calcula como plazas ocupadas divididas por plazas configuradas, por cien.

La evaluación en una cámara real requiere anotaciones. La consulta de disponibilidad por reglas queda como base verificable. El requisito de un Transformer funcional requerirá entrenar desde cero un modelo compacto de clasificación de preguntas en español, con respuestas estructuradas calculadas por la aplicación.

## Instalación y pruebas

En un entorno nuevo con Python 3.12 de 64 bits:

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
& .\.venv\Scripts\python.exe -m pytest -q
```
