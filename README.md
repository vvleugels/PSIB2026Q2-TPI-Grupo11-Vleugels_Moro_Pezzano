# PSIB 2026 Q2 — TPI — Grupo 11

Segmentación y cuantificación de lesiones hepáticas en cortes axiales de TC.
Integrantes: Vleugels, Moro y Pezzano.

La solución principal aplica una ventana HU, suavizado gaussiano, Otsu y
morfología dentro de una **máscara hepática de referencia**. La selección de
cortes también utiliza anotaciones. No es un detector autónomo sobre TC sin
máscara, ni una validación clínica. La GUI permite inspeccionar casos, ajustar
parámetros y ejecutar alternativas experimentales.

## Datos y alcance

Se utilizó **Medical Segmentation Decathlon, Task03 Liver**, con 131 casos
etiquetados. Se ajustaron parámetros con un subconjunto de diez pacientes
(3, 9, 45, 57, 76, 94, 99, 101, 105 y 117). Según lo informado por el equipo,
posteriormente se probó el método en cinco cortes de cada uno de los 131
pacientes en otra computadora. Esa ejecución no se repitió al preparar este
repositorio y aquí no se incluyen sus resultados ni se atribuyen métricas.
El conjunto completo incluye los pacientes utilizados para ajustar parámetros;
por eso no se presenta como una evaluación independiente.

Fuente: [Medical Segmentation Decathlon](https://medicaldecathlon.com/).
Referencia: Antonelli et al., *A large annotated medical image dataset for the
development and evaluation of segmentation algorithms*,
[arXiv:1902.09063](https://arxiv.org/abs/1902.09063).
Los metadatos de origen conservados por el proyecto indican CC-BY-SA 4.0;
conservar la atribución y los avisos originales del dataset.

Los volúmenes no se distribuyen en este repositorio. Colocar los pares en:

```text
data/Task03_Liver/
  imagesTr/liver_0.nii.gz
  labelsTr/liver_0.nii.gz
  ...
```

Los identificadores no tienen que ser consecutivos. También se admiten pares
`volume-N.nii[.gz]` y `segmentation-N.nii[.gz]` dentro de la carpeta configurada.
Las etiquetas son 0=fondo, 1=hígado y 2=tumor. Se requieren imágenes y máscaras
alineadas; las imágenes sin anotación de `imagesTs` no son entradas de este flujo.

## Instalación y ejecución

Usar Python 3.12 de 64 bits. Ejecutar los comandos desde la raíz del proyecto.

### Windows

Instalar Python 3.12 con el lanzador `py` y abrir `PSIB.bat`. Crea `.venv`,
comprueba dependencias y abre la GUI. La instalación inicial requiere internet.
Para trabajar por consola:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip check
python scripts/gui.py
```

### Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
./liver_tumor.sh
```

La GUI requiere Tk y una sesión gráfica. En Ubuntu/Debian, instalar el paquete
Tk correspondiente al Python utilizado (por ejemplo, `python3-tk`). El lanzador
Linux utiliza el entorno `.venv` creado en este directorio. No copiar entornos
virtuales entre equipos o sistemas.

## Configuración final

`config.yaml` selecciona **131 pacientes y hasta cinco cortes por paciente**,
con semilla 2026 y separación mínima de 10 mm. Selecciona cortes con tumor
mayor, mediano, pequeño, percentil 75 y uno sin tumor. Si no puede completar
los roles, informa la limitación y puede producir menos cortes.

El método principal usa ventana centro 30/ancho 150 HU, gaussiano σ=1 píxel,
sin CLAHE, contraste 1, Otsu, apertura y cierre con disco de radio 2 píxeles,
relleno de huecos y área mínima de 50 píxeles.

Para otro conjunto, cambiar `data_dir` y `selection.n_cases`. Para IDs concretos,
agregar `selection.case_ids: [3, 9]`: esta lista prevalece sobre `n_cases`.
Para evaluar todos los disponibles, quitar esa lista y establecer su cantidad.
No se reduce automáticamente un pedido de 131 casos si faltan datos.

Puede copiarse el archivo completo a `config.local.yaml`, ignorado por Git.
Ese archivo **reemplaza** la configuración base; no la combina. Las rutas
relativas se resuelven respecto del YAML seleccionado. Los comandos que admiten
`--config` permiten elegir otro archivo; un `config.local.yaml` a su lado también
lo reemplaza. Las salidas se escriben bajo `output_dir` (por defecto `results/`).

## Uso de la GUI

La interfaz sigue cuatro pasos, con **◀ Anterior / Siguiente ▶** abajo; también
se puede saltar a cualquier paso desde las pestañas. Cada paso muestra solo lo
esencial; el resto está en paneles plegables (▸).

1. **Casos**: número de casos y cortes por caso, y **Generar selección**.
   Plegados: *Elegir casos específicos* (si hay casos marcados se usan en lugar
   de la muestra aleatoria) y *Datos y configuración* (archivo YAML, abrir un
   volumen LiTS, restaurar parámetros).
2. **Ajustar**: elegir caso y corte, método de umbral y área mínima, y
   **Probar en este corte** (Dice/IoU del corte). Plegados: *Preprocesamiento*,
   *Morfología*, *Visualización* y *Avanzado · detectores experimentales*.
3. **Ejecutar**: resumen de lo que se va a correr, **▶ Ejecutar flujo 2D** y el
   registro en vivo. La selección de cortes se regenera al ejecutar.
4. **Resultados**: resumen de la última corrida (Dice medio y tumores medianos
   10–20 mm y grandes ≥20 mm cubiertos), **Tabla por caso…**, figuras y tablas
   generadas. Plegados: *Análisis 3D* (**Analizar 3D**, **Segmentar en 3D**) y
   *Compresión y reconstrucción* (requieren la selección del paso 1).

En *Avanzado* están los presets `hu_sensitive`, `default` y `otsu_059` y los
detectores locales de `config/presets/` (**Cargar preset local YAML**). Son
alternativas experimentales; `default` es un nombre histórico y **no** el Otsu
final. Mientras un detector está activo, el paso 2 muestra un aviso y el paso 3
lo indica; **Restaurar parámetros del config** vuelve al método final. La
interfaz permite cancelar trabajos (**■ Cancelar**) y guarda sus preferencias en
`.psib_gui/`.

## Flujo reproducible por consola

Con el entorno activado:

```bash
# Contextualización: inventario de una muestra de hasta 15 casos y figuras.
python scripts/explore_dataset.py
# Generar selección y ejecutar la segmentación principal.
python scripts/run_tpi.py
# Mejoramiento: imágenes antes/después y comparación de filtros.
python scripts/preprocess_selected.py
python scripts/compare_filters.py
# Geometría y texturas GLCM sobre regiones de referencia.
python scripts/extract_features.py
python scripts/plot_features.py
# Comparación adicional de cuantización.
python scripts/extract_features.py --levels 8 16 32
# Compresión/descompresión y reconstrucción simulada.
python scripts/run_compression.py
python scripts/run_reconstruction.py
```

Cada corrida de segmentación tiene un ID, impreso al finalizar y usado como
nombre de carpeta en `results/segmentation/`. Para los comandos siguientes,
reemplazar `ID_CORRIDA` por ese valor:

```bash
python scripts/evaluate_lesions.py --run-id ID_CORRIDA --configuration-id cfg_000
python scripts/analyze_agreement.py --lesion-matches results/segmentation/ID_CORRIDA/lesion_eval/lesion_matches.csv
```

La evaluación produce métricas por lesión y paciente; el análisis de acuerdo
incluye correlaciones y Bland–Altman. Para comparar dos configuraciones dentro
de una misma corrida, usar `compare_methods.py --run-id ID_CORRIDA --compare
cfg_000,cfg_001`, siempre que ambos IDs existan en `configurations.csv`.
`run_segmentation.py --run-id ID_NUEVO --sweep` genera un barrido básico de
configuraciones; debe existir la selección. Las comparaciones usan Wilcoxon
pareado y corrección de Holm. No implican validación independiente.

La reconstrucción simula proyecciones de cortes ya reconstruidos: no utiliza
sinogramas adquiridos por un tomógrafo. La compresión compara el control de
8 bits con PNG, Huffman, RLE y DCT tipo JPEG. No incluye un decodificador de
archivos JPEG clínicos ni una evaluación de adquisición real.

La [guía técnica](documentation.md) explica cada script, las salidas y las
extensiones 3D. No se incluyen resultados precalculados, datos, informe ni
presentación en esta entrega de código.

## Verificación

```bash
python scripts/check_requirements.py
python -m pip check
python -m unittest discover -s tests -v
```

Los tests usan datos sintéticos temporales y no requieren descargar el dataset.
La preparación se verifica en Linux; el lanzador Windows se revisa estáticamente,
sin afirmar una prueba nativa en Windows. Los barridos de investigación y las
auditorías históricas no forman parte del árbol final; el historial Git se conserva.

Verificación de esta preparación (5 de noviembre de 2026): 134 tests aprobados;
selección sintética de 131 pacientes y 655 cortes; corrida real reducida de tres
cortes del caso 94; inicio de GUI y carga de seis presets; ayuda de 16 comandos,
dependencias y sintaxis del lanzador Linux verificadas. Se utilizó un entorno
Python 3.12 existente, sin copiarlo a la entrega. La instalación desde cero y la
ejecución nativa en Windows no se verificaron. Los resultados de estas pruebas
fueron temporales y no se incluyen.
