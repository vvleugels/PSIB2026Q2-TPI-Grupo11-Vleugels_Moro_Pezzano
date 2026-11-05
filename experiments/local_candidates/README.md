# Candidatos automáticos y segmentación local — ejecución diferida

Implementación experimental sin ML. **No se ejecutaron el barrido, las pruebas
numéricas ni una validación visual de la GUI durante esta entrega.** Los presets
son puntos de partida, no configuraciones ganadoras. La grilla tiene 108 variantes
más tres controles congelados. Nada modifica `config.yaml` automáticamente.

## Traslado y entorno Linux

Copiar el repositorio completo y los diez pares NIfTI de `data/Selected_database/`.
`results/` y `data/` están ignorados por Git: los resultados históricos NO son
necesarios para este ejecutor; selección, folds y controles se incluyen aquí.
No copiar `.venv`: crear el entorno en la computadora destino.

El archivo `requirements-linux.lock.txt` captura versiones exactas del entorno
actual (Python **3.14.8**, Linux). Es un snapshot de dependencias, **no una
instalación probada de este nuevo código**. Preparar Python 3.14 en la máquina
destino. Tk requiere el paquete del sistema correspondiente (`python3-tk` o
su equivalente para Python 3.14); no es necesario para el barrido sin interfaz.
Las pruebas de importación de GUI sí requieren Tk. No se necesita GPU.

Desde la raíz del repositorio:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r experiments/local_candidates/requirements-linux.lock.txt
.venv/bin/python -m pip check
```

Si no existen paquetes binarios compatibles en la máquina destino, resolver el
entorno antes de iniciar el barrido y guardar sus versiones; nunca cambiar
versiones a mitad de una corrida que se pretende reanudar.

## Orden de ejecución en la otra computadora

1. Comprobar entradas (lee cabeceras; no segmenta ni crea resultados):

```bash
.venv/bin/python scripts/run_local_search.py --preflight
# Para datos en otra ubicación:
.venv/bin/python scripts/run_local_search.py --preflight --data-dir /ruta/Selected_database
```

2. Ejecutar las pruebas preparadas y las regresiones históricas:

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest tests.test_local_segmentation tests.test_local_experiment -v
PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests -v
# Integración adicional: trabajadores reales, serial/paralelo y reanudación.
PSIB_RUN_LOCAL_INTEGRATION=1 PYTHONPATH=src:. .venv/bin/python -m unittest tests.test_local_experiment.LocalExperimentTests.test_serial_parallel_and_resume_identical_masks -v
```

3. Una vez resueltos posibles fallos, lanzar el barrido:

```bash
.venv/bin/python scripts/run_local_search.py --run-id local-v1 --workers 1
# Ejemplo para una máquina con RAM suficiente para varios volúmenes:
.venv/bin/python scripts/run_local_search.py --run-id local-v1-parallel --workers 4
```

Cada trabajador carga un paciente completo y sus máscaras. El costo de memoria
crece con los trabajadores; comenzar conservadoramente y revisar
`performance.json` antes de aumentar concurrencia. Registrar también la memoria
total del sistema cuando se ejecuten varios trabajadores. Variables BLAS/OMP se
limitan por defecto a un hilo, salvo valores ya definidos por el usuario.

4. Continuar la misma corrida interrumpida, con exactamente el mismo código,
datos, parámetros y versiones:

```bash
.venv/bin/python scripts/run_local_search.py --run-id local-v1 --resume --workers 2
```

Cambiar el número de trabajadores no invalida la caché. Cambiar datos, código,
grilla, controles, folds o dependencias sí la invalida. Los hashes usan contenido,
no rutas absolutas. Los volúmenes se leen completos para calcular sus hashes al
iniciar/reanudar. Un bloqueo exclusivo impide dos escritores sobre una corrida.
Si un artefacto completado está corrupto, la reanudación falla explícitamente.
Los directorios `.partial` son trabajo incompleto y se recalculan.

5. Regenerar solo análisis y figuras desde máscaras existentes:

```bash
.venv/bin/python scripts/run_local_search.py --run-id local-v1 --report-only
```

Este comando no ejecuta detectores y no necesita los volúmenes originales.
Conserva el análisis anterior como `analysis.previous`. El análisis original
queda vinculado al manifiesto de ejecución; una regeneración usa el código de
análisis actualmente instalado y registra su versión por contenido.

`--config`, `--selection` y `--folds` aceptan rutas alternativas. Se sigue
exigiendo el protocolo de diez pacientes, cinco cortes por paciente, cinco folds
externos y particiones internas válidas. Las rutas de `config.yaml` se resuelven
contra ese archivo; `--data-dir` relativo se interpreta respecto de la terminal.

## Salidas y selección

En `results/local_candidates/<run-id>/`:

- `manifest.json`, `grid.json`, `controls.json`, `selection.csv`, `folds.json`,
  `run.json`, `performance.json`: entradas, versiones, huellas, estado y recursos.
- Un directorio por paciente/corte con `inputs.npz`, `masks.npz`,
  `candidate_masks.npz`, `candidates.json`, `metrics.csv`, `lesions.csv` y sello
  SHA-256. Las matrices binarias se guardan con `np.packbits` en orden C;
  `shape` y `ids` permiten recuperarlas con `np.unpackbits(..., count=H*W)`.
  Las máscaras individuales de candidatos se recuperan con la caja de
  `candidates.json`, en el orden de sus componentes (índice desde cero en NPZ).
- `analysis/summary.csv`, `per_case.csv`, `choices.csv`, `outer_per_slice.csv`,
  `outer_per_lesion.csv`, `exploratory.csv`, `ablations.csv`, `report.md`, figuras,
  `promotion.json` y `preset.yaml`.

Selección interna: maximizar Dice medio por paciente sin bajar recall total ni
recall <10 mm frente al default. Desempatar por menor área falsa, menos FP/corte
 e identificador. El default también compite: si no hay mejora admisible,
permanece como respaldo. Variantes con candidatos fallidos en entrenamiento no
son seleccionables. También se informa selección por Dice sin restricciones.
Todos los controles se evalúan en los pacientes retenidos.

El criterio externo de promoción compara el procedimiento de selección con el
default a IoU ≥0,10. IoU ≥0,25 es análisis de sensibilidad. No implica validación
independiente del preset final elegido con toda la cohorte. Son pacientes ya
explorados; cualquier nuevo resultado es exploratorio. La evaluación de área
falsa física suma mm² de cortes seleccionados, **no estima volumen tumoral**.

Los fallos por candidato se registran sin cambiar a otro método. Si ocurre
alguno en la grilla, el estado es `completed_with_candidate_failures`; inspeccionar
sus razones antes de interpretar los resultados. Excepciones de infraestructura
interrumpen la corrida con estado `failed`. No hay truncamiento por máximo
arbitrario de candidatos. Contacto con el borde de ROI y falta de convergencia
se exportan como diagnósticos, sin confundirlos con exactitud clínica.

## Métodos, límites y bibliografía

Las semillas se extraen de contraste multiescala en HU, sin etiquetas tumorales.
El suavizado es 0,6 mm; fondos de 3/6/12 mm; normalización MAD con piso 5 HU;
máximos separados al menos 2 mm; soporte a puntuación >0,5 y margen de 5 mm.
Se comparan umbrales 1,5/2/2,5, polaridad oscura/ambas y contexto ausente/presente.
No hay filtro de área ni apertura morfológica.

El contexto usa hasta dos vecinos por lado dentro de 5 mm. En cada píxel se
normaliza la media vecina por soporte hepático y pesos `exp(-distancia_mm/2.5)`;
se combina 75 % central y 25 % vecinos. Sin soporte vecino se usa el central.
La escala de origen almacenada identifica la respuesta central; la puntuación
puede incluir contexto. No se exige persistencia para aceptar una lesión.

- Crecimiento: mediana/MAD actualizados después de cada expansión, contraste
  local mínimo y tolerancia 1,5/2/2,5. Puede seguir limitado por el soporte de
  contraste; su evaluación indicará si mejora sobre el detector anterior.
- Contorno: level set positivo interior, medias interiores/exteriores distintas
  por píxel en ventana física de radio 5 mm, ajuste regional y curvatura física.
  Pesos 0,1/0,3/0,6; hasta 150 pasos, parada por cambio <0,1 % durante cinco
  pasos. Se conserva la semilla y se registra energía de ajuste local, no una
  promesa de descenso monótono de la energía completa. No es Chan–Vese global.
- Graph cuts: costos regionales robustos truncados a 25, vecindad de ocho con
  pesos por distancia y diferencia de HU, semillas duras y fondo fuera del
  soporte. SciPy resuelve flujo máximo con Dinic. Las capacidades finitas se
  cuantizan con escala común ≤1024, mínimo 1 para costos positivos y suma
  <2^28. Cada restricción dura vale suma+1, mayor que cualquier corte factible
  de costos finitos, manteniendo cada capacidad por debajo de 2^31.
  Se registra escala y flujo; la cuantización puede producir empates y debe
  verificarse con las pruebas de enumeración antes de usar resultados.

Las tres ramas mantienen solo componentes conectados a semillas y confinados
al hígado. Las máscaras por candidato se conservan aunque se unan para medir
la salida binaria. Los límites entre ramas y los fallos de propuestas iniciales
son parte del experimento: no se garantiza un resultado mejor.

Referencias de inspiración, **no reproducciones literales**:

- Hoogi et al. (2017), *Adaptive local window for level set segmentation of CT
  and MRI liver lesions*. [Texto](https://web.stanford.edu/group/rubinlab/pubs/Hoogi-2017.pdf).
- Yang et al. (2021), *Semi-automatic liver tumor segmentation with adaptive
  region growing and graph cuts*. [Artículo](https://doi.org/10.1016/j.bspc.2021.102670).

Nuestro sistema reemplaza inicializaciones manuales por candidatos automáticos;
la parte de localización no tiene garantizados los resultados de esos trabajos.
Las reglas y energías no entrenan un modelo ni aplican clustering.

## GUI y preset fijo

Activar detector experimental, elegir `local_growth`, `local_contour` o
`local_graphcut` y cargar uno de los `preset_*.yaml` de esta carpeta. Los presets
solo contienen la sección del detector; la GUI conserva la configuración del
resto del proyecto. Tras una ejecución puede cargarse `analysis/preset.yaml`
si seleccionó un método local; si seleccionó default, conservar el baseline.

En modo Overlay, la opción adicional muestra semillas verdes y el borde de
candidatos violeta. Las métricas se calculan sobre la máscara final, no sobre
esas capas. La vista previa es asíncrona y descarta respuestas de cortes viejos.
El botón habitual de ejecución usa el mismo detector y proveedor de contexto,
pero ejecuta **un preset** sobre la selección habitual, no las 108 variantes.

Para ejecutar un preset fijo desde CLI, combinar su sección `detection` con una
configuración completa y usar:

```bash
.venv/bin/python scripts/run_segmentation.py --config /ruta/config-completa.yaml --run-id local-preset-01
```

El barrido histórico `--sweep` no acepta métodos locales: usar el ejecutor nuevo.
Compresión, reconstrucción y la ruta volumétrica anterior no suministran contexto;
con `context: true` fallan explícitamente. No hacen fallback a Otsu. Para estudiar
esas extensiones hará falta adaptar sus entradas y definir su comparación.
