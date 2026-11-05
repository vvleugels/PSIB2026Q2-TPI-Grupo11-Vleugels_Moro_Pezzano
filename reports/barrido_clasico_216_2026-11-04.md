# Barrido clásico de 216 configuraciones y uniones automáticas

## Resultado

Se implementó y ejecutó el plan completo sobre los **diez pacientes y 50 cortes
congelados**: 83 referencias lesionales 2D, 32 menores de 10 mm y 21 cortes negativos.
Se mantuvieron las cinco particiones externas y cuatro internas por partición.
La selección maximiza F2 por lesión a IoU ≥0,10, sin límite obligatorio de FP.
Las uniones se evaluaron desde sus máscaras OR, con un máximo de tres miembros.

**No se encontró una mejora uniforme que justifique cambiar los defaults.**
Seleccionar por F2 reduce componentes falsos positivos, pero sacrifica sensibilidad,
en particular lesiones pequeñas. La unión voraz no mejora los resultados externos
respecto de una configuración individual. El caso 94 empeora considerablemente.

| Procedimiento, pacientes retenidos | F2 | Sensibilidad | Pequeñas <10 mm | FP/corte | Dice medio por paciente |
|---|---:|---:|---:|---:|---:|
| Default actual | 0,243 | 74,7 % (62/83) | 46,9 % (15/32) | 17,68 | 0,383 |
| Selección individual por F2 | **0,365** | 63,9 % (53/83) | 34,4 % (11/32) | **6,84** | 0,408 |
| Unión seleccionada por F2 | 0,360 | 63,9 % (53/83) | 34,4 % (11/32) | 7,02 | 0,408 |
| Selección de máxima sensibilidad | 0,015 | **88,0 % (73/83)** | **75,0 % (24/32)** | 492,72 | 0,194 |

La selección individual reduce el número de FP por corte un 61,3 %, pero pierde
9 detecciones netas y 4 detecciones pequeñas netas frente al default. Además,
menos componentes falsos no equivale necesariamente a menos área falsa: sus
píxeles fuera de referencia aumentan de 49.879 a 70.449. Los componentes pueden
ser más grandes; por eso se conservaron métricas de área y precisión por píxel.

Con IoU ≥0,25, la sensibilidad es 66,3 % para el default, 53,0 % para individual
y unión, y 72,3 % para máxima sensibilidad. No se seleccionó retrospectivamente
la variante que mejor rinde en cada paciente retenido.

## Caso 94: selección realizada sin usar este paciente

| Procedimiento | Referencias detectadas | FP/corte | Cobertura de referencia | Dice |
|---|---:|---:|---:|---:|
| Default | 10/13 | 17,4 | 22,1 % | 0,238 |
| Individual (`otsu_113`) | 3/13 | 5,2 | 5,0 % | 0,080 |
| Unión (`otsu_023` + `otsu_113`) | 3/13 | 5,4 | 5,0 % | 0,081 |
| Máxima sensibilidad (`otsu_030`) | 13/13 | 705,0 | 51,6 % | 0,096 |

En el corte 463, tanto individual como unión devuelven máscara vacía y pierden
la referencia pequeña. El default cubre el 68 % de sus píxeles. En el corte
negativo 608, individual y unión generan 8 componentes falsos, el default 29 y
máxima sensibilidad 722. En pequeñas, individual y unión recuperan 1/4 frente
al 3/4 del default. Las figuras muestran estas mismas máscaras, sin retoques.

## Qué se probó

- Ventanas centro/ancho: 30/150, 45/100 y 100/200.
- Suavizado: ninguno o gaussiano sigma 0,3 / 0,6 / 1,0 píxeles.
- Ecualización: ninguna, histograma global o CLAHE (ventana 8, límite 0,01).
- Otsu dentro del hígado, seguido de los seis perfiles de limpieza acordados:
  sin limpieza; solo área 10; cierre radio 1 + relleno + área 10;
  apertura/cierre radio 1 + relleno sin filtro de área; ese mismo perfil con
  área 10; apertura/cierre radio 2 + relleno + área 10.
- Se desactivaron detector HU, bordes, watershed y realce. Se reutilizaron
  imágenes preprocesadas y máscaras intermedias. Los tamaños se mantuvieron
  en píxeles y se guardó su equivalencia física por paciente.

Las configuraciones elegidas en los cinco folds fueron `otsu_041`, `otsu_131`,
`otsu_131`, `otsu_113`, `otsu_131`. Todas usan apertura/cierre radio 2, relleno y
área 10, sin ecualización. Cambian ventana y suavizado. Solo un fold agregó un
segundo miembro (`otsu_023` + `otsu_113`); ningún fold llegó a tres porque no
mejoraba estrictamente el F2 interno. No se exploraron exhaustivamente todos
los tríos: se aplicó el procedimiento voraz especificado.

## Presets exploratorios para futuras pruebas

Al seleccionar con todos los pacientes, el mejor individual fue **`otsu_041`**:
ventana **30/150**, gaussiano **0,6 píxeles**, **sin ecualización**, Otsu,
apertura/cierre con disco radio **2**, relleno y área mínima **10 píxeles**.
La búsqueda de unión se detuvo en ese único miembro: ningún agregado mejoró F2.
Su resultado descriptivo en toda la cohorte es F2 0,414, sensibilidad 68,7 %,
pequeñas 34,4 % y 6 FP/corte. **No confundir estos números con validación
independiente del preset** ni con la tabla externa del procedimiento anterior.

El preset de máxima sensibilidad `otsu_030` usa ventana 30/150, gaussiano 0,3,
CLAHE y ninguna limpieza. Su abundancia de fragmentos produce demasiados FP.
Los YAML se entregan como propuestas para pruebas futuras, no como promoción
automática. No se modificó `config.yaml` durante esta búsqueda.

## Implementación, verificación y reproducción

- Nuevo ejecutor `scripts/run_classical_search.py` y módulo
  `src/psib/classical_search.py` con grilla, puntuación y selección.
- **115 comparaciones de máscaras contra nueve corridas históricas: todas
  idénticas píxel a píxel.**
- Suite completa final: **115 pruebas correctas**, incluidas las siete pruebas
  específicas del buscador (máscaras empaquetadas, unión y reanudación de caché).
- Pruebas de correspondencias comparan el evaluador rápido con el evaluador
  público, incluidos casos vacíos y componentes múltiples. Se comprueba que
  las uniones se recalculan, que la selección consulta solo pacientes permitidos
  y que el filtro de área puede eliminar una referencia pequeña.
- Las figuras se revisaron visualmente. Se guardan máscaras externas completas
  del caso 94, parámetros efectivos, unidades físicas, selección por fold,
  trazas de unión y conteos por corte/paciente.
- Se verificaron las **400 filas de evaluación externa** contra las máscaras
  guardadas: áreas, intersecciones, confinamiento hepático y conteos de componentes.

```bash
.venv/bin/python scripts/run_classical_search.py --run-id nueva-busqueda
# Solo para continuar la misma ejecución interrumpida:
.venv/bin/python scripts/run_classical_search.py --run-id nueva-busqueda --resume
```

Resultados: `results/classical_search/classical-search-20261104/`.
Archivos principales: `configurations.csv`, `validation_summary.csv`,
`validation_per_case.csv`, `outer_per_slice.csv`, `selected_configurations.csv`,
`union_selection_trace.json`, `sensitivity_fp.png`, `case94/` y `presets/`.
Las máscaras individuales se guardan como bits empaquetados con sus metadatos
espaciales, para evitar cientos de archivos grandes por paciente.

Este es un estudio interno pequeño sobre cortes seleccionados usando referencia,
con pacientes ya explorados. El resultado muestra que maximizar F2 no garantiza
mejor sensibilidad, buena cobertura en todos los pacientes ni conservación de
lesiones pequeñas. La búsqueda quedó completada; su resultado es una comparación
reproducible, no una afirmación de mejora generalizable.
