# Implementación y evaluación del detector clásico

Actualización: se evaluaron límites más amplios y selección sin límite en
[la reevaluación de FP](reevaluacion_fp_2026-11-04.md). El detector alcanza mayor
sensibilidad con un costo elevado en falsos positivos; las conclusiones siguientes
corresponden al protocolo original de 8 FP/corte.

## Resultado principal

Se implementó el plan y se ejecutó el barrido completo: **48 variantes + baseline
histórico + Otsu coherente, sobre 50 cortes de 10 pacientes**. Hay 83 componentes
lesionales de referencia en 2D, de los cuales 32 miden menos de 10 mm de Feret;
21 cortes no contienen tumor. Una lesión presente en distintos cortes cuenta
como varias observaciones 2D, no como varias lesiones volumétricas independientes.

**El detector nuevo no cumple el criterio de promoción.** Ninguna de las 48
variantes respeta el presupuesto interno de 8 FP/corte en ninguno de los cinco
folds. Tampoco hay variantes factibles para los presupuestos 1/2/4. El selector
registra abstención explícita: su recall externo cero no significa que cada
variante individual produzca máscaras vacías.

| Método fijo, mismos 50 cortes | Recall lesión IoU ≥0,10 | Recall <10 mm | FP/corte | Dice medio por paciente |
|---|---:|---:|---:|---:|
| Histórico | 30,12 % (25/83) | 9,38 % (3/32) | 7,78 | 0,231 |
| Otsu coherente | 77,11 % (64/83) | 46,88 % (15/32) | 9,68 | 0,414 |
| Detector con menor FP global, `det_042` | 67,47 % (56/83) | 40,63 % (13/32) | 22,62 | 0,208 |
| Detector de máximo recall global, `det_009` | 87,95 % (73/83) | 75,00 % (24/32) | 156,54 | 0,227 |

Las dos filas del detector son **diagnósticos exploratorios seleccionados viendo
todos los resultados**, no ganadores validados. `det_013` empata el recall máximo
con más FP. No reemplazan las decisiones por fold ni justifican promover defaults.
Con IoU ≥0,25 el histórico recupera 13,25 % y Otsu 69,88 % de las lesiones.

La observación útil para su prioridad —menos falsos negativos— es que el Otsu
coherente supera al histórico en sensibilidad agregada, aunque también supera el
presupuesto de 8 FP/corte y tiene limitaciones por paciente. El nuevo contraste
adaptativo dispara demasiados candidatos: no corresponde afirmar que mejora el
sistema por tener mayor recall bruto. No se cambió el protocolo para favorecerlo.

## Qué quedó implementado

- `src/psib/detection.py`: API `detect_slice`, contraste global/local en HU,
  suavizado normalizado por máscara en mm, MAD con piso 5 HU, semillas y crecimiento
  de ocho vecinos, watershed por región hepática con fallbacks registrados,
  limpieza opcional por área en mm² y diagnósticos de componentes.
- Detector compartido entre GUI, ejecución 2D y extensión 3D. La máscara elegida
  se muestra, guarda y cuantifica consistentemente; los controles de visualización
  no alteran la predicción experimental. El 3D no agrega un filtro de componentes.
- `scripts/run_detection.py`: barrido reproducible, congelamiento de selección y
  folds, dos baselines, máscaras de cada variante, tablas, curva de puntos de
  operación e informe. La entrada habitual `run_segmentation.py` despacha al
  detector si se habilita en configuración.
- `src/psib/detection_validation.py`: cinco folds externos de dos pacientes y
  cuatro particiones internas para los otros ocho; elección por recall bajo
  presupuestos de FP, sin consultar resultados externos. Al no haber modelo
  entrenado, la validación interna evalúa una vez cada configuración fija en cada
  paciente interno. Se documenta abstención cuando ningún candidato es factible.
- Emparejamiento uno-a-uno que maximiza cardinalidad antes que IoU; criterios
  0,10/0,25. La evaluación por grupos de fusiones/divisiones sigue disponible para
  geometría, separada de los conteos nuevos.
- Correcciones de F1=0 cuando corresponde y de la media de pares de Bland–Altman.
- Adaptación del detector en reconstrucción y compresión. En compresión se trabaja
  con HU recuperados de la ventana cuantizada, no se confunde ese control con HU
  originales. Los baselines históricos siguen disponibles.

## Validación de implementación

- Suite completa: **106 tests, todos pasan**.
- Después de los ajustes finales de integración: **19 tests específicos, todos pasan**.
- Casos cubiertos: imagen constante, ROI vacía, ausencia de semillas, lesiones
  pequeñas/múltiples/periféricas, spacing anisótropo, confinamiento al hígado,
  falta de marcadores, caché coherente, independencia de ventana/CLAHE, igualdad
  de vista previa/2D/3D, compresión sin pérdida y elección sin filtración externa.
- Un caso adversario verifica que dos matches válidos prevalezcan sobre un único
  match de mayor IoU. Los tests de F1 y Bland–Altman reproducen los errores previos.
- Las máscaras históricas se reprodujeron exactamente en los **15 cortes** de la
  corrida inicial. Las áreas de referencia coinciden con la selección congelada
  en los **50 cortes**.
- Compilación Python y `git diff --check` sin errores.

La GUI se comprobó mediante su lógica real de vista previa con controles simulados;
no se realizó una revisión visual interactiva completa de la ventana. La extensión
3D se verificó con volúmenes sintéticos; el benchmark real es 2D, no una evaluación
volumétrica de los diez pacientes.

## Uso

En la GUI, activar **Detector HU experimental (prioriza sensibilidad)** y elegir
`hysteresis` o `watershed`. En YAML, establecer `detection.enabled: true` en una
copia completa de la configuración. Los defaults del proyecto siguen desactivados
porque no se cumple el criterio de promoción.

```bash
.venv/bin/python scripts/run_detection.py --run-id nueva-validacion \
  --benchmark --historical-run results/segmentation/20261103-135329

PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -q
```

Los resultados de esta ejecución están en
`results/segmentation/classical-20261104/`. Archivos clave:

- `validation_report.md`, `validation_summary.csv`, `validation_per_case.csv`:
  resultados externos y explicación de la abstención.
- `selected_configurations.csv`, `folds.json`, `inner_validation.csv`:
  selección y separación por paciente.
- `exploratory_detection_summary.csv`: comportamiento bruto de cada variante fija;
  no confundir con validación del selector.
- `components.csv`, `effective_parameters.csv`, `predictions/`: diagnósticos y
  máscaras; el ID de componente diagnóstico corresponde a la máscara anterior
  al descarte por área, no necesariamente al ID del emparejamiento posterior.
- `promotion.json`: criterio no satisfecho; defaults sin cambios automáticos.

El estudio es interno, pequeño, con cortes elegidos por referencia y ROI hepática
conocida; tres pacientes ya habían sido explorados antes. No es validación externa.
Los datos completos de `results/` están ignorados por Git; este informe conserva
las cifras principales junto al código.
