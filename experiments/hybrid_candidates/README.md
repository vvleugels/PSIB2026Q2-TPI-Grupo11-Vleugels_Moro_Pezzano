# Combinación adaptativa por tamaño

Protocolo clásico sin entrenamiento. La máscara principal conserva todos sus
componentes. Una rama HU watershed propone componentes adicionales; se admiten
según su Feret físico **predicho**, nunca según la referencia. La normalización
HU usa la mediana y MAD hepáticas de cada corte. No hay reglas por ID de paciente.

## Protocolo congelado

Se reutilizan selección, controles y folds de `experiments/local_candidates/`:
50 cortes, diez pacientes, cinco folds externos y cuatro particiones internas.

- Etapa 1: `otsu_059` + `hu_sensitive`, límites estrictos `<10`, `<20`, `<30` mm,
  con incorporación completa o solo componentes sin contacto con la máscara base.
- Etapa 2: base `default`/`otsu_059`; watershed sigma 0,6/1,0 mm, alto 1/1,5/2,
  bajo 0,33/0,50, fondo 10 mm, piso de ruido 5 HU y área mínima cero. Los mismos
  tres límites y dos reglas producen 144 variantes.
- El contacto incluye vecinos diagonales. No se aplica apertura ni limpieza
  adicional a las pequeñas. Las salidas quedan dentro de la ROI hepática.
- Controles: `default`, `otsu_059`, `hu_sensitive`, bilateral y unión completa
  de `otsu_059` con `hu_sensitive`. Cada máscara final se evalúa nuevamente,
  incluyendo fusiones que pueden reducir la detección uno-a-uno.

Cada etapa y el procedimiento combinado seleccionan entre sus variantes y los
dos controles Otsu. En los ocho pacientes internos: Dice medio ≥0,40 y máxima
sensibilidad total, desempate por sensibilidad <10 mm, Dice, menor área falsa
física, menor FP/corte e ID. Si ninguna opción es factible, se usa `otsu_059`
y se registra `no_variant_reaches_dice_floor`. No se consulta el resultado
externo para elegir parámetros. Los métodos son fijos, sin entrenamiento:
agrupar las cuatro particiones internas evalúa cada paciente una sola vez.

No hay límite obligatorio de FP. El éxito externo requiere Dice ≥0,40, mayor
sensibilidad total que ambos controles Otsu y sensibilidad <10 mm y ≥30 mm no
inferior a la mejor de ambos. El piso interno no garantiza el piso externo.
La selección conjunta incluye ambas etapas sin elegir la etapa ganadora a partir
de sus resultados externos. No se amplía la grilla después de observarlos.

## Ejecución

```bash
.venv/bin/python scripts/run_hybrid_search.py --preflight
.venv/bin/python scripts/run_hybrid_search.py --run-id hybrid-v1
.venv/bin/python scripts/run_hybrid_search.py --run-id hybrid-v1 --resume
.venv/bin/python scripts/run_hybrid_search.py --run-id hybrid-v1 --report-only
```

Las entradas predeterminadas son `results/local_candidates/local-v2-liver/` y
`results/otsu-bilateral-50/segmentation/otsu-bilateral-50/`. Pueden indicarse con
`--source` y `--bilateral`. Deben ser corridas completas: se verifican selección,
folds, configuración, geometría y sellos; los cuatro controles reutilizados se
reproducen exactamente con el código actual antes de combinar cada corte.
El ejecutor usa los HU y las máscaras de entrada conservados por la corrida
local, sin volver a cargar volúmenes NIfTI. Conserva la huella del manifiesto
original y de todos los artefactos reutilizados.

`--preflight` no crea resultados ni segmenta. `--resume` exige las mismas
entradas, código del experimento y dependencias; verifica sellos de checkpoints.
`--report-only` no necesita las corridas fuente: recalcula las métricas de todas
las máscaras propias, verifica contra los CSV guardados y vuelve a seleccionar.
Conserva el análisis anterior en `analysis.previous` y registra la versión del
código de análisis. Un bloqueo exclusivo evita dos escritores simultáneos.

## Salidas

`results/hybrid_candidates/<run-id>/` contiene protocolo, grilla, controles,
selección, folds, versiones, huellas y estado de ejecución. Cada corte conserva
HU, ROI, referencia, spacing, máscaras empaquetadas por configuración, máscaras
HU por parámetros, decisiones por candidato, parámetros efectivos y fallbacks.

`analysis/` incluye métricas recalculadas, selección por fold, resultados internos
y externos, tamaños, fusiones/divisiones, ganancias/pérdidas frente a ambos Otsu,
negativos ordenados por área falsa, frontera sensibilidad–Dice y figuras de los
50 cortes. Los tamaños de referencia son <10, [10,30), [30,50] y >50 mm; también
se agrega ≥30 mm. Las correspondencias se evalúan a IoU ≥0,10 y ≥0,25.
Las áreas falsas suman mm² en los cortes seleccionados: no son volúmenes.

`analysis/preset.yaml` contiene una configuración completa elegida sobre toda
la cohorte, **sin evaluación independiente**. Conserva rutas absolutas locales;
ajustarlas al trasladarlo. Para usarla con la selección existente en `output_dir`:

```bash
.venv/bin/python scripts/run_segmentation.py \
  --config results/hybrid_candidates/hybrid-v1/analysis/preset.yaml \
  --run-id hybrid-preset
```

Este comando usa la selección existente; no fuerza los 50 cortes congelados.
Para reproducir exactamente esos cortes, copiar la selección de la corrida
híbrida a `selected_cases_slices.csv` en un directorio de salida separado y
apuntar allí el YAML. `run_tpi.py` regenera la selección según su configuración.

El despachador compartido admite `detection.method: hybrid_size`, con
`base_config` (preprocesamiento y segmentación clásicos), `hu_settings`,
`max_diameter_mm` y `addition_rule` (`complete`/`disjoint`). Devuelve máscara
final, base, candidatos HU, píxeles añadidos y decisiones por componente. No
acepta referencia tumoral. Se integra el preset fijo en el ejecutor 2D; no se
agregan controles específicos de GUI ni una validación de las extensiones 3D,
compresión o reconstrucción. Los defaults del proyecto no cambian.

## Verificación

```bash
PYTHONPATH=src:. .venv/bin/python -m unittest tests.test_hybrid_segmentation -v
PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests -q
```

Las pruebas cubren spacing anisotrópico, límites estrictos, contactos diagonales,
fusiones, vacío, confinamiento, identidad entre API y ejecutor 2D, selección sin
consulta a pacientes externos, respaldo, checkpoints y repetición del análisis.
Este experimento usa una cohorte ya explorada y ROI hepática de referencia;
no es validación externa ni evaluación sobre volúmenes completos.
