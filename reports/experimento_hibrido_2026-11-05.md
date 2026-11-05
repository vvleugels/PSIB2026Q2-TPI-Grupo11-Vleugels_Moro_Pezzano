# Método combinado por tamaño — 5 de noviembre de 2026

Se implementaron y ejecutaron las dos etapas previstas sobre los **50 cortes
congelados de diez pacientes**, con 83 referencias lesionales 2D, 32 menores de
10 mm y 21 cortes negativos. **No se alcanzó el objetivo conjunto de aumentar
sensibilidad y mantener Dice medio por paciente ≥0,40.** Los defaults permanecen
sin cambios. No se amplió la grilla después de observar los resultados externos.

## Resultado externo del procedimiento

La selección por paciente admite variantes con Dice interno ≥0,40 y maximiza
sensibilidad; desempata por sensibilidad pequeña, Dice, menor área falsa, menos
FP/corte e ID. Los controles `default` y `otsu_059` también compiten. Si ninguna
opción cumple el piso, conserva `otsu_059` y registra el incumplimiento.

Las tres selecciones —etapa 1, etapa 2 y conjunto de ambas— terminaron eligiendo
las mismas máscaras: `otsu_059` en folds 0, 1 y 4, y `default` en folds 2 y 3.
**Ninguna configuración híbrida fue elegida.** En folds 0 y 4 no hubo variante
factible; el Dice interno del respaldo fue 0,399847 y 0,359504, respectivamente.
El valor del fold 0 está cerca de 0,40, pero no se redondeó para admitirlo.

| Método/procedimiento | Dice medio | TP/83 | FN | TP <10 mm /32 | TP ≥30 mm /20 | FP/corte | Área falsa mm² |
|---|---:|---:|---:|---:|---:|---:|---:|
| `default` fijo | 0,3565 | 63 | 20 | 16 | 18 | 20,88 | 37.119,5 |
| `otsu_059` fijo | 0,4417 | 62 | 21 | 13 | 19 | 10,82 | 77.091,8 |
| Selector combinado externo | 0,4206 | 61 | 22 | 14 | 18 | 15,10 | 34.526,8 |
| `hu_sensitive` fijo | 0,2275 | 73 | 10 | 24 | 19 | 156,54 | 97.928,4 |
| Bilateral fijo | 0,3462 | 42 | 41 | 4 | 19 | 2,28 | 91.025,4 |

Los conteos principales usan correspondencia uno-a-uno a IoU ≥0,10. A IoU ≥0,25,
el selector combinado detecta 51/83 y omite 32, frente a 57/83 y 26 omisiones de
`otsu_059`. No se presenta la selección entre controles como una mejora del
nuevo segmentador híbrido.

El selector obtiene 73,5 % de sensibilidad total y 43,8 % en menores de 10 mm.
Frente a `default`, gana una referencia y pierde tres; frente a `otsu_059`, gana
una y pierde dos. Las pérdidas frente a Otsu ocurren en el caso 94/corte 577:
referencias de 28,89 y 31,37 mm; la ganancia está en el caso 94/corte 463, de
5,30 mm. El promedio Dice supera 0,40, pero los casos 3, 45, 57, 94 y 105 quedan
por debajo; el caso 105 no tiene referencia tumoral en estos cortes y presenta FP.

## Lo que sí recupera la combinación, y su costo

Los siguientes son resultados **descriptivos en toda la cohorte**, seleccionados
después de puntuar la grilla; no son rendimiento externo de un preset ganador.

| Variante | Dice medio | TP/83 | FN | TP <10 mm /32 | FP/corte |
|---|---:|---:|---:|---:|---:|
| Unión completa Otsu/HU | 0,2332 | 74 | 9 | 24 | 148,22 |
| Híbrido de mayor sensibilidad, `h1_004` / `h2_094` | 0,2400 | 76 | 7 | 26 | 149,60 |
| Híbrido de mayor Dice, `h2_133` | 0,3271 | 67 | 16 | 18 | 39,28 |

`h1_004` conserva Otsu y agrega componentes completos de `hu_sensitive` con
Feret predicho <30 mm. Detecta el 91,6 % del total y 81,3 % de las referencias
<10 mm, pero no cumple el piso de Dice. Su duplicado en etapa 2 produce las
mismas máscaras, como corresponde a sus parámetros idénticos.

`h2_133` conserva Otsu y agrega solo candidatos <10 mm sin contacto con la base,
con sigma HU 1 mm, umbral alto 2 y bajo 0,33. Reduce el costo frente a las
combinaciones más sensibles, pero su Dice 0,327 tampoco cumple el objetivo.
**Entre las 150 combinaciones, ninguna llegó a 0,40 en toda la cohorte.** El
único control/variante con ese piso fue `otsu_059`, que queda como preset
exploratorio elegido con todos los pacientes.

El tamaño predicho no distingue suficientemente pequeñas lesiones de otras
regiones. Además, conservar una base no garantiza conservar su Dice ni su
sensibilidad por lesión: el área añadida puede ser falsa o fusionar componentes.
Reducir FN por sí solo no basta para obtener una máscara bien delimitada.

## Implementación y trazabilidad

- Módulo `psib.hybrid_segmentation`, despachador `detection.method: hybrid_size`
  y ejecución de preset fijo por el flujo 2D habitual. Recibe HU, ROI hepática,
  spacing y ajustes; nunca referencia tumoral.
- Seis combinaciones iniciales, 144 ajustadas y cinco controles. El ajuste HU
  usa mediana/MAD hepática por corte; los tamaños y áreas de candidatos se
  calculan en unidades físicas. No hay reglas por ID de paciente ni contexto 3D.
- `run_hybrid_search.py` ofrece preflight, reanudación y regeneración del informe.
  Comprueba selección, folds, configuraciones, geometría, huellas y sellos;
  reproduce los cuatro controles guardados en los 50 cortes antes de usarlos.
- Se guardaron y auditaron **7.750 máscaras**. Las métricas recalculadas desde
  ellas coinciden con los CSV; todas quedan dentro de la ROI hepática.
- El análisis conserva elecciones internas/externas, métricas por paciente y
  tamaño, ganancias/pérdidas, fusiones/divisiones, negativos por área falsa,
  frontera sensibilidad–Dice, decisiones por candidato y figuras por corte.
- Las figuras incluyen máximo recall y máximo Dice híbridos elegidos con toda
  la cohorte, rotulados `exploratory_*`; no deben confundirse con las máscaras
  externas del selector.

La configuración elegida con toda la cohorte es `analysis/preset.yaml` y no
modifica `config.yaml`. La integración probada es 2D por API y ejecutor; no se
agregaron controles específicos a la GUI ni se validaron las extensiones
volumétricas, de compresión o reconstrucción con este método.

## Reproducción

```bash
.venv/bin/python scripts/run_hybrid_search.py --preflight
.venv/bin/python scripts/run_hybrid_search.py --run-id hybrid-nueva
.venv/bin/python scripts/run_hybrid_search.py --run-id hybrid-nueva --resume
.venv/bin/python scripts/run_hybrid_search.py --run-id hybrid-v1 --report-only
PYTHONPATH=src:. .venv/bin/python -m unittest tests.test_hybrid_segmentation -v
```

Artefactos: `results/hybrid_candidates/hybrid-v1/`. Protocolo y explicación de
salidas: `experiments/hybrid_candidates/README.md`. Las fuentes requeridas son
`local-v2-liver` y `otsu-bilateral-50`, conservadas localmente bajo `results/`.
Los datos y artefactos generados están ignorados por Git; este informe conserva
las cifras principales.

La selección utiliza referencias, la ROI hepática es conocida y los pacientes
ya fueron explorados. Los 83 componentes pertenecen a cortes 2D y no equivalen
a 83 lesiones volumétricas independientes. No se realizó validación externa.

## Pruebas y revisión visual

La suite completa ejecutó 150 pruebas: 149 correctas y una integración opcional
omitida. Las diez pruebas nuevas del híbrido cubren máscaras, selección,
reanudación, integridad de metadatos y coherencia del preset fijo con la API.
`git diff --check` terminó sin errores.

Se revisaron las figuras de todos los cortes con ganancias/pérdidas externas
frente a ambos controles: 9/371, 117/399, 117/462, 94/463, 94/577 y 76/111.
También se revisaron los cinco negativos de mayor área falsa del selector:
45/51, 45/61, 9/483, 101/557 y 45/22, además de la frontera sensibilidad–Dice.

En 76/111 las combinaciones recuperan la referencia pequeña, pero agregan 19 o
122 FP según la variante diagnóstica. En 94/577 recuperan las cinco referencias,
con 86 o 320 FP, frente a 19 de Otsu. En los negativos 45/51 y 45/61, Otsu ya
marca una gran superficie; conservar esa base conserva también ese error.
Estos ejemplos apoyan dos límites observados: falta discriminar los candidatos
pequeños verdaderos y también reconocer cuándo la máscara base es inadecuada.
No se introdujeron reglas especiales para esos casos después de inspeccionarlos.
