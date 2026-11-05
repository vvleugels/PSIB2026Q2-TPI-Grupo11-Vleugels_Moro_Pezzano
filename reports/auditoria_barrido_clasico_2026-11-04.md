# Auditoría del barrido clásico — 4 de noviembre de 2026

## Conclusión

El barrido no demuestra que se haya alcanzado el techo de los métodos clásicos.
Demuestra el comportamiento de **una familia muy concreta: Otsu global sobre
imágenes ventaneadas, con tres opciones de ecualización y seis limpiezas**.
Hay limitaciones verificables en el objetivo y en el espacio de búsqueda.
No encontré un error de conteo que explique por sí solo los malos resultados.
La conclusión original «no hay mejora uniforme» es compatible con los datos;
interpretarla como «no se puede obtener un resultado mejor» no lo es.

## 1. El objetivo no expresa bien la calidad que se pretende obtener

`src/psib/classical_search.py:65-81` prioriza F2 por componentes. El recall de
pequeñas y Dice solo desempatan: no protegen contra una caída si F2 aumenta.
F2 está bien calculado, pero penaliza la cantidad de componentes falsos, no su
superficie. Tampoco mide directamente cuánto de cada lesión se delimita.

En los resultados externos guardados, seleccionar por F2 reduce FP de 884 a 342,
pero aumenta los píxeles falsos de 49.879 a 70.449 (+41,2 %) y pierde nueve
detecciones netas. La precisión por píxel cae de 42,5 % a 34,2 %.
No significa que se haya manipulado el evaluador: significa que mejorar ese
objetivo admite máscaras peores en aspectos relevantes.

Repetí la selección sobre los mismos ocho pacientes internos de cada fold,
evaluando después en los dos retenidos. Cambié exclusivamente el objetivo y
conservé los desempates de `ranking`:

| Selección | TP / 83 | Pequeñas / 32 | FP/corte | Dice por paciente | F2 | Píxeles FP |
|---|---:|---:|---:|---:|---:|---:|
| Default fijo | 62 | 15 | 17,68 | 0,383 | 0,243 | 49.879 |
| F2 original, reproducido | 53 | 11 | 6,84 | 0,408 | 0,365 | 70.449 |
| Dice, auditoría posterior | 61 | 12 | 8,18 | **0,455** | **0,380** | 79.271 |
| Recall original, reproducido | 73 | 24 | 492,72 | 0,194 | 0,015 | 243.796 |
| Recall de pequeñas, auditoría posterior | 74 | 26 | 544,44 | 0,177 | 0,013 | 246.199 |

Dice selecciona `otsu_059` en los cinco folds: ventana 30/150, gaussiano sigma
1 píxel, sin ecualización, perfil P5. Frente a la selección F2 mejora Dice y
recupera ocho detecciones netas, con más FP. Incluso obtiene mayor F2 externo;
maximizar F2 interno no garantiza maximizarlo en los pacientes retenidos.
Frente al default no mejora todo, especialmente pequeñas y superficie falsa.

Esta comparación fue decidida después de conocer la cohorte: es diagnóstico
post hoc, **no una nueva validación independiente ni prueba para promover un preset**.

## 2. La limpieza está eliminando lesiones que sí eran detectables

Comparé identidades de referencia, no solo diferencias de conteos. P0 y P5
comparten exactamente ventana, suavizado, ecualización y Otsu; cambia el paquete
apertura/cierre de radio 2, relleno y área mínima 10:

| Sin limpieza → P5 | TP antes → después | Detectadas antes y perdidas | De esas, pequeñas | Nuevas detecciones |
|---|---:|---:|---:|---:|
| `otsu_036` → `otsu_041` | 75 → 57 | 18 | 15 | 0 |
| `otsu_048` → `otsu_053` (default) | 73 → 62 | 16 | 13 | 5 |
| `otsu_054` → `otsu_059` | 72 → 61 | 11 | 11 | 0 |
| `otsu_108` → `otsu_113` | 74 → 49 | 25 | 17 | 0 |

La ablación atribuye el cambio al **paquete de limpieza**, no permite adjudicar
toda la pérdida a la apertura por separado. Una apertura puede borrar regiones
delgadas aunque su área total supere 10 píxeles; bajar solo `min_area` no resuelve
ese problema. Los perfiles actuales no incluyen apertura sola ni apertura por
reconstrucción, y acoplan varias operaciones. Conviene desacoplarlas y medir
qué referencias se pierden en cada paso.

Quitar toda limpieza tampoco resuelve el problema: por ejemplo `otsu_054`
produce 6.332 FP en 50 cortes. Hay que conservar candidatos verdaderos y
rechazar falsos con información adicional, en lugar de basarse únicamente
en eliminar estructuras pequeñas.

## 3. Hay 216 configuraciones, pero escasa diversidad de criterios

`src/psib/classical_search.py:20-33` fija Otsu, detector HU apagado, gaussiano o
ningún suavizado, CLAHE de kernel 8 y límite 0,01 cuando está activado.
No compara umbral estadístico, cuantiles, contraste local multiescala,
crecimiento por semillas, ni otros suavizados disponibles en el proyecto.
Cambiar ventanas tampoco equivale a probar tres detectores independientes:
sin saturación, una transformación afín positiva conserva la separación ideal
de Otsu, salvo discretización. Aquí el recorte sí puede cambiarla.

No hay un fallo que repita toda la grilla: los 216 vectores de predicciones sobre
la cohorte son distintos. Por corte hay entre 43 y 216 máscaras distintas,
con mediana 202. La limitación es conceptual, no un supuesto duplicado total.

`segment_slice` toma siempre el lado oscuro para Otsu. No comprueba si hay
evidencia de tumor ni incorpora características de los componentes. En una
ROI no constante el umbral divide intensidades aunque el corte sea negativo;
la limpieza puede vaciar la salida, pero no existe una decisión explícita de
ausencia. Ecualización global usa la imagen completa, no solo hígado; CLAHE
se aplica antes de Otsu con una única escala local. Son hipótesis a revisar,
no errores numéricos demostrados por esta auditoría.

## 4. OR de máscaras no equivale a sumar detecciones

`greedy_union` comienza por el mejor individual y solo admite incorporaciones
con mejora estricta de F2. No explora todas las parejas ni tríos, y no compara
fusión por candidatos, consenso ni crecimiento condicionado por semillas.
La ausencia de mejora de esta búsqueda no descarta otras combinaciones.

OR puede unir componentes verdaderos mediante un puente, aumentar su área y
bajar su IoU. Con correspondencia uno-a-uno, dos lesiones fusionadas pueden
pasar a contar como una. Por eso ni el TP ni F2 son monótonos al agregar máscaras,
aunque la cobertura de píxeles de referencia sí lo sea. El evaluador actual
recalcula las uniones correctamente; el problema es esperar complementariedad
de variantes muy similares y de una fusión tan simple.

## 5. Escala física, cohorte y reproducibilidad

- Diez píxeles representan entre 4,624 y 10 mm² según paciente. Radio 2 equivale
  a entre 1,36 y 2 mm. Se exporta esta diferencia, pero no se normaliza el filtro.
- Los 83 objetos son componentes 2D de 50 cortes seleccionados con referencia,
  no 83 tumores volumétricos independientes. La separación por paciente es
  correcta; el ajuste sucesivo sobre estos mismos pacientes limita las
  conclusiones de generalización.
- Agrupar las cuatro particiones internas no es por sí mismo una fuga: estas
  configuraciones no ajustan un modelo. Pero el F2 agrupado pondera conteos,
  mientras Dice promedia pacientes; eso también cambia qué se optimiza.
- Riesgo de caché pendiente: el fingerprint de `run_classical_search.py:275`
  incluye configuración base, selección y folds, pero no versión del código,
  grilla completa ni contenido de los volúmenes. Reanudar después de cambiar
  métodos puede mezclar máscaras anteriores con metadatos nuevos. No encontré
  evidencia de que esto haya causado los resultados actuales. Antes de un
  nuevo barrido conviene versionar la grilla y el código y validar artefactos.

## Siguiente experimento recomendado

1. Definir la prioridad: delimitación (Dice/cobertura) y conservación de lesiones
   pequeñas deben aparecer como objetivos explícitos. Reportar una frontera de
   compromisos con cantidad **y área** de FP; no elegir solo entre F2 y recall
   ilimitado ni imponer un presupuesto de FP sin justificarlo.
2. Mantener Otsu como control y desacoplar umbral de limpieza. Usar unidades
   físicas, probar reconstrucción morfológica y registrar detección por lesión
   antes/después de cada operación. No volver a barrer solo ventanas y radios.
3. Evaluar candidatos por contraste local a varias escalas, persistencia y
   características de componente, con crecimiento que conserve la extensión
   de la lesión. El detector HU ya existente es otra fuente de candidatos:
   su resultado previo no demuestra que el rechazo de FP esté resuelto.
4. Si se combinan ramas, estudiar candidatos y semillas complementarios antes
   de OR ciego de máscaras finales. La continuidad entre cortes adyacentes
   sería una extensión posterior que requiere cargar y evaluar esos cortes.
5. Congelar el procedimiento antes de evaluar pacientes no utilizados para
   estas decisiones. Con la cohorte actual, etiquetar nuevas comparaciones
   como exploratorias y presentar resultados por paciente y tamaño.

Estas propuestas son hipótesis comprobables, no una promesa de un Dice concreto.
La evidencia sí justifica seguir investigando y cambiar el enfoque del barrido.

## Reproducción y alcance de los cambios

Se agregó `scripts/audit_classical_search.py`; salidas nuevas en
`results/audit/classical-search-review-20261104/`. Se conservaron las corridas,
los métodos y la configuración del usuario. La auditoría reproduce los
resúmenes originales de F2 y máxima sensibilidad, y exporta elecciones por
fold, métricas, identidades de lesiones perdidas y máscaras únicas por corte.

```bash
.venv/bin/python scripts/audit_classical_search.py --output results/audit/nueva_auditoria_clasica
PYTHONPATH=src .venv/bin/python -m unittest tests.test_classical_search tests.test_segmentation tests.test_lesion_eval -q
```

Las 25 pruebas dirigidas pasan. Se inspeccionó además la figura del caso 94,
corte 541: la baja cobertura con la selección F2 y la abundancia de fragmentos
sin limpieza son coherentes con los conteos. No se regeneraron volúmenes ni
se ejecutó una nueva familia de detectores en esta revisión.
