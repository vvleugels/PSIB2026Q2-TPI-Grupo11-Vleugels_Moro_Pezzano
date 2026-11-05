# Revisión del TP y calidad de detecciones — 4 de noviembre de 2026

## Conclusión

Hay problemas corregibles de coherencia del pipeline y límites del segmentador por intensidad. La prioridad es unificar qué imagen se segmenta y qué máscara se evalúa, corregir las métricas de detección y recién después ajustar parámetros. Agregar más realce o morfología sin medir cada cambio no garantiza mejoras.

Revisé carga y selección, preprocesamiento, segmentación, bordes, GUI, métricas, evaluación por lesión, geometría/textura, extensión 3D, compresión, reconstrucción y comparación estadística. Ejecuté los 95 tests existentes: todos pasan. La revisión experimental usa **15 cortes de 3 pacientes (57, 76, 94)** de la corrida completada `20261103-135329`; no constituye una evaluación de los 10 pacientes disponibles ni una validación independiente. No ejecuté todos los experimentos volumétricos ni validé interactivamente toda la GUI.

## Evidencia medida

La auditoría reproduce exactamente los conteos y Dice/IoU por paciente de la corrida guardada. Compara 30 variantes con la misma selección, máscara hepática y referencia tumoral. El Dice se calcula acumulando píxeles por paciente y luego promediando pacientes, como en el ejecutor original.

| Variante | Dice medio | IoU medio |
|---|---:|---:|
| Corrida guardada: sin denoising, CLAHE, umbral 0,5, área mínima 25 px | 0,226 | 0,139 |
| Watershed guardado, aplicado a esa predicción | 0,190 | 0,111 |
| Quitar únicamente CLAHE | 0,147 | 0,087 |
| Sin CLAHE + Otsu, conservando área mínima 25 px | 0,346 | 0,252 |
| Gaussiano σ=1 px + sin CLAHE + Otsu + apertura disco r=1 px + mínimo 25 px | 0,420 | 0,305 |

La última variante es la mejor de este barrido **sobre los mismos casos utilizados para compararlas**. No es una configuración final validada.

| Paciente | Dice guardado | Dice mejor variante media | Observación |
|---|---:|---:|---|
| 57 | 0,188 | 0,156 | Empeora: aparecen demasiados falsos positivos. |
| 76 | 0,460 | 0,773 | Mejora importante. |
| 94 | 0,029 | 0,332 | Recupera tumor, pero sigue teniendo baja precisión. |

En el caso 94, el recall de píxeles pasa de 1,6 % a 58,4 %, pero la precisión queda en 23,2 %. En el caso 57, la precisión de la variante baja a 8,7 %. En los cuatro cortes sin tumor, los píxeles falsamente positivos aumentan de 1.625 a 4.490; ambas variantes tienen falsos positivos en tres de esos cuatro cortes. Optimizar solo Dice medio ocultaría este costo.

## Hallazgos prioritarios

### 1. Escala de segmentación y HU incompatibles después de CLAHE

`scripts/run_segmentation.py:38` y `scripts/gui.py:715` aplican realce antes de segmentar. Sin embargo, `build_configurations` convierte umbrales de 30/40/50 HU usando la relación lineal de la ventana y los exporta como `threshold_hu`. Esa relación deja de representar HU después de una ecualización local. La documentación de `segment_slice` pide la imagen `denoised`, no la ecualizada.

Además, con la ventana actual centro=30/ancho=150, el umbral 0,45 corresponde a **22,5 HU antes del realce**, no a los 40 HU del ejemplo antiguo centro=60/ancho=400. El 0,5 de la corrida equivale a 30 HU antes del realce. En los cortes tumorales del caso 94 medí medianas tumorales de 74–83 HU, frente a 100–101 HU de hígado sano: mantener ese umbral tras quitar CLAHE deja afuera casi todo el tumor. Por eso quitar CLAHE por sí solo empeora.

**Cambio propuesto:** separar imagen de visualización y entrada del segmentador. Mantener una rama de ventana HU + denoising para la segmentación base. Si se desea comparar CLAHE como entrada, declararlo como otra variante, con umbrales propios sin etiquetarlos como HU. Recalibrar conjuntamente ventana y umbral; registrar el umbral efectivo de Otsu/estadístico por corte.

### 2. No se ejecuta el mismo algoritmo en todos los módulos

`evaluate_morphology_3d.py:79`, compresión y reconstrucción usan `prepare_denoised_slice` sin el realce del ejecutor 2D. Una configuración ganadora con CLAHE se reutiliza en 3D sobre otra imagen. Los controles internos de compresión/reconstrucción pueden servir para sus comparaciones, pero no son el mismo baseline del ejecutor principal.

**Cambio propuesto:** una función compartida que produzca entrada de segmentación, máscara base y variantes opcionales; usarla en GUI, 2D y 3D, y explicitar las transformaciones adicionales de compresión/reconstrucción. Añadir una prueba de igualdad de máscara para el mismo corte y configuración entre esas rutas.

### 3. La máscara que se muestra no siempre es la que se cuantifica

El panel final de Sobel en `export_slice_comparison` muestra watershed. `summary.csv`, la selección de ganadores y `evaluate_lesions.py` usan la máscara base `.npy`; watershed se guarda aparte. El encabezado de la figura incluye ambos Dice, pero el flujo presenta dos candidatos distintos sin una elección única de salida final. La GUI interactiva calcula la máscara base.

Watershed parte de componentes de esa máscara; si no hay predicción, devuelve vacío. No puede recuperar una lesión omitida lejos de las semillas. En el caso 94/corte 577, tanto la máscara base como watershed están vacías pese a la referencia positiva.

**Cambio propuesto:** identificar `base`, `watershed` y `sharpened` explícitamente en figuras, métricas y evaluación por lesión; elegir una variante final y usarla consistentemente. No activar watershed como mejora asumida: en esta corrida empeora el Dice.

### 4. Un umbral de intensidad no discrimina suficientemente tumor de otras regiones

La predicción usa intensidad dentro de una máscara hepática conocida; no utiliza las características GLCM o geométricas extraídas después. `extract_features.py` describe regiones de referencia, no aporta esas variables al detector. Otsu tampoco cuenta con una decisión explícita de ausencia de tumor: divide intensidades incluso en cortes negativos no constantes.

**Experimentos siguientes:** comparar umbral fijo calibrado, Otsu y estadístico en la rama consistente; medir contraste local y características de los componentes para reducir falsos positivos. Probar semillas más selectivas y crecimiento restringido como una variante posterior, sin usar la máscara tumoral para generarlas. La continuidad entre cortes puede estudiarse como extensión, pero no sustituye la validación 2D mínima del TP. Son hipótesis de mejora, no resultados ya demostrados.

### 5. La evaluación por lesión puede ocultar fallos

- `evaluate_lesions.py:48` devuelve F1=NaN cuando TP=0, FP=1 y FN=1. Debe ser 0. Luego `dropna()` excluye ese fracaso del promedio. Reproducido con un ejemplo mínimo. Calcular `2*TP/(2*TP+FP+FN)` y definir aparte el caso totalmente vacío.
- `min_iou: 0` acepta cualquier solapamiento positivo. Eso permite correspondencias muy débiles. Reportar sensibilidad a criterios de correspondencia predefinidos, junto con Dice por grupo.
- Se cuentan grupos de correspondencia con divisiones/fusiones, no necesariamente lesiones individuales. Mantener esos grupos para cuantificar geometría, pero distinguirlos del conteo de detección por lesión; evaluar un emparejamiento uno-a-uno y reportar fusiones/divisiones aparte.
- Agregar falsos positivos por corte negativo y recall por tamaño; la media de Dice acumulado por paciente puede ocultar que las lesiones pequeñas se pierden.

## Mejoras del resto del TP

- **Diseño experimental:** usar los 10 casos previstos y separar ajuste de evaluación por paciente, nunca mezclando cortes del mismo paciente entre ambos. Con pocos casos, considerar validación cruzada por paciente con ajuste dentro de cada partición. Las selecciones basadas en referencia y la máscara hepática conocida deben seguir declaradas como alcance del experimento. No presentar el mejor resultado del barrido como generalización.
- **Morfología:** área mínima, radio y sigma están en píxeles; el filtro 3D está en vóxeles. Convertir parámetros a mm²/mm/mm³ usando spacing para comparar pacientes. Medir específicamente pérdidas de lesiones pequeñas al aplicar apertura.
- **Selección:** el fallback incluye un corte del caso 57 con solo 36 píxeles hepáticos. Conservar el criterio y rol en las tablas, pero revisar su utilidad para representar negativos; priorizar cortes negativos con región hepática suficiente mediante una regla predefinida.
- **Textura/geometría:** las medidas de referencia sirven para caracterizar el dataset. Separar claramente esas tablas de las medidas de predicción; cualquier uso de textura para clasificar candidatos exige ajuste y evaluación por paciente.
- **Concordancia:** `analyze_agreement.py:43` guarda `mean_of_pair_means` sin dividir por dos. En un ejemplo con áreas referencia [10,20] y predicción [12,22], devuelve 32 en vez de 16. El eje de la figura sí divide por dos; el error afecta ese campo del resumen, no implica que todos los límites de concordancia estén mal.
- **Compresión/reconstrucción:** mantenerlos como experimentos de degradación/simulación; no resuelven la falta de discriminación del detector. Unificar el baseline antes de comparar su impacto en detección.
- **Pruebas:** agregar regresiones para F1 cero, media de pares y coherencia entre rutas. La suite actual verifica muchos contratos, pero pasar 95 tests no demuestra buena segmentación en datos reales.

## Orden de trabajo recomendado

1. Corregir coherencia de entrada/salida y errores de métricas; conservar la corrida actual como baseline histórico.
2. Congelar selección y protocolo por paciente; incluir negativos y lesiones pequeñas explícitamente.
3. Comparar las variantes clásicas con parámetros en unidades físicas. La combinación Otsu medida arriba es un candidato experimental, no reemplazar aún el default por ella.
4. Atacar los falsos positivos con información local y por componente, evaluando también cuánto tumor se pierde.
5. Con el detector elegido, regenerar cuantificación, figuras, comparación de métodos y extensiones 3D/compresión/reconstrucción.

## Archivos y reproducción

Se agregó `scripts/audit_detection.py`, que no modifica defaults ni corridas previas. Sus salidas están en `results/audit/20261104/`: variantes, configuración original, selección, métricas por corte/paciente, resumen e intensidades HU por región. La carpeta `results/` está ignorada por Git; este informe conserva los resultados principales.

```bash
.venv/bin/python scripts/audit_detection.py \
  --run-dir results/segmentation/20261103-135329 \
  --output results/audit/nueva_revision

.venv/bin/python -m unittest discover -s tests -q
```

La salida de auditoría debe ser una carpeta nueva. Las cifras publicadas aquí son exploratorias y no sustituyen la evaluación pendiente en los otros siete pacientes.
