# Revisión de la corrida 20261104-175047

Caso 76, cinco cortes (111,120,125,135,146). Detector HU desactivado; Otsu,
ventana centro 45/ancho 100, gaussiano sigma 0,3 píxeles, CLAHE, apertura y cierre
con disco radio 1, relleno de agujeros, área mínima 0. Se reprodujeron exactamente
las cinco máscaras guardadas antes de comparar once variantes.

| Variante | Detecciones IoU ≥0,10 | FP totales | Detecciones IoU ≥0,25 | Dice agregado del caso |
|---|---:|---:|---:|---:|
| Actual | 9/9 | 92 | 8/9 | 0,760 |
| Solo área mínima 10 píxeles | 9/9 | 45 | 8/9 | 0,765 |
| Solo área mínima 20 píxeles | 8/9 | 34 | 8/9 | 0,768 |
| Radio 2 y área 10, sigma original | 7/9 | 45 | 7/9 | 0,723 |
| Sigma 0,6 + radio 2 + área 10 | 9/9 | 48 | 9/9 | 0,774 |

Los FP corresponden a IoU ≥0,10 y se suman sobre los cinco cortes. Las nueve
referencias son observaciones de componentes 2D, no nueve lesiones volumétricas
independientes. La fila de área 20 pierde la única observación de menos de 10 mm.

Recomendación para la próxima comparación visual: mantener ventana, CLAHE y Otsu;
usar sigma gaussiano 0,6 **píxeles**, disco radio 2 **píxeles**, apertura/cierre y
relleno activados, área mínima 10 **píxeles** (5,51 mm² para este paciente).
Son controles del método clásico; los parámetros HU no participan aquí.
La alternativa más conservadora es cambiar solamente el área mínima a 10.

No conviene aumentar solo el radio, ni eliminar por área más agresivamente:
ambas opciones perdieron detecciones en esta prueba. Aumentar solo el suavizado
a 0,6 o 1 no dio una mejora; la combinación con morfología debe evaluarse junta.
El área mínima borra componentes aislados, pero no espículas unidas a una lesión.

El corte 125 no contiene tumor: quedan siete componentes falsos con ambas
propuestas, frente a diez originalmente. La limpieza no resuelve por sí sola
los falsos positivos de mayor tamaño. El resultado de esta comparación es local
y exploratorio, seleccionado tras observar este paciente; falta verificarlo en
los otros pacientes antes de cambiar defaults.

Artefactos en `results/diagnostics/20261104-175047/`: `compare.py` reproduce la
comparación, `per_slice.csv` y `summary.csv` contienen conteos, `comparison.png`
muestra máscaras sobre TC con referencia verde. `sugerencia.yaml` contiene una
copia de los parámetros propuestos; no se modificó la configuración del usuario.
