# Reevaluación sin imponer 8 falsos positivos por corte

El límite de 8 FP/corte era exploratorio y demasiado restrictivo para comparar
las variantes nuevas. Al retirarlo, el detector recupera más lesiones que Otsu,
especialmente pequeñas, pero genera una cantidad muy alta de falsos positivos.

Se reutilizaron las máscaras de 48 variantes, los mismos 50 cortes de diez
pacientes y los folds congelados. Para cada límite, la selección usa solo ocho
pacientes internos y se evalúa en los otros dos. Este análisis fue decidido
después de ver los resultados originales: es una reevaluación interna posterior,
no evidencia independiente ni una nueva validación externa.

| Método / límite interno | TP / 83 referencias | Sensibilidad | Pequeñas / 32 | Sensibilidad pequeñas | FP/corte observado |
|---|---:|---:|---:|---:|---:|
| Histórico fijo | 25 | 30,1 % | 3 | 9,4 % | 7,78 |
| Otsu coherente fijo | 64 | 77,1 % | 15 | 46,9 % | 9,68 |
| Detector, límite 32 | 58 | 69,9 % | 14 | 43,8 % | 29,88 |
| Detector, límite 48 o 64 | 65 | 78,3 % | 17 | 53,1 % | 39,72 |
| Detector, límite 128 | 66 | 79,5 % | 18 | 56,3 % | 102,02 |
| Detector, límite 192 | 70 | 84,3 % | 21 | 65,6 % | 156,52 |
| Detector, sin límite | 71 | 85,5 % | 22 | 68,8 % | 171,74 |

Criterio principal: emparejamiento uno-a-uno con IoU ≥0,10. Las referencias son
componentes 2D, no lesiones volumétricas independientes. Los valores de FP son
los observados en pacientes retenidos, no los límites usados para seleccionar.

## Interpretación

- Sin límite, frente a Otsu hay una ganancia neta de 7 detecciones entre 83
  referencias y de 7 entre las 32 pequeñas. No implica que sean exactamente
  las mismas lesiones recuperadas: también puede haber pérdidas y ganancias.
  Los falsos positivos aumentan de 484 a 8587 en 50 cortes (unas 17,7 veces).
- Con límite 48, la ganancia neta es de una detección total y dos pequeñas,
  con 1986 FP frente a 484 de Otsu (unas 4,1 veces). Es un punto intermedio
  para inspeccionar, no un límite recomendado ni un ganador independiente.
- Sin límite, la precisión es 0,82 % y hay 152,95 FP por corte negativo.
  El problema no se limita a fragmentar lesiones verdaderas: aparecen muchas
  regiones candidatas en cortes sin tumor.
- La mejora de sensibilidad sin límite ocurre en cuatro pacientes (76, 99,
  101 y 117); en los otros cinco con lesiones empata a Otsu. Ambos fallan
  en el paciente 45. El paciente 105 no tiene lesiones en estos cortes.
- Con IoU ≥0,25, el detector sin límite alcanza 78,3 % frente a 69,9 % de Otsu;
  en pequeñas, 56,3 % frente a 40,6 %. Sin embargo, el Dice medio por paciente
  es 0,221 frente a 0,414: encontrar más componentes no mejora automáticamente
  la delimitación de las lesiones.
- No hay configuraciones factibles con límites 8, 12 o 16 en ningún fold.
  Con 24 hay cuatro de cinco; su resultado mezcla detección y abstención.
  Por eso no se interpretan esos puntos como calidad de máscaras vacías.

La conclusión anterior debe matizarse: **el detector sí aumenta la sensibilidad
si se admiten muchos más FP**, pero el costo observado es considerable incluso
priorizando no perder lesiones. No corresponde descartarlo solo por el número 8,
ni activarlo automáticamente por su recall. Otsu sigue siendo una referencia
fuerte de equilibrio; el detector nuevo es útil como generador de candidatos
para investigar cómo reducir falsos positivos conservando su ganancia en pequeñas.
Esta reevaluación no modifica los defaults ni reescribe la decisión original.

## Reproducción y archivos

```bash
.venv/bin/python scripts/evaluate_fp_tradeoff.py \
  --run results/segmentation/classical-20261104 \
  --output results/segmentation/classical-20261104/fp_tradeoff
```

En ese directorio: `report.md` detalla todos los límites y pacientes;
`summary.csv`, `selected_configurations.csv`, `per_case.csv` y
`outer_per_slice.csv` permiten auditar los resultados; `tradeoff.png` muestra
sensibilidad total y pequeña frente a FP. `exploratory_frontier.csv` identifica
variantes no dominadas en toda la cohorte, sin confundirlas con selección interna.

Las 11 pruebas del módulo de detección pasan, incluida la selección sin límite,
la abstención bajo restricciones inviables y la conservación del criterio original.
