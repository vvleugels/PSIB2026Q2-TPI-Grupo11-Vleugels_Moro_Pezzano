# Diagnóstico del caso 94, corte 593

Se examinó la corrida `20261104-171431` y se recalculó el detector con su
configuración guardada, sin usar la referencia para detectar.

La ventana de pantalla es centro 30, ancho 150: recorta HU a [-45,105].
El 45,51 % de los píxeles de la ROI hepática alcanza o supera 105 HU y queda
blanco. Sus percentiles 5/25/50/75/95 son 46/79/101/123/156 HU.
La ventana ilustrativa centro 100/ancho 200 muestra diferencias que estaban
saturadas; no se propone como parámetro validado ni modifica la detección.

La figura exportada duplicaba la imagen original en el panel «HU detector».
Se corrigió `run_detection.py` para mostrar allí los HU suavizados dentro de
la ROI con la misma ventana y guardar `smoothed_hu` en los diagnósticos NPZ.
La corrección afecta futuras figuras, no las máscaras ni los resultados históricos.

La detección usa HU sin ventana ni CLAHE. En este corte, sigma 0,6 mm equivale
a aproximadamente 0,72 píxeles; la dispersión robusta es 19,03 HU y los umbrales
alto/bajo son 1/0,5. El 17,08 % de los píxeles hepáticos fuera de la referencia
ya son semillas, y el 31,19 % queda en la predicción. Se cubre el 69,07 % de los
píxeles de referencia. Son porcentajes de píxeles, no sensibilidad por lesión.

El mapa de puntuación muestra respuesta a textura dispersa además de lesiones.
Por lo tanto, hay dos problemas distintos: saturación visual y baja selectividad
del detector. Mejorar la ventana no cambia la máscara actual. Aplicar CLAHE
con la configuración guardada tampoco constituye una solución demostrada:
el panel diagnóstico mantiene una textura intensa y la saturación de entrada
ya perdió diferencias de intensidad.

Artefactos locales: `results/diagnostics/case094-593/diagnosis.json` y
`results/diagnostics/case094-593/hu_diagnosis.png`. El contorno de referencia
se utiliza únicamente para diagnóstico. Las conclusiones son de este corte;
cualquier cambio de suavizado o selectividad debe compararse en la cohorte.
