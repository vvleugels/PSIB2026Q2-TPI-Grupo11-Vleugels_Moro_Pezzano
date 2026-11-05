from pathlib import Path
import csv, json, statistics
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'Informe_TPI_Grupo11.docx'
RES = ROOT / 'results' / 'branch_otsu059_20261105'
FONT = 'Aptos'

def readcsv(p):
    with open(p, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd'); shd.set(qn('w:fill'), color); tcPr.append(shd)

def set_cell_text(cell, text, bold=False, color='202A35', size=8.2):
    cell.text = ''
    p = cell.paragraphs[0]; p.paragraph_format.space_after = Pt(0); p.paragraph_format.space_before = Pt(0)
    r = p.add_run(str(text)); r.bold = bold; r.font.name = FONT; r.font.size = Pt(size); r.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

def set_repeat_table_header(row):
    trPr = row._tr.get_or_add_trPr(); el = OxmlElement('w:tblHeader'); el.set(qn('w:val'), 'true'); trPr.append(el)

def table(doc, headers, rows, widths=None, size=8.0):
    t = doc.add_table(rows=1, cols=len(headers)); t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.style = 'Table Grid'; t.autofit = False
    set_repeat_table_header(t.rows[0])
    for i, h in enumerate(headers):
        set_cell_text(t.rows[0].cells[i], h, True, 'FFFFFF', size); shade(t.rows[0].cells[i], '17365D')
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, val in enumerate(row):
            set_cell_text(cells[i], val, False, '202A35', size)
            if ri % 2: shade(cells[i], 'F1F5F8')
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths): row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return t

def para(doc, text='', boldlead=None, size=9.5, after=5, align=None, italic=False):
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(after); p.paragraph_format.line_spacing = 1.08
    if align is not None: p.alignment = align
    if boldlead and text.startswith(boldlead):
        r=p.add_run(boldlead); r.bold=True; r.font.name=FONT; r.font.size=Pt(size)
        r=p.add_run(text[len(boldlead):]); r.font.name=FONT; r.font.size=Pt(size); r.italic=italic
    else:
        r=p.add_run(text); r.font.name=FONT; r.font.size=Pt(size); r.italic=italic
    return p

def heading(doc, text, level=1):
    p=doc.add_paragraph(style=f'Heading {level}'); p.paragraph_format.keep_with_next=True; p.paragraph_format.space_before=Pt(7 if level==1 else 4); p.paragraph_format.space_after=Pt(3)
    p.add_run(text); return p

def caption(doc, text):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after=Pt(5); p.paragraph_format.keep_together=True
    r=p.add_run(text); r.font.name=FONT; r.font.size=Pt(8); r.italic=True; r.font.color.rgb=RGBColor(85,96,108)

def add_page_field(p):
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    r=p.add_run('PSIB 2026 Q2  |  TPI Grupo 11  |  '); r.font.name=FONT; r.font.size=Pt(8); r.font.color.rgb=RGBColor(90,100,110)
    fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),'PAGE'); p._p.append(fld)

def fit_image(doc, path, width=6.6):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after=Pt(1)
    p.add_run().add_picture(str(path), width=Inches(width))

def add_refs(doc):
    doc.add_page_break(); heading(doc, 'Referencias', 1)
    refs=[
      '[1] M. Antonelli et al., “The Medical Segmentation Decathlon,” Nature Communications, vol. 13, art. 4128, 2022. doi:10.1038/s41467-022-30695-9.',
      '[2] P. Bilic et al., “The Liver Tumor Segmentation Benchmark (LiTS),” Medical Image Analysis, vol. 84, art. 102680, 2023. doi:10.1016/j.media.2022.102680.',
      '[3] N. Otsu, “A Threshold Selection Method from Gray-Level Histograms,” IEEE Transactions on Systems, Man, and Cybernetics, vol. 9, no. 1, pp. 62–66, 1979. doi:10.1109/TSMC.1979.4310076.',
      '[4] L. R. Dice, “Measures of the Amount of Ecologic Association Between Species,” Ecology, vol. 26, no. 3, pp. 297–302, 1945. doi:10.2307/1932409.',
      '[5] R. M. Haralick, K. Shanmugam, and I. Dinstein, “Textural Features for Image Classification,” IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-3, no. 6, pp. 610–621, 1973. doi:10.1109/TSMC.1973.4309314.',
      '[6] A. C. Kak and M. Slaney, Principles of Computerized Tomographic Imaging. New York: IEEE Press, 1988.',
      '[7] G. K. Wallace, “The JPEG Still Picture Compression Standard,” IEEE Transactions on Consumer Electronics, vol. 38, no. 1, pp. xviii–xxxiv, 1992. doi:10.1109/30.125072.',
      '[8] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image Quality Assessment: From Error Visibility to Structural Similarity,” IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004. doi:10.1109/TIP.2003.819861.',
      '[9] A. H. Andersen and A. C. Kak, “Simultaneous Algebraic Reconstruction Technique (SART): A Superior Implementation of the ART Algorithm,” Ultrasonic Imaging, vol. 6, no. 1, pp. 81–94, 1984. doi:10.1177/016173468400600107.',
    ]
    for ref in refs:
        p=para(doc, ref, size=8.5, after=5); p.paragraph_format.left_indent=Inches(.18); p.paragraph_format.first_line_indent=Inches(-.18)
    heading(doc, 'Fuentes del proyecto', 2)
    para(doc, 'La base local corresponde a un subconjunto de 10 pares etiquetados del conjunto Task03 Liver del Medical Segmentation Decathlon [1]. Se preservó el README y la atribución/licencia del subconjunto. Las figuras y tablas de resultados se generaron con los scripts del repositorio; la rama experimental indicada en Métodos no se integró a main.', size=8.5)

doc=Document()
sec=doc.sections[0]; sec.page_width=Inches(8.27); sec.page_height=Inches(11.69); sec.top_margin=Inches(.68); sec.bottom_margin=Inches(.68); sec.left_margin=Inches(.78); sec.right_margin=Inches(.78)
styles=doc.styles
for stylename in ['Normal','Title','Subtitle','Heading 1','Heading 2']:
    s=styles[stylename]; s.font.name=FONT; s.font.color.rgb=RGBColor(0,0,0)
styles['Normal'].font.size=Pt(9.5); styles['Normal'].paragraph_format.space_after=Pt(5)
styles['Heading 1'].font.size=Pt(14); styles['Heading 1'].font.bold=True
styles['Heading 2'].font.size=Pt(11); styles['Heading 2'].font.bold=True
styles['Title'].font.size=Pt(25); styles['Title'].font.bold=True
styles['Subtitle'].font.size=Pt(13)

# Cover
p=doc.add_paragraph(); p.paragraph_format.space_before=Pt(80); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=p.add_run('INSTITUTO TECNOLÓGICO DE BUENOS AIRES'); r.font.name=FONT; r.font.size=Pt(13); r.bold=True; r.font.color.rgb=RGBColor(23,54,93)
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(28)
r=p.add_run('16.63 Procesamiento de Señales e Imágenes Biomédicas'); r.font.name=FONT; r.font.size=Pt(14); r.bold=True
p=doc.add_paragraph(style='Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(65); p.paragraph_format.space_after=Pt(10)
p.add_run('Segmentación y cuantificación de lesiones hepáticas en tomografía computada')
p=doc.add_paragraph(style='Subtitle'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run('Trabajo Práctico Integrador  |  Grupo 11  |  Comisión BIO')
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(58)
r=p.add_run('INTEGRANTES'); r.font.name=FONT; r.font.size=Pt(10); r.bold=True; r.font.color.rgb=RGBColor(23,54,93)
for line in ['Tomas Pezzano  ·  64699','Amalia Moro  ·  65009','Vicente Vleugels  ·  63660']:
    para(doc,line,size=11,after=3,align=WD_ALIGN_PARAGRAPH.CENTER)
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(20)
r=p.add_run('DOCENTES'); r.font.name=FONT; r.font.size=Pt(10); r.bold=True; r.font.color.rgb=RGBColor(23,54,93)
para(doc,'Roberto Sebastián Tomás  ·  Francisco Joaquín Tassara',size=10.5,after=3,align=WD_ALIGN_PARAGRAPH.CENTER)
para(doc,'5 de noviembre de 2026',size=10,after=0,align=WD_ALIGN_PARAGRAPH.CENTER)

# Body page 1
doc.add_page_break(); heading(doc,'Resumen',1)
para(doc,'Se desarrolló un flujo clásico para explorar y segmentar lesiones hepáticas en cortes axiales de tomografía computada, con cuantificación morfológica y análisis de cómo la compresión y una reconstrucción simulada afectan la imagen. Se trabajó con diez casos etiquetados del subconjunto local de LiTS/Task03 Liver y cinco cortes separados por caso. El método exploratorio Otsu 059 obtuvo Dice medio 0,455 e IoU medio 0,354 en 50 cortes; superó al control de la misma rama (0,383 y 0,287), aunque presentó gran variabilidad y no constituye validación independiente porque fue seleccionado luego de explorar la cohorte. PNG fue reversible y JPEG redujo tamaño con pérdida creciente al bajar calidad. La textura GLCM no separó significativamente tumor e hígado sano en los nueve casos emparejables. La medición 3D sobre máscaras de referencia se implementó como análisis morfológico, no como segmentación automática 3D.')
heading(doc,'Introducción',1)
para(doc,'La segmentación de lesiones hepáticas en TC puede apoyar la medición reproducible de su extensión, pero el contraste, tamaño y forma de las lesiones varían. LiTS reúne tomografías abdominales con anotaciones de hígado y tumor y es una referencia para comparar métodos [1, 2]. En este trabajo se eligieron técnicas interpretables de procesamiento clásico, sin entrenar redes neuronales: una ventana de intensidades Hounsfield, suavizado, umbralización y operaciones morfológicas.')
para(doc,'El objetivo general fue construir un flujo reproducible que permita visualizar cortes, proponer máscaras tumorales y cuantificar su concordancia con la anotación de referencia. Como objetivo ampliado se describieron textura, morfología 3D de las máscaras de referencia, compresión y reconstrucción tomográfica simulada. El análisis es exploratorio y no busca diagnóstico clínico.')
heading(doc,'Materiales y métodos',1)
heading(doc,'Datos y selección',2)
para(doc,'Se usaron 10 pares imagen–máscara del subconjunto local Task03 Liver, de los casos 3, 9, 45, 57, 76, 94, 99, 101, 105 y 117. Los volúmenes son NIfTI y los cortes axiales tienen matriz 512 × 512; el espaciado físico varía entre casos. El subconjunto se distribuyó en terciles de volumen tumoral (4/3/3 casos) y siete casos presentan más de una lesión conectada. El caso 105 no tiene vóxeles tumorales en su máscara. Se seleccionaron cinco cortes por caso, un total de 50, priorizando cortes con tumor y separación espacial. Por ello, la muestra de cortes no representa la prevalencia de cortes positivos en un volumen completo.')
heading(doc,'Preprocesamiento',2)
para(doc,'La serie original ya contiene imágenes de TC reconstruidas; el proyecto no ejecuta reconstrucción clínica desde proyecciones adquiridas. Se transformaron intensidades HU a una ventana de centro 30 HU y ancho 150 HU, se aplicó filtro gaussiano con σ = 1 píxel y se omitió ecualización en el método Otsu 059. La región hepática de referencia restringió la búsqueda de candidatos. El preset predeterminado de la rama, con Gauss σ = 0,6 y CLAHE, se usó como control pareado. Como ambos cambios difieren juntos, esta comparación no aísla el efecto individual de CLAHE o de σ.')

# Body page 2
doc.add_page_break(); heading(doc,'Métodos de segmentación y análisis',1)
heading(doc,'Segmentación y evaluación',2)
para(doc,'Dentro del hígado se aplicó el umbral automático de Otsu, que separa intensidades por el histograma maximizando la separación entre clases [3]. Se eligió el lado de menor intensidad, bajo la hipótesis de que algunas lesiones son hipodensas. La máscara se limpió con apertura y cierre morfológicos, relleno de huecos y eliminación de componentes menores de 10 píxeles. Estas operaciones pueden borrar lesiones pequeñas o producir falsos positivos; por eso se evaluaron frente a las etiquetas y se inspeccionaron figuras.')
para(doc,'La concordancia por píxel se midió con Dice = 2|P∩R|/(|P|+|R|) [4] e IoU = |P∩R|/|P∪R|, donde P es la predicción y R la referencia. Se promediaron primero los píxeles de los cinco cortes de cada caso y luego se reportó la media entre diez casos. Un valor alto indica mayor solapamiento; el promedio no resume por sí solo la variabilidad ni distingue falsos positivos de falsos negativos.')
heading(doc,'Textura y morfología',2)
para(doc,'Se calcularon contraste, homogeneidad, energía y correlación de la matriz de coocurrencia de niveles de gris (GLCM) en regiones tumorales y de hígado sano de referencia [5]. Las diferencias se emparejaron por caso y se corrigieron por comparaciones múltiples mediante Holm. En 3D se midieron componentes, volumen físico y diámetros sobre las máscaras manuales de referencia con el espaciado NIfTI; esa medición describe la lesión anotada y no evalúa la calidad de una máscara automática volumétrica.')
heading(doc,'Compresión y reconstrucción simulada',2)
para(doc,'La comparación de compresión parte de la imagen ventaneada de 8 bits: PNG y Huffman se evaluaron como alternativas sin pérdida; RLE se estimó; JPEG se probó con calidades 95, 75, 50, 25 y 10. PSNR y SSIM cuantifican fidelidad de intensidad y estructura [7, 8], mientras Dice/IoU miden la segmentación resultante. Por tanto, los cocientes no describen la compresión del NIfTI original ni preservan directamente HU cuantitativos.')
para(doc,'La reconstrucción fue un experimento didáctico 2D sobre cortes TC ya reconstruidos: se simularon proyecciones y se recuperó la imagen mediante retroproyección filtrada (rampa o Hann), SART y un método de retroproyección simple, con 20, 60 y 180 ángulos [6, 9]. Se evaluaron tres cortes de un único caso; es evidencia ilustrativa, no una comparación clínica generalizable.')
heading(doc,'Interfaz y organización del código',2)
para(doc,'La GUI permite elegir caso y corte, ajustar preprocesamiento y umbral, ejecutar el flujo 2D, revisar figuras/tablas y lanzar compresión, reconstrucción y morfología 3D de referencia. Los scripts principales son `explore_dataset.py` y `select_cases.py` (inventario/selección), `preprocess_selected.py` (figuras de filtros), `run_segmentation.py` (máscaras y métricas), `extract_features.py` y `plot_features.py` (GLCM y gráficos), `analyze_morphology_3d.py` (medidas/superficies de referencia), `run_compression.py` y `run_reconstruction.py` (experimentos). Las funciones reutilizables están en `src/psib/`; `scripts/gui.py` implementa la interfaz.')

# Body page 3
doc.add_page_break(); heading(doc,'Resultados de segmentación',1)
para(doc,'La comparación pareada mantuvo los mismos diez casos y 50 cortes. Otsu 059 elevó el solapamiento medio respecto del control, pero la dispersión entre casos fue amplia y el resultado por caso no fue uniforme.')
table(doc,['Configuración','Casos / cortes','Dice medio ± DE','IoU medio ± DE','Cobertura ref.'],[
 ['Control de rama','10 / 50','0,383 ± 0,310','0,287 ± 0,260','—'],
 ['Otsu 059','10 / 50','0,455 ± 0,329','0,354 ± 0,282','0,593']
], widths=[1.6,1.0,1.4,1.35,1.1],size=8.5)
para(doc,'La cobertura media de la referencia fue 0,593 para los nueve casos con tumor. Por caso, Dice varió desde 0,002 (caso 45) y 0 (caso 105 sin tumor anotado, pero con predicción positiva) hasta 0,835 (caso 99). Los casos 76, 99, 101 y 117 superaron Dice 0,75; los valores bajos reflejan límites importantes del umbral global y de la limpieza morfológica. El caso 9, corte 386, ilustra un buen solapamiento puntual (Dice 0,922; IoU 0,856), pero no debe confundirse con el promedio del caso (0,600/0,429).')
fit_image(doc,RES/'segmentation'/'report_otsu059_20261105'/'predictions'/'cfg_000'/'case_009_slice_386_none.png',6.35)
caption(doc,'Figura 1. Caso 9, corte 386. De izquierda a derecha: TC en ventana HU, imagen suavizada, referencia y predicción binaria. La predicción coincide bien en este corte, aunque incluye un pequeño foco adicional.')
heading(doc,'Alcance de la selección del método',2)
para(doc,'Otsu 059 corresponde a ventana 30/150 HU, gaussiano σ = 1 y sin ecualización, con Otsu y limpieza morfológica. Se presenta como método principal exploratorio, según el barrido disponible; sin embargo, fue escogido después de observar resultados de la misma cohorte. La diferencia frente al control es descriptiva y no constituye una mejora validada en datos independientes. Además, cambiaron simultáneamente σ y CLAHE, de modo que no se atribuye la diferencia a un solo paso.')

# Body page 4
doc.add_page_break(); heading(doc,'Resultados de compresión y reconstrucción',1)
para(doc,'En los 50 cortes, PNG mantuvo la imagen ventaneada sin pérdida, con SSIM = 1 y una razón media de compresión de 3,35:1. La estimación Huffman fue 1,98:1. RLE expandió el tamaño (0,88:1), consistente con imágenes sin largas secuencias de niveles repetidos. JPEG redujo más el tamaño a costa de fidelidad; la razón q50 fue 13,22:1 y q10 fue 28,56:1.')
table(doc,['Condición','Razón media','PSNR medio','SSIM medio','Dice medio'],[
 ['PNG sin pérdida','3,35:1','sin cambio','1,000','0,400'],
 ['JPEG q95','4,87:1','51,26 dB','0,999','0,400'],
 ['JPEG q50','13,22:1','39,88 dB','0,984','0,403'],
 ['JPEG q10','28,56:1','31,80 dB','0,923','0,380'],
 ['RLE estimado','0,88:1','sin cambio','1,000','0,400']
],widths=[1.35,1.05,1.05,1.0,1.0],size=8)
para(doc,'Las diferencias de Dice de la tabla son pequeñas frente a la variación entre casos (desvío del control ≈ 0,416); q50 no demuestra que la pérdida mejore la segmentación. A calidad q10 bajaron SSIM y Dice medios. Los resultados corresponden a compresión con pérdida de una representación de 8 bits, no a almacenar las series originales en formato diagnóstico.')
fit_image(doc,RES/'compression'/'report-otsu059-20261105'/'figures'/'compression_example_grid.png',6.5)
caption(doc,'Figura 2. Ejemplo de compresión JPEG a distintas calidades sobre un corte seleccionado; el contorno naranja indica la anotación tumoral. La imagen de diferencia está amplificada visualmente.')
heading(doc,'Reconstrucción simulada',2)
para(doc,'En el corte de ejemplo evaluado, FBP con filtro rampa pasó de PSNR 9,12 dB y SSIM 0,274 con 20 ángulos a 22,87 dB y 0,902 con 180 ángulos. Aun así, el Dice tumor fue 0 con 180 ángulos en ese resumen, frente a 0,026 con 20 ángulos. La calidad global de la imagen no garantizó mejor solapamiento del tumor. La simulación sirve para mostrar artefactos y el efecto del muestreo angular; no reemplaza una reconstrucción desde datos crudos ni autoriza inferencias sobre dosis o práctica clínica.')

# Body page 5
doc.add_page_break(); heading(doc,'Textura, cuantificación 3D y conclusiones',1)
heading(doc,'Textura y morfología 3D',2)
para(doc,'En nueve pares tumor–hígado sano, ninguna de las cuatro características GLCM mostró una diferencia significativa tras el ajuste de Holm (p ajustado = 1,00 en todas). El contraste tuvo diferencia media tumor menos hígado de +0,395; homogeneidad −0,027; energía −0,028 y correlación −0,016. El tamaño de muestra pequeño y el análisis sobre cortes seleccionados limitan la potencia y la generalización; no se concluye que la textura sea inútil en general.')
para(doc,'El análisis morfológico 3D de las etiquetas de referencia describió, por ejemplo, una componente en el caso 3: volumen 1,343 cm³, extensión de caja 14,56 × 16,64 × 15,00 mm y Feret 3D 17,32 mm. Es una medición geométrica sobre la máscara de referencia; no representa resultado del segmentador automático. El GUI puede mostrar estas superficies y tablas como herramienta de revisión.')
fit_image(doc,ROOT/'results'/'morphology_3d'/'case_003_surface_3d.png',4.0)
caption(doc,'Figura 3. Superficie de una lesión del caso 3 reconstruida a partir de la máscara de referencia en el volumen; no es una predicción automática.')
heading(doc,'Conclusiones y limitaciones',2)
para(doc,'Se cumplió el objetivo de implementar un flujo reproducible para selección, preprocesamiento, segmentación 2D, visualización y cuantificación comparativa. Otsu 059 obtuvo mejores promedios que el control pareado, pero la variabilidad, los falsos positivos y la selección post hoc impiden afirmar robustez o utilidad clínica. La textura GLCM no distinguió los tejidos en este análisis. PNG preservó la representación ventaneada; JPEG q50 mantuvo métricas promedio cercanas al control en este conjunto y q10 redujo calidad y Dice. La reconstrucción simulada mostró que PSNR/SSIM e índice de segmentación responden a aspectos diferentes.')
para(doc,'Como trabajo futuro se propone fijar el método antes de evaluar una cohorte independiente, comparar de manera aislada filtros y operaciones morfológicas, incorporar criterios explícitos de ausencia de lesión, cuantificar predicciones 3D frente a la referencia y repetir compresión/reconstrucción en más volúmenes. Las conclusiones deben leerse como resultados exploratorios sobre diez casos y cortes seleccionados, no como validación clínica.')

add_refs(doc)

# Header/footer
for section in doc.sections:
    section.different_first_page_header_footer=True
    footer=section.footer.paragraphs[0]; add_page_field(footer)
    header=section.header.paragraphs[0]; header.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    r=header.add_run('PSIB 2026 Q2  |  TPI'); r.font.name=FONT; r.font.size=Pt(8); r.font.color.rgb=RGBColor(90,100,110)

doc.core_properties.title='Segmentación y cuantificación de lesiones hepáticas en tomografía computada'
doc.core_properties.subject='Trabajo Práctico Integrador de Procesamiento de Señales e Imágenes Biomédicas'
doc.core_properties.author='Tomas Pezzano, Amalia Moro y Vicente Vleugels'
doc.save(OUT)
print(OUT)
