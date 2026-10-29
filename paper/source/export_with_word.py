"""Export the editable masters with installed Microsoft Word on Windows."""
from pathlib import Path
import win32com.client
from anonymize_docx import anonymize
base=Path(__file__).resolve().parent.parent
for name in ['STaR-Body_CVPR_2027','STaR-Body_Supplement']:
    app=win32com.client.DispatchEx('Word.Application')
    app.Visible=False
    app.DisplayAlerts=0
    app.AutomationSecurity=3
    doc=None
    try:
        doc=app.Documents.Open(str(base/(name+'.docx')),ReadOnly=False,AddToRecentFiles=False,Visible=False)
        doc.EmbedTrueTypeFonts=True
        doc.SaveSubsetFonts=False
        doc.DoNotEmbedSystemFonts=False
        doc.Save()
        doc.Repaginate()
        doc.ExportAsFixedFormat(str(base/(name+'.pdf')),17,OpenAfterExport=False,OptimizeFor=0,CreateBookmarks=1,DocStructureTags=True,BitmapMissingFonts=False,UseISO19005_1=False)
        print(name,doc.ComputeStatistics(2))
    finally:
        if doc is not None:doc.Close(False)
        app.Quit()
    anonymize(base/(name+'.docx'))
