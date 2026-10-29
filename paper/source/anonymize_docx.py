"""Remove the editor identity reintroduced by Word without changing any document content."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
import hashlib,json,sys

def anonymize(path):
    path=Path(path)
    with ZipFile(path) as z:entries=[(i,z.read(i.filename)) for i in z.infolist()]
    before={i.filename:hashlib.sha256(data).hexdigest() for i,data in entries}
    out=[]
    for info,data in entries:
        if info.filename=='docProps/core.xml':
            root=etree.fromstring(data)
            for node in root:
                if etree.QName(node).localname in ['creator','lastModifiedBy']:node.text='Anonymous'
            data=etree.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
        out.append((info,data))
    temp=path.with_suffix('.metadata.tmp')
    with ZipFile(temp,'w') as z:
        for info,data in out:z.writestr(info,data)
    with ZipFile(temp) as z:
        assert all(hashlib.sha256(z.read(n)).hexdigest()==s for n,s in before.items() if n!='docProps/core.xml')
    temp.replace(path)
    return {'file':path.name,'changed_part':'docProps/core.xml','all_content_layout_math_image_parts_unchanged':True}

if __name__=='__main__':
    paths=[Path(p) for p in sys.argv[1:]] or list((Path(__file__).resolve().parent/'deliverables').glob('*.docx'))
    result=[anonymize(p) for p in paths]
    print(json.dumps(result,ensure_ascii=False))
