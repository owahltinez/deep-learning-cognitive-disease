from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import numpy

@dataclass
class Box(object):
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0

    def __str__(self):
        return '%d,%d,%d,%d' % (self.left, self.top, self.right, self.bottom)

    def center(self) -> Tuple[int, int]:
        return (self.right - self.left) / 2, (self.bottom - self.top) / 2

    @staticmethod
    def load(box: str) -> 'Box':
        left, top, right, bottom = list(map(int, box.split(',')))
        return Box(left=left, top=top, right=right, bottom=bottom)
        
@dataclass
class Template(object):
    name: str
    path: Path
    image: numpy.ndarray
    height: int
    width: int

@dataclass
class PatientRecord:
    key: str
    processed_path: Path
    image_path: Path
    template_name: str
    template_path: Path
    template_box: Box
    drawing_box: Box
    pathological: bool
        
    @staticmethod
    def build_key(path: Path) -> str:
        fname = path.name
        fname = fname.replace('EV', 'v').replace('Ev', 'v').replace('ev', 'v')
        patient_id = int(fname.split('_', 2)[1].split('v', 2)[0])
        patient_ev = fname.split('_', 2)[1].split('v', 2)[1].split('.', 2)[0][0]
        return '%03d_%s' % (patient_id, patient_ev)
    
    def to_dict(self) -> Dict:
        foo_types = (type(lambda: None), type(self.to_dict))
        all_attrs = {k: getattr(self, k) for k in dir(self)}
        return {k: v for k, v in all_attrs.items() if not k.startswith('_') and not isinstance(v, foo_types)}
    
    @staticmethod
    def from_dict(dic: Dict) -> 'PatientDrawing':
        return PatientRecord(**dic)
            # key=dic['key'],
            # processed_path=Path(dic['processed_path']),
            # image_path=Path(dic['image_path']),
            # template_name=dic['template_name'],
            # template_path=Path(dic['template_name']),
            # template_box=Box.load(dic['template_box']),
            # drawing_box=Box.load(dic['drawing_box']),
            # pathological=int(dic['pathological']))
    