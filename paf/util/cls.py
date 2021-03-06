from enum import Enum
import json


class SupervisionMode(str, Enum):
    VALUE_ONLY = 'value',
    GRADIENT_ONLY = 'gradient',
    VALUE_AND_GRADIENT = 'gv'


PUBLIC_ENUMS = {
    "SupervisionMode": SupervisionMode
}


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj) in PUBLIC_ENUMS.values():
            return {"__enum__": str(obj)}
        return json.JSONEncoder.default(self, obj)


def as_enum(d):
    if "__enum__" in d:
        name, member = d["__enum__"].split(".")
        return getattr(PUBLIC_ENUMS[name], member)
    else:
        return d

