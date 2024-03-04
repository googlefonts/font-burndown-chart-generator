"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from fontTools.designspaceLib import DesignSpaceDocument, SimpleLocationDict, VariableFontDescriptor

def clamp(value, minimum, maximum):
    ...

@dataclass
class Range:
    minimum: float
    maximum: float
    default: float = ...
    def __post_init__(self): # -> None:
        ...
    
    def __contains__(self, value: Union[float, Range]) -> bool:
        ...
    
    def intersection(self, other: Range) -> Optional[Range]:
        ...
    


Region = Dict[str, Union[Range, float]]
ConditionSet = Dict[str, Range]
Rule = List[ConditionSet]
Rules = Dict[str, Rule]
def locationInRegion(location: SimpleLocationDict, region: Region) -> bool:
    ...

def regionInRegion(region: Region, superRegion: Region) -> bool:
    ...

def userRegionToDesignRegion(doc: DesignSpaceDocument, userRegion: Region) -> Region:
    ...

def getVFUserRegion(doc: DesignSpaceDocument, vf: VariableFontDescriptor) -> Region:
    ...
