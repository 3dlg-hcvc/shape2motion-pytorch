from enum import Enum

class Stage(Enum):
    stage1 = 'stage1'
    stage2 = 'stage2'
    stage3 = 'stage3'

class JointType(Enum):
    ROT = 1
    TRANS = 2
    BOTH = 3