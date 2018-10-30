from enum import Enum


class WorkType(Enum):
    train = 'train'
    evaluation = 'evaluation'
    test = 'test'
    predict = 'predict'
