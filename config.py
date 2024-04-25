import json

class Config():

    def __init__(self, *args, **kwargs):
        self.config = json.load(open('config.json', 'r'))
        self.label_mapping = {
            'Sliding Two Fingers Up': 0,
            'Sliding Two Fingers Left': 1,
            'Sliding Two Fingers Right': 2,
            'Sliding Two Fingers Down': 3,
            'Zooming In With Two Fingers': 4,
            'Zooming Out With Two Fingers': 5,
            'Swiping Right': 6,
            'Swiping Up': 7,
            'Swiping Down': 8,
            'Thumb Up': 9,
            'Thumb Down': 10,
            'Stop Sign': 11,
            'Drumming Fingers': 12,
            'Doing other things': 13,
            'No gesture': 14
        }