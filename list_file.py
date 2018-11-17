import numpy as np
import json

class Lists():
    def __init__(self):
        self.targets = ['bottle', 'cup', 'cell phone']
        self.references = ['dining table', 'chair']
        self.pair = {}
        self.target_pos = {}
        self.references_pos = {}

    def is_target(self, label):
        if label in self.targets:
            return True
        else:
            return False

    def is_reference(self, label):
        if label in self.references:
            return True
        else:
            return False

    def is_person(self, label):
        if label == 'person':
            return True
        else:
            return False

    def is_collide(self, box1, box2):
        if (box1[0]>box2[0] and box1[0]<box2[2] and box1[1]>box2[1] and box1[1]<box2[3]):
            return True
        elif (box2[0]>box1[0] and box2[0]<box1[2] and box2[1]>box1[1] and box2[1]<box1[3]):
            return True
        else:
            return False

    def connect_pairs(self):
        for k1,v1 in self.target_pos.items():
            for k2,v2 in self.references_pos.items():
                if (self.is_collide(v1, v2)):
                    self.pair[k1] = k2

def dic_to_txt(path, mode, dic):
    if len(dic) == 0:
        print('{} is empty'.format(dic))
        return False
    with open(path, mode) as f:
            for k,v in dic.items():
                f.write(str(k)+'\n')
                f.write(str(v)+'\n')
    return True

def getdic(filename):
    dic = {}
    with open(filename, 'r') as f:
        line = f.readline()
        dic = json.loads(line)
    return dic

def txt_to_dic(path, mode):
    dic = {}
    with open(path, mode) as f:
        list = f.readlines()
    print(list)
    i=1
    while (i in range(1,len(list))):
        k = list[i-1].strip('\n')
        v = list[i].strip('\n')
        dic[k] = v
        i += 2
    print(dic)
    return dic
