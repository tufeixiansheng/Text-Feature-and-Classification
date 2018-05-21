#coding=utf-8
import os, math
class DataLoader:
    def __init__(self):
        self.datafile = './Text/data.csv'
        self.dataset, self.cate_dict = self.load_data()
        self.catetype_dict = {
            '0': 'auto',
            '1': 'business',
            '2': 'cul',
            '3': 'sports',
            '4': 'travel',
            '5': 'yule'
        }

    '''加载数据集'''
    def load_data(self):
        dataset = []
        cate_dict = {}
        for line in open(self.datafile):
            line = line.strip().split(',')
            cate = line[0]
            if cate not in cate_dict:
                cate_dict[cate] = 1
            else:
                cate_dict[cate] += 1
            dataset.append([line[0], [word for word in line[1].split(' ') if 'nbsp' not in word]])
        return dataset, cate_dict

# f = open('data.txt', 'w+')
# for root, dirs, files in os.walk('data/corpus'):
#     for file in files:
#         file_path = os.path.join(root, file)
#         for line in open(file_path):
#             line = line.strip()
#             if not line:
#                 continue
#             f.write(line + '\n')
# f.close()
