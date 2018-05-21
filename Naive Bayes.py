#coding=utf-8
import sys
from matplotlib import pyplot as plt
'''得到类别特征集合,返回一个字典'''
def getFeature(filename):
    f = open(filename)
    feature_dct = {}
    for line in f.readlines():
        cate, words = line.strip().split(',', 1)
        feature_dct[cate] = words.split(' ')
    # print(feature_dct)
    return feature_dct
    f.close()

'''将提取出来的特征写入文件'''
def Navie_Bayes(testname, predictname, featureFile):

    feature = getFeature(featureFile)

    f_test = open(testname)
    f_predict = open(predictname, 'w')
    f_predict.truncate()
    f_predict.write('original_cate' + ',' + 'predict_cate' + '\n')
    for line in f_test.readlines():
        original_cate, words = line.strip().split(',', 1)
        resultdct = {}
        for cate, featurelst in feature.items():
            count = 0
            for word in words.split(' '):
                if word in featurelst:
                    count += 1
                else:
                    pass
            resultdct[cate] = count

        '''根据类别计算结果，选择最大类别'''
        predict = max(resultdct.items(), key=lambda x: x[1])[0]
        f_predict.write(original_cate + ',' + predict + '\n')

    f_predict.close()
    f_test.close()

# Navie_Bayes('./Text/testData_cutword.csv', './Predict_result/df_result.csv', './Feature/df.txt')
# Navie_Bayes('./Text/testData_cutword.csv', './Predict_result/chi_result.csv', './Feature/chi.txt')
# Navie_Bayes('./Text/testData_cutword.csv', './Predict_result/mi_result.csv', './Feature/mi.txt')
# Navie_Bayes('./Text/testData_cutword.csv', './Predict_result/tfidf_result.csv', './Feature/tfidf.txt')

'''根据结果文件计算准确度'''
def cal_Acc(filename):
    cate_dct = {
    '0': 'auto',
    '1': 'business',
    '2': 'cul',
    '3': 'sports',
    '4': 'travel',
    '5': 'yule'}

    f = open(filename)
    f.readline()
    same = 0
    sum = 0
    for line in f.readlines():
        cate1, cate2 = line.strip().split(',', 1)
        if cate_dct[cate2] == cate1:
            same += 1
        sum += 1
    return same * 1.0/sum

# print(cal_Acc('./Predict_result/df_result.csv'))
# print(cal_Acc('./Predict_result/chi_result.csv'))
# print(cal_Acc('./Predict_result/mi_result.csv'))
# print(cal_Acc('./Predict_result/tfidf_result.csv'))

'''根据结果文件宏平均F1值'''
def cal_macro_Average(filename):
    cate_dct = {
        '0': 'auto',
        '1': 'business',
        '2': 'cul',
        '3': 'sports',
        '4': 'travel',
        '5': 'yule'}

    f = open(filename)
    f.readline()
    s = f.readlines()
    percision = {}
    recall = {}
    F_Avg_value = {}
    for cate in cate_dct.keys():
        per_same = 0
        per_sum = 0
        rec_same = 0
        rec_sum = 0
        for line in s:
            cate1, cate2 = line.strip().split(',', 1)
            '''计算每个类别的精确率'''
            if cate2 == cate:
                if cate_dct[cate2] == cate1:
                    per_same += 1
                per_sum += 1

            '''计算每个类别的召回率'''
            if cate1 == cate_dct[cate]:
                if cate_dct[cate2] == cate1:
                    rec_same += 1
                rec_sum += 1
        if per_sum == 0:
            percision[cate] = 0
        else:
            percision[cate] = per_same * 1.0 / per_sum
        if rec_sum == 0:
            recall[cate] = 0
        else:
            recall[cate] = rec_same * 1.0 / rec_sum
    f.close()
    '''计算宏平均F1值'''
    for key in percision.keys():
        F_Avg_value[key] = (2.0 * percision[key] * recall[key]) / (percision[key] + recall[key])

    # print(percision)
    # print(recall)
    return sum(F_Avg_value.values())/6

# print(cal_macro_Average('./Predict_result/df_result.csv'))
# print(cal_macro_Average('./Predict_result/chi_result.csv'))
# print(cal_macro_Average('./Predict_result/mi_result.csv'))
# print(cal_macro_Average('./Predict_result/tfidf_result.csv'))

def pltimage():
    X = [10,20,30,40,50,100,150,200,250,300,400,500,600,700]
    '''取不同特征维度时分类的准确率'''
    df_Acc = [69.50,73.70,78.58,79.44,82.12,85.65,86.17,87.02,87.38,89.01,88.47,88.86,89.51,90.02]
    chi_Acc = [85.91,88.46,89.97,90.24,91.41,90.41,90.72,89.76,90.31,89.73,88.43,87.89,88.13,86.51]
    mi_Acc = [71.73,74.67,82.79,80.80,81.75,85.02,86.39,87.39,88.07,88.33,88.43,89.07,89.50,89.72]
    tfidf_Acc = [71.38,80.09,84.06,85.11,87.47,88.08,88.14,89.35,90.42,90.66,90.20,90.56,91.04,90.89]
    '''取不同维度下分类结果的宏平均值'''
    df_Macro_avg = [52.87,55.42,59.73,63.15,65.62,70.82,71.08,70.56,70.50,71.93,73.51,74.09,75.65,76.96]
    chi_Macro_avg = [72.86,78.55,77.90,77.71,77.74,78.08,77.93,76.58,76.08,74.78,73.76,73.36,73.15,72.01]
    mi_Macro_avg = [55.79,57.74,66.80,65.84,67.64,70.03,71.76,70.71,71.65,73.00,74.16,75.09,75.62,76.52]
    tfidf_Macro_avg = [62.42,65.20,69.58,71.99,72.95,72.84,74.55,74.64,75.43,76.71,76.10,77.55,77.53,77.72]

    plt.figure()
    plt.title('Acc Result')
    plt.plot(X, df_Acc, 'ro-',color='green', label='tf_accuracy')
    plt.plot(X, chi_Acc, 'ro-', color='red', label='chi_accuracy')
    plt.plot(X, mi_Acc, 'ro-', color='skyblue', label='mi_accuracy')
    plt.plot(X, tfidf_Acc, 'ro-', color='blue', label='tfidf_accuracy')
    plt.legend()  # 显示图例
    plt.xlabel('Features Dimension')
    plt.ylabel('Accuracy (%)')
    plt.show()

    plt.figure()
    plt.title('Macro_avg Result')
    plt.plot(X, df_Macro_avg, 'ro-', color='green', label='tf_macro_ave')
    plt.plot(X, chi_Macro_avg, 'ro-', color='red', label='chi_macro_ave')
    plt.plot(X, mi_Macro_avg, 'ro-', color='skyblue', label='mi_macro_ave')
    plt.plot(X, tfidf_Macro_avg, 'ro-', color='blue', label='tfidf_macro_ave')
    plt.legend()  # 显示图例
    plt.xlabel('Features Dimension')
    plt.ylabel('Macro_avg Value (%)')
    plt.show()

# pltimage()