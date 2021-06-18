```python
import re
import linecache
import os
import matplotlib.pyplot as plt
import math
import fractions
from fractions import Fraction
#获取剪接位点序列
def get_line(file, nums_line):  #方便获取后续第几行数据
    return linecache.getline(file, nums_line).strip()
donor_seq=[[] for i in range(3000)]  #创建二维数组，即donor_seq
j = 0                              #初始化列表的值
file = os.listdir('C:\\Users\\Alan‘s Lenovo\\Desktop\\Training Set')  #获取训练集路径中的所有文件名
for l in file:
    path = 'C:\\Users\\Alan‘s Lenovo\\Desktop\\Training Set\\' + str(l) #获取每一个txt文件的路径
    with open(path) as training:
        contents = training.read()             #得到文件内容
        position = re.findall(r'(?<=\.\.)\d+',contents) #用正则表达式得到donor位点位置
        for cds in position[:len(position)-1]:
            cds = int(cds)            #将字符串中的数字转换为整数形式
            cds_line = cds // 60 + 3    #获取剪接位点行数
            cds_position = cds%60       #获取剪接位点在那一行的第几位
            current_context = get_line(path, cds_line)
            if cds_position < 3:       #如果位置在某一行的前几个，涉及到上一行的碱基
                donor_seq[j] = get_line(path,cds_line - 1)[-(3-cds_position):] + current_context[:(10-cds_position)]
            elif cds_position > 54:    #如果位置在某一行的后几个，涉及到下一行的碱基
                donor_seq[j] = current_context[(cds_position-3):] + get_line(path,cds_line + 1)[:cds_position-54]
            else:
                donor_seq[j] = current_context[(cds_position-3):(cds_position+6)]
            #print(donor_seq[j])
            j = j + 1
#所有的剪切位点序列储存在donor_seq二维数组中
Anumber = [0,0,0,0,0,0,0,0,0]
Cnumber = [0,0,0,0,0,0,0,0,0]
Tnumber = [0,0,0,0,0,0,0,0,0]
Gnumber = [0,0,0,0,0,0,0,0,0]
for q in range(9):
    for f in range(j):
        if donor_seq[f][q] == 'a':
            Anumber[q] = Anumber[q] + 1
        if donor_seq[f][q] == 'c':
            Cnumber[q] = Cnumber[q] + 1
        if donor_seq[f][q] == 't':
            Tnumber[q] = Tnumber[q] + 1
        if donor_seq[f][q] == 'g':
            Gnumber[q] = Gnumber[q] + 1
print(Anumber)
print(Cnumber)
print(Tnumber)
print(Gnumber)
j
```

    [782, 1394, 224, 0, 0, 1167, 1699, 155, 360]
    [866, 328, 81, 0, 0, 65, 184, 118, 388]
    [284, 341, 198, 0, 2381, 58, 218, 112, 1117]
    [449, 318, 1878, 2381, 0, 1091, 280, 1996, 516]
    




    2381




```python
#利用Weblogo画图：
#! pip install weblogo
#! pip install --upgrade pip
#%time !pip install weblogo
fc = open("C:\\Users\\Alan‘s Lenovo\\Desktop\\donor_seq.txt",'w')
for i in range(j):
    fc.write(donor_seq[i][0:9]+'\n')
fc.close()
```


```python
p_plus1 = [[] for i in range(9)]
for q in range(9):
    A_number1 = 0
    C_number1 = 0
    T_number1 = 0
    G_number1 = 0
    for k in range(j):
        if donor_seq[k][q] == 'a':
            A_number1 = A_number1 + 1
        elif donor_seq[k][q] == 'c':
            C_number1 = C_number1 + 1
        elif donor_seq[k][q] == 't':
            T_number1 = T_number1 + 1
        else:
            G_number1 = G_number1 + 1
    p_plus1[q].append(A_number1/j)
    p_plus1[q].append(C_number1/j)
    p_plus1[q].append(T_number1/j)
    p_plus1[q].append(G_number1/j)
print(p_plus1)
```

    [[0.32843343133137337, 0.3637127257454851, 0.11927761444771104, 0.18857622847543049], [0.5854682906341874, 0.1377572448551029, 0.14321713565728686, 0.13355732885342292], [0.09407811843763125, 0.034019319613607726, 0.08315833683326333, 0.7887442251154977], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.49013019739605207, 0.027299454010919783, 0.024359512809743807, 0.4582108357832843], [0.7135657286854263, 0.07727845443091139, 0.09155816883662327, 0.11759764804703907], [0.06509869802603947, 0.049559008819823606, 0.047039059218815626, 0.8383032339353212], [0.1511969760604788, 0.1629567408651827, 0.46913061738765227, 0.21671566568668627]]
    


```python
import seaborn as sns
import numpy as np
import pandas as pd
a = p_plus1[0:9]
fig, ax = plt.subplots(figsize = (9,9))
#二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
#和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
sns.heatmap(pd.DataFrame(np.round(a,2), columns = ['P(A)','P(C)','P(T)','P(G)'], index = range(-3,6)), 
                annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap="YlGnBu")
#sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, 
#            square=True, cmap="YlGnBu")
ax.set_title("p+ single", fontsize = 18)
ax.set_ylabel("site", fontsize = 18)
ax.set_xlabel("nucleotide", fontsize = 18) #横变成y轴，跟矩阵原始的布局情况是一样的
```




    Text(0.5, 60.0, 'nucleotide')




    
![png](output_3_1.png)
    



```python
#提取剪接位点信息,正样本信息p+
p_plus=[[] for i in range(16)] 
A_number = 0
C_number = 0
T_number = 0
G_number = 0
for k in range(j):
    if donor_seq[k][0] == 'a':
        A_number = A_number + 1
    elif donor_seq[k][0] == 'c':
        C_number = C_number + 1
    elif donor_seq[k][0] == 't':
        T_number = T_number + 1
    else:
        G_number = G_number + 1
print(A_number)
p_plus[0].append(A_number/j)
p_plus[0].append(C_number/j)
p_plus[0].append(T_number/j)
p_plus[0].append(G_number/j)
for i in range(1,9):
    AA_number,AC_number,AT_number,AG_number = 0,0,0,0
    CA_number,CC_number,CT_number,CG_number = 0,0,0,0
    TA_number,TC_number,TT_number,TG_number = 0,0,0,0
    GA_number,GC_number,GT_number,GG_number = 0,0,0,0
    for k in range(j):
        if donor_seq[k][i-1] == 'a':
            if donor_seq[k][i] == 'a':
                AA_number = AA_number + 1
            elif donor_seq[k][i] == 'c':
                AC_number = AC_number + 1
            elif donor_seq[k][i] == 't':
                AT_number = AT_number + 1
            elif donor_seq[k][i] == 'g':
                AG_number = AG_number + 1
        elif donor_seq[k][i-1] == 'c':
            if donor_seq[k][i] == 'a':
                CA_number = CA_number + 1
            elif donor_seq[k][i] == 'c':
                CC_number = CC_number + 1
            elif donor_seq[k][i] == 't':
                CT_number = CT_number + 1
            elif donor_seq[k][i] == 'g':
                CG_number = CG_number + 1
        elif donor_seq[k][i-1] == 't':
            if donor_seq[k][i] == 'a':
                TA_number = TA_number + 1
            elif donor_seq[k][i] == 'c':
                TC_number = TC_number + 1
            elif donor_seq[k][i] == 't':
                TT_number = TT_number + 1
            elif donor_seq[k][i] == 'g':
                TG_number = TG_number + 1
        else:
            if donor_seq[k][i] == 'a':
                GA_number = GA_number + 1
            elif donor_seq[k][i] == 'c':
                GC_number = GC_number + 1
            elif donor_seq[k][i] == 't':
                GT_number = GT_number + 1
            elif donor_seq[k][i] == 'g':
                GG_number = GG_number + 1
    A_number = AA_number + AC_number + AT_number + AG_number
    C_number = CA_number + CC_number + CT_number + CG_number
    T_number = TA_number + TC_number + TT_number + TG_number
    G_number = GA_number + GC_number + GT_number + GG_number
    if A_number != 0:
        p_plus[i].append(AA_number/A_number),p_plus[i].append(AC_number/A_number)
        p_plus[i].append(AT_number/A_number),p_plus[i].append(AG_number/A_number)
    else:
         p_plus[i].append(0),p_plus[i].append(0),p_plus[i].append(0),p_plus[i].append(0)
    if C_number != 0:
        p_plus[i].append(CA_number/C_number),p_plus[i].append(CC_number/C_number)
        p_plus[i].append(CT_number/C_number),p_plus[i].append(CG_number/C_number)
    else:
        p_plus[i].append(0),p_plus[i].append(0),p_plus[i].append(0),p_plus[i].append(0)
    if T_number != 0:
        p_plus[i].append(TA_number/T_number),p_plus[i].append(TC_number/T_number)
        p_plus[i].append(TT_number/T_number),p_plus[i].append(TG_number/T_number)
    else:
        p_plus[i].append(0),p_plus[i].append(0),p_plus[i].append(0),p_plus[i].append(0)
    if G_number != 0:
        p_plus[i].append(GA_number/G_number),p_plus[i].append(GC_number/G_number)
        p_plus[i].append(GT_number/G_number),p_plus[i].append(GG_number/G_number) 
    else:
        p_plus[i].append(0),p_plus[i].append(0),p_plus[i].append(0),p_plus[i].append(0)
for i in range(9):
    print(p_plus[i])
```

    782
    [0.32843343133137337, 0.3637127257454851, 0.11927761444771104, 0.18857622847543049]
    [0.6163682864450127, 0.09718670076726342, 0.13682864450127877, 0.149616368286445, 0.6778290993071594, 0.1235565819861432, 0.1397228637413395, 0.05889145496535797, 0.18309859154929578, 0.24647887323943662, 0.23943661971830985, 0.33098591549295775, 0.6080178173719376, 0.16703786191536749, 0.10022271714922049, 0.12472160356347439]
    [0.07388809182209469, 0.018651362984218076, 0.05308464849354376, 0.8543758967001435, 0.18902439024390244, 0.0701219512195122, 0.21646341463414634, 0.524390243902439, 0.03519061583577713, 0.03225806451612903, 0.08211143695014662, 0.8504398826979472, 0.14779874213836477, 0.0660377358490566, 0.07861635220125786, 0.7075471698113207]
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 1.0, 0.0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0.49013019739605207, 0.027299454010919783, 0.024359512809743807, 0.4582108357832843, 0, 0, 0, 0]
    [0.6461011139674379, 0.0959725792630677, 0.13624678663239073, 0.12167952013710369, 0.676923076923077, 0.03076923076923077, 0.18461538461538463, 0.1076923076923077, 0.3448275862068966, 0.06896551724137931, 0.22413793103448276, 0.3620689655172414, 0.8075160403299725, 0.06049495875343722, 0.031164069660861594, 0.1008249312557287]
    [0.05768098881695115, 0.02707474985285462, 0.0329605650382578, 0.8822836962919365, 0.16304347826086957, 0.22282608695652173, 0.18478260869565216, 0.42934782608695654, 0.045871559633027525, 0.0779816513761468, 0.06422018348623854, 0.8119266055045872, 0.060714285714285714, 0.05, 0.02857142857142857, 0.8607142857142858]
    [0.2129032258064516, 0.13548387096774195, 0.2064516129032258, 0.44516129032258067, 0.2966101694915254, 0.288135593220339, 0.2966101694915254, 0.11864406779661017, 0.13392857142857142, 0.16964285714285715, 0.2857142857142857, 0.4107142857142857, 0.13877755511022044, 0.15731462925851702, 0.5100200400801603, 0.1938877755511022]
    


```python
import seaborn as sns
import numpy as np
import pandas as pd
a = p_plus[1:9]
fig, ax = plt.subplots(figsize = (9,9))
#二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
#和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
sns.heatmap(pd.DataFrame(np.round(a,2), columns = ['P(A,A)', 'P(A,C)', 'P(A,T)','P(A,G)','P(C,A)','P(C,C)','P(C,T)','P(C,G)','P(T,A)','P(T,C)','P(T,T)','P(T,G)','P(G,A)','P(G,C)','P(G,T)','P(G,G)'], index = range(-2,6)), 
                annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap="YlGnBu")
#sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, 
#            square=True, cmap="YlGnBu")
ax.set_title("p+", fontsize = 18)
ax.set_ylabel("site", fontsize = 18)
ax.set_xlabel("nucleotide", fontsize = 18) #横变成y轴，跟矩阵原始的布局情况是一样的
```




    Text(0.5, 204.17999999999998, 'nucleotide')




    
![png](output_5_1.png)
    



```python
#提取负样本p-,选择第一个样本训练集滑动窗口得到随机情况下碱基序列
path = 'C:\\Users\\Alan‘s Lenovo\\Desktop\\Training Set\\AB000381.TXT'
with open(path) as training:
    contents = training.read().replace("\n",'')    #得到文件内容
seqs = re.findall(r'(?<=\))\w+',contents) #用正则表达式得到文本序列
seqs = str(seqs[0])
random_seq=[[] for i in range(40000)]
m=0
for i in range(len(seqs)-8):
    random_seq[m] = seqs[i:i+9]
    m = m + 1
p_minus=[[] for i in range(16)] 
A_number = 0
C_number = 0
T_number = 0
G_number = 0
for k in range(m):
    if random_seq[k][0] == 'a':
        A_number = A_number + 1
    elif random_seq[k][0] == 'c':
        C_number = C_number + 1
    elif random_seq[k][0] == 't':
        T_number = T_number + 1
    else:
        G_number = G_number + 1
print(m)
p_minus[0].append(A_number/m)
p_minus[0].append(C_number/m)
p_minus[0].append(T_number/m)
p_minus[0].append(G_number/m)
for i in range(1,9):
    AA_number,AC_number,AT_number,AG_number = 0,0,0,0
    CA_number,CC_number,CT_number,CG_number = 0,0,0,0
    TA_number,TC_number,TT_number,TG_number = 0,0,0,0
    GA_number,GC_number,GT_number,GG_number = 0,0,0,0
    for k in range(m):
        if random_seq[k][i-1] == 'a':
            if random_seq[k][i] == 'a':
                AA_number = AA_number + 1
            elif random_seq[k][i] == 'c':
                AC_number = AC_number + 1
            elif random_seq[k][i] == 't':
                AT_number = AT_number + 1
            elif random_seq[k][i] == 'g':
                AG_number = AG_number + 1
        elif random_seq[k][i-1] == 'c':
            if random_seq[k][i] == 'a':
                CA_number = CA_number + 1
            elif random_seq[k][i] == 'c':
                CC_number = CC_number + 1
            elif random_seq[k][i] == 't':
                CT_number = CT_number + 1
            elif random_seq[k][i] == 'g':
                CG_number = CG_number + 1
        elif random_seq[k][i-1] == 't':
            if random_seq[k][i] == 'a':
                TA_number = TA_number + 1
            elif random_seq[k][i] == 'c':
                TC_number = TC_number + 1
            elif random_seq[k][i] == 't':
                TT_number = TT_number + 1
            elif random_seq[k][i] == 'g':
                TG_number = TG_number + 1
        else:
            if random_seq[k][i] == 'a':
                GA_number = GA_number + 1
            elif random_seq[k][i] == 'c':
                GC_number = GC_number + 1
            elif random_seq[k][i] == 't':
                GT_number = GT_number + 1
            elif random_seq[k][i] == 'g':
                GG_number = GG_number + 1
    A_number = AA_number + AC_number + AT_number + AG_number
    C_number = CA_number + CC_number + CT_number + CG_number
    T_number = TA_number + TC_number + TT_number + TG_number
    G_number = GA_number + GC_number + GT_number + GG_number
    if A_number != 0:
        p_minus[i].append(AA_number/A_number),p_minus[i].append(AC_number/A_number)
        p_minus[i].append(AT_number/A_number),p_minus[i].append(AG_number/A_number)
    else:
        p_minus[i].append(0),p_minus[i].append(0),p_minus[i].append(0),p_minus[i].append(0)
    if C_number != 0:
        p_minus[i].append(CA_number/C_number),p_minus[i].append(CC_number/C_number)
        p_minus[i].append(CT_number/C_number),p_minus[i].append(CG_number/C_number)
    else:
        p_minus[i].append(0),p_minus[i].append(0),p_minus[i].append(0),p_minus[i].append(0)
    if T_number != 0:
        p_minus[i].append(TA_number/T_number),p_minus[i].append(TC_number/T_number)
        p_minus[i].append(TT_number/T_number),p_minus[i].append(TG_number/T_number)
    else:
        p_minus[i].append(0),p_minus[i].append(0),p_minus[i].append(0),p_minus[i].append(0)
    if G_number != 0:
        p_minus[i].append(GA_number/G_number),p_minus[i].append(GC_number/G_number)
        p_minus[i].append(GT_number/G_number),p_minus[i].append(GG_number/G_number) 
    else:
        p_minus[i].append(0),p_minus[i].append(0),p_minus[i].append(0),p_minus[i].append(0)
for i in range(9):
    print(p_minus[i])
```

    35855
    [0.24548877422953563, 0.24529354343885093, 0.25725840189652766, 0.25195928043508575]
    [0.26756080927483517, 0.21323027960900204, 0.21516253694021367, 0.3040463741759491, 0.3136585920618674, 0.29614466052541794, 0.3166154895939952, 0.07358125781871944, 0.16201844818231145, 0.24199674443841562, 0.2884427563754748, 0.30754205100379817, 0.2434647762516615, 0.23094816127603013, 0.2092379264510412, 0.3163491360212672]
    [0.2675304011819525, 0.2133196954199341, 0.2151380838731674, 0.304011819524946, 0.3136585920618674, 0.29614466052541794, 0.3166154895939952, 0.07358125781871944, 0.16201844818231145, 0.24199674443841562, 0.2884427563754748, 0.30754205100379817, 0.2434917469812784, 0.23086296665558878, 0.20926110557217237, 0.3163841807909605]
    [0.2675304011819525, 0.2133196954199341, 0.2151380838731674, 0.304011819524946, 0.3136585920618674, 0.29614466052541794, 0.31672921642215396, 0.07346753099056068, 0.16201844818231145, 0.24199674443841562, 0.2884427563754748, 0.30754205100379817, 0.2434917469812784, 0.23086296665558878, 0.20926110557217237, 0.3163841807909605]
    [0.2675304011819525, 0.2133196954199341, 0.2151380838731674, 0.304011819524946, 0.3136585920618674, 0.29614466052541794, 0.31672921642215396, 0.07346753099056068, 0.162109375, 0.2419704861111111, 0.2884114583333333, 0.3075086805555556, 0.24351872368712607, 0.23088854420562818, 0.2092842898293818, 0.3163084422778639]
    [0.2675, 0.21329545454545454, 0.21522727272727274, 0.3039772727272727, 0.3136585920618674, 0.29614466052541794, 0.31672921642215396, 0.07346753099056068, 0.162109375, 0.2419704861111111, 0.2884114583333333, 0.3075086805555556, 0.24354570637119113, 0.23080332409972298, 0.20930747922437673, 0.31634349030470915]
    [0.2675, 0.21329545454545454, 0.21522727272727274, 0.3039772727272727, 0.31369426751592355, 0.2960646041856233, 0.31676524112829846, 0.07347588717015469, 0.16209178691548226, 0.24194423348161007, 0.2884886622545297, 0.307475317348378, 0.24354570637119113, 0.23080332409972298, 0.20930747922437673, 0.31634349030470915]
    [0.2675, 0.21329545454545454, 0.21522727272727274, 0.3039772727272727, 0.3137299510863383, 0.29609828233420543, 0.3168012740302582, 0.07337049254919804, 0.162074202646995, 0.24191798654805816, 0.288565849425038, 0.30744196137990887, 0.24354570637119113, 0.23080332409972298, 0.20930747922437673, 0.31634349030470915]
    [0.2675, 0.21329545454545454, 0.21522727272727274, 0.3039772727272727, 0.3137299510863383, 0.29609828233420543, 0.3168012740302582, 0.07337049254919804, 0.16205662219329645, 0.2418917453086018, 0.28864301985030916, 0.3074086126477926, 0.243572695035461, 0.23082890070921985, 0.20933067375886524, 0.3162677304964539]
    


```python
p_plus2 = [[] for i in range(9)]
for q in range(9):
    A_number2 = 0
    C_number2 = 0
    T_number2 = 0
    G_number2 = 0
    for k in range(m):
        if random_seq[k][q] == 'a':
            A_number2 = A_number2 + 1
        elif random_seq[k][q] == 'c':
            C_number2 = C_number2 + 1
        elif random_seq[k][q] == 't':
            T_number2 = T_number2 + 1
        else:
            G_number2 = G_number2 + 1
    p_plus2[q].append(A_number2/m)
    p_plus2[q].append(C_number2/m)
    p_plus2[q].append(T_number2/m)
    p_plus2[q].append(G_number2/m)
print(p_plus2)
```

    [[0.24548877422953563, 0.24529354343885093, 0.25725840189652766, 0.25195928043508575], [0.2455166643424906, 0.24529354343885093, 0.25725840189652766, 0.2519313903221308], [0.2455166643424906, 0.24529354343885093, 0.25725840189652766, 0.2519313903221308], [0.2455166643424906, 0.24529354343885093, 0.25728629200948266, 0.25190350020917585], [0.24554455445544554, 0.24529354343885093, 0.25728629200948266, 0.2518756100962209], [0.24554455445544554, 0.24526565332589598, 0.2573141821224376, 0.2518756100962209], [0.24554455445544554, 0.24523776321294102, 0.25734207223539257, 0.2518756100962209], [0.24554455445544554, 0.24523776321294102, 0.2573699623483475, 0.25184771998326594], [0.24554455445544554, 0.24523776321294102, 0.2573978524613025, 0.251819829870311]]
    


```python
import seaborn as sns
import numpy as np
import pandas as pd
a = p_plus2[0:9]
print(len(a))
fig, ax = plt.subplots(figsize = (9,9))
#二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
#和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
sns.heatmap(pd.DataFrame(np.round(a,4), columns = ['P(A)','P(C)','P(T)','P(G)'], index = range(-3,6)), 
                annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap="YlGnBu")
#sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, 
#            square=True, cmap="YlGnBu")
ax.set_title("p- single", fontsize = 18)
ax.set_ylabel("site", fontsize = 18)
ax.set_xlabel("nucleotide", fontsize = 18) #横变成y轴，跟矩阵原始的布局情况是一样的
```

    9
    




    Text(0.5, 60.0, 'nucleotide')




    
![png](output_8_2.png)
    



```python
a = p_minus[1:9]
fig, ax = plt.subplots(figsize = (9,9))
#二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
#和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
sns.heatmap(pd.DataFrame(np.round(a,2), columns = ['P(A,A)', 'P(A,C)', 'P(A,T)','P(A,G)','P(C,A)','P(C,C)','P(C,T)','P(C,G)','P(T,A)','P(T,C)','P(T,T)','P(T,G)','P(G,A)','P(G,C)','P(G,T)','P(G,G)'], index = range(-2,6)), 
                annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap="YlGnBu")
#sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, 
#            square=True, cmap="YlGnBu")
ax.set_title("p-", fontsize = 18)
ax.set_ylabel("site", fontsize = 18)
ax.set_xlabel("nucleotide", fontsize = 18) #横变成y轴，跟矩阵原始的布局情况是一样的
```




    Text(0.5, 204.17999999999998, 'nucleotide')




    
![png](output_9_1.png)
    



```python
#定义WAM中得分S（X）函数
def score(stri):
    s = []
    score_sub = 0
    if stri[0] == 'a' or stri[0] == 'A':
        s.append(math.log(p_plus[0][0]/p_minus[0][0]))
    elif stri[0] == 'c' or stri[0] == 'C':
        s.append(math.log(p_plus[0][1]/p_minus[0][1]))
    elif stri[0] == 't' or stri[0] == 'T':
        s.append(math.log(p_plus[0][2]/p_minus[0][2]))
    elif stri[0] == 'g' or stri[0] == 'G':
        s.append(math.log(p_plus[0][3]/p_minus[0][3]))
    for i in range(1,9):
        if stri[i-1] == 'a' or stri[i-1] == 'A':
            if stri[i] == 'a' or stri[i] == 'A':
                s.append(math.log((p_plus[i][0]+0.00000001)/p_minus[i][0]))
            elif stri[i] == 'c' or stri[i] == 'C':
                s.append(math.log((p_plus[i][1]+0.00000001)/p_minus[i][1]))
            elif stri[i] == 't' or stri[i] == 'T':
                s.append(math.log((p_plus[i][2]+0.00000001)/p_minus[i][2]))
            elif stri[i] == 'g' or stri[i] == 'G':
                s.append(math.log((p_plus[i][3]+0.00000001)/p_minus[i][3]))
        elif stri[i-1] == 'c' or stri[i-1] == 'C':
            if stri[i] == 'a' or stri[i] == 'A':
                s.append(math.log((p_plus[i][4]+0.00000001)/p_minus[i][4]))
            elif stri[i] == 'c' or stri[i] == 'C':
                s.append(math.log((p_plus[i][5]+0.00000001)/p_minus[i][5]))
            elif stri[i] == 't'or  stri[i] == 'T':
                s.append(math.log((p_plus[i][6]+0.00000001)/p_minus[i][6]))
            elif stri[i] == 'g'or stri[i] == 'G':
                s.append(math.log((p_plus[i][7]+0.00000001)/p_minus[i][7]))
        elif stri[i-1] == 't' or stri[i-1] == 'T':
            if stri[i] == 'a'or stri[i] == 'A':
                s.append(math.log((p_plus[i][8]+0.00000001)/p_minus[i][8]))
            elif stri[i] == 'c'or stri[i] == 'C':
                s.append(math.log((p_plus[i][9]+0.00000001)/p_minus[i][9]))
            elif stri[i] == 't'or stri[i] == 'T':
                s.append(math.log((p_plus[i][10]+0.00000001)/p_minus[i][10]))
            elif stri[i] == 'g' or stri[i] == 'G':
                s.append(math.log((p_plus[i][11]+0.00000001)/p_minus[i][11]))
        elif stri[i-1] == 'g' or stri[i-1] == 'G':
            if stri[i] == 'a' or stri[i] == 'A':
                s.append(math.log((p_plus[i][12]+0.00000001)/p_minus[i][12]))
            elif stri[i] == 'c' or stri[i] == 'C':
                s.append(math.log((p_plus[i][13]+0.00000001)/p_minus[i][13]))
            elif stri[i] == 't' or stri[i] == 'T':
                s.append(math.log((p_plus[i][14]+0.00000001)/p_minus[i][14]))
            elif stri[i] == 'g' or stri[i] == 'G':
                s.append(math.log((p_plus[i][15]+0.00000001)/p_minus[i][15]))
    for i in range(9):
        score_sub = score_sub + s[i]
    return score_sub
```


```python
#根据测试集打分
test_seq=[[] for i in range(3000)]  #创建二维数组，即donor_seq
jt = 0                              #初始化列表的值
file1 = os.listdir('C:\\Users\\Alan‘s Lenovo\\Desktop\\Testing Set')  #获取训练集路径中的所有文件名
for l in file1:
    path1 = 'C:\\Users\\Alan‘s Lenovo\\Desktop\\Testing Set\\' + str(l) #获取每一个txt文件的路径
    with open(path1) as testing:
        contents1 = testing.read()             #得到文件内容
        position1 = re.findall(r'(?<=\.\.)\d+',contents1) #用正则表达式得到donor位点位置
        for cds in position1[:len(position1)-1]:
            cds = int(cds)            #将字符串中的数字转换为整数形式
            cds_line = cds // 60 + 3    #获取剪接位点行数
            cds_position = cds%60       #获取剪接位点在那一行的第几位
            current_context1 = get_line(path1,cds_line)
            if cds_position < 3:       #如果位置在某一行的前几个，涉及到上一行的碱基
                test_seq[jt] = get_line(path1,cds_line - 1)[-(3-cds_position):] + current_context1[:(10-cds_position)]
            elif cds_position > 54:    #如果位置在某一行的后几个，涉及到下一行的碱基
                test_seq[jt] = current_context1[(cds_position-3):] + get_line(path1,cds_line + 1)[:cds_position-54]
            else:
                test_seq[jt] = current_context1[(cds_position-3):(cds_position+6)]
            jt = jt + 1
#for i in range(jt):
   # print(test_seq[i])
Anumber1 = [0,0,0,0,0,0,0,0,0]
Cnumber1 = [0,0,0,0,0,0,0,0,0]
Tnumber1 = [0,0,0,0,0,0,0,0,0]
Gnumber1 = [0,0,0,0,0,0,0,0,0]
for q in range(9):
    for f in range(jt):
        if test_seq[f][q] == 'A' or test_seq[f][q] == 'a':
            Anumber1[q] = Anumber1[q] + 1
        if test_seq[f][q] == 'C' or test_seq[f][q] == 'c':
            Cnumber1[q] = Cnumber1[q] + 1
        if test_seq[f][q] == 'T' or test_seq[f][q] == 't':
            Tnumber1[q] = Tnumber1[q] + 1
        if test_seq[f][q] == 'G' or test_seq[f][q] == 'g':
            Gnumber1[q] = Gnumber1[q] + 1
print(Anumber1)
print(Cnumber1)
print(Tnumber1)
print(Gnumber1)
```

    [724, 1230, 162, 0, 0, 1063, 1480, 135, 320]
    [725, 261, 49, 0, 0, 60, 167, 106, 323]
    [231, 297, 165, 0, 2079, 52, 182, 96, 1074]
    [399, 291, 1703, 2079, 0, 904, 250, 1742, 362]
    


```python
true_positive = 0
false_positive = 0
for i in range(jt):
    if score(test_seq[i]) > 1.5:
        true_positive += 1
    else:
        false_positive +=1
print(true_positive,false_positive)
print(test_seq[0],score(test_seq[0]))
```

    2068 11
    GGGGTGAGC 4.578795657688521
    


```python
fc = open("C:\\Users\\Alan‘s Lenovo\\Desktop\\donor_seq_testing1.txt",'w')
fc.write("seq\tscore\n")
for i in range(jt):
    fc.write("{}\t{}\n".format(test_seq[i][:9],score(test_seq[i][:9])))
fc.close()
```


```python
import numpy as np
donor_test_score = np.loadtxt("C:\\Users\\Alan‘s Lenovo\\Desktop\\donor_seq_testing1.txt", delimiter="\t", skiprows=1,
                                                        usecols=1)
donor_test_score
```




    array([4.57879566, 5.19741346, 6.65543173, ..., 4.86996933, 5.55699527,
           5.8527594 ])




```python
import numpy as np
non_donor_test_score = np.loadtxt("C:\\Users\\Alan‘s Lenovo\\Desktop\\non_donor_seq_testing1.txt", delimiter="\t", skiprows=1,
                                                        usecols=1)
non_donor_test_score.size
```




    2883625




```python
import numpy as np
non_donor_test_GC_score = np.loadtxt("C:\\Users\\Alan‘s Lenovo\\Desktop\\non_donor_seq_GC_testing1.txt", delimiter="\t", skiprows=1,
                                                        usecols=1)
non_donor_test_GC_score.size
```




    149206




```python
#计算数据集1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
y_lable = []
for i in range(2079):
    y_lable.append(1)
for i in range(2883625):
    y_lable.append(0)
y_score = np.hstack((donor_test_score, non_donor_test_score))
fpr,tpr,threshold = roc_curve(y_lable, y_score) ###计算真正率和假正率
precision, recall, thresholds = precision_recall_curve(y_lable, y_score)
```


```python
#计算数据集2
y_lable2 = []
for i in range(2079):
    y_lable2.append(1)
for i in range(149206):
    y_lable2.append(0)
y_score2 = np.hstack((donor_test_score, non_donor_test_GC_score))
fpr2,tpr2,threshold2 = roc_curve(y_lable2, y_score2) ###计算真正率和假正率
precision2, recall2, thresholds2 = precision_recall_curve(y_lable2, y_score2)
```


```python
roc_auc = auc(fpr, tpr)
roc_auc2 = auc(fpr2, tpr2)
plt.plot(fpr, tpr, 'k--', label='dataset1 (area = {0:.2f})'.format(roc_auc), lw=2,color = "darkorange")
plt.plot(fpr2, tpr2, 'k--', label='dataset2 (area = {0:.2f})'.format(roc_auc2), lw=2)
plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
```


    
![png](output_19_0.png)
    



```python
import mpl_toolkits.axes_grid1.inset_locator as isl
pr, ax = plt.subplots(1, 1)
ax.plot(recall,precision, '--', label="dataset1",color = "darkorange")
ax.plot(recall2,precision2, '--', label="dataset2")
ax.set_title("PR picture")
ax.set_xlabel("recall")
ax.set_ylabel("precision")
ax.legend(loc="best")
plt.show()
```


    
![png](output_20_0.png)
    



```python
import mpl_toolkits.axes_grid1.inset_locator as isl
pr, ax = plt.subplots(1, 1)
ax.plot(recall,precision, '--', label="dataset1",color = "darkorange")
ax.plot(recall2,precision2, '--', label="dataset2")
ax.set_title("PR picture")
ax.set_xlabel("recall")
ax.set_ylabel("precision")
ax.legend(loc="best")
axins = isl.inset_axes(ax, width="30%", height="40%", loc=10)
axins.plot(recall,precision, '-.')
axins.plot(recall2,precision2, '--')
axins.set_xlim(-0.03, 0.05)
axins.set_ylim(0.4, 0.6)
isl.mark_inset(ax, axins, loc1=2, loc2=4, ls='--')
plt.show()
```


    
![png](output_21_0.png)
    



```python
F1_score = 2 * recall[:-1] * precision[:-1] / (recall[:-1]+precision[:-1])
pr, ax = plt.subplots(1, 1)
ax.plot(thresholds,precision[:-1],'--', label="precision")
ax.plot(thresholds,recall[:-1],'--', label="recall")
ax.plot(thresholds,F1_score, '--', label="F1-score")
y1_max=np.argmax(F1_score)
show_max='['+str(thresholds[y1_max].round(2))+' '+str(F1_score[y1_max].round(2))+']'
ax.plot(thresholds[y1_max],F1_score[y1_max],'ko')
ax.annotate(show_max,xy=(thresholds[y1_max],F1_score[y1_max]),xytext=(thresholds[y1_max],F1_score[y1_max]))
ax.set_xlabel("thresholds")
ax.set_ylabel("lable")
ax.legend(loc="best")
plt.show()
```


    
![png](output_22_0.png)
    



```python
F1_score2 = 2 * recall2[:-1] * precision2[:-1] / (recall2[:-1]+precision2[:-1])
pr, ax = plt.subplots(1, 1)
ax.plot(thresholds2,precision2[:-1],'--', label="precision")
ax.plot(thresholds2,recall2[:-1],'--', label="recall")
ax.plot(thresholds2,F1_score2, '--', label="F1-score")
y1_max2=np.argmax(F1_score2)
show_max2='['+str(thresholds2[y1_max2].round(2))+' '+str(F1_score2[y1_max2].round(2))+']'
ax.plot(thresholds2[y1_max2],F1_score2[y1_max2],'ko')
ax.annotate(show_max2,xy=(thresholds2[y1_max2],F1_score2[y1_max2]),xytext=(thresholds2[y1_max2],F1_score2[y1_max2]))
ax.set_xlabel("thresholds")
ax.set_ylabel("lable")
ax.legend(loc="best")
plt.show()
```


    
![png](output_23_0.png)
    

