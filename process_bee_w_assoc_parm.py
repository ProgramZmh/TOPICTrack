import os
import csv
gate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
dataset = 'BEE'
file_path = 'results/trackers/' + dataset + '-val' +'/w_assoc_parm'
examp_file = '2023-07-02_02-42-04+best_paper_ablations_alpha0.5-gate0.3-w_assoc_emb0.0_post'
listFiles = os.listdir(file_path)

f_test = open(os.path.join(file_path,examp_file,'pedestrian_summary.txt'),'r')
csvfile = open('bee1.csv',mode='w',newline='')
line_test = f_test.readlines()
fieldnames_test = line_test[0].split(' ')
fieldnames_test.insert(0, 'gate')
fieldnames_test.insert(0, 'alpha_gate')
fieldnames_test.insert(0, 'w_assoc_parm')
write = csv.DictWriter(csvfile,fieldnames=fieldnames_test)
write.writeheader()

for i in listFiles:
    if 'post' in i:
        print(i)
        alpha_gate = i[46:49]
        gate = i[54:57]
        w_assoc_parm = i[69:72]
        txt_file = os.path.join(file_path,i,'pedestrian_summary.txt')
        f = open(txt_file,'r')
        line = f.readlines()
        data = line[1].split(' ')
        data.insert(0,gate)
        data.insert(0,alpha_gate)
        data.insert(0,w_assoc_parm)
        print(data)
        dicts = dict(zip(fieldnames_test, data))
        write.writerow(dicts)
        print(alpha_gate,gate)


