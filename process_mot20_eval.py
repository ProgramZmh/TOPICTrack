import os
import csv
dataset = 'MOT20'
file_path = 'results/trackers/' + dataset + '-val/' + 'mot20_val_7_1'
examp_file = '2023-07-01_12-25-21+best_paper_ablations_alpha0.0-gate0.0_post'
listFiles = os.listdir(file_path)

f_test = open(os.path.join(file_path,examp_file,'pedestrian_summary.txt'),'r')
csvfile = open('mot20_test.csv',mode='w',newline='')
line_test = f_test.readlines()
fieldnames_test = line_test[0].split(' ')
fieldnames_test.insert(0, 'gate')
fieldnames_test.insert(0, 'alpha_gate')

write = csv.DictWriter(csvfile,fieldnames=fieldnames_test)
write.writeheader()

for i in listFiles:
    if 'post' in i:
        print(i)
        alpha_gate = i[46:49]
        gate = i[54:57]
        txt_file = os.path.join(file_path,i,'pedestrian_summary.txt')
        f = open(txt_file,'r')
        line = f.readlines()
        data = line[1].split(' ')
        data.insert(0,gate)
        data.insert(0,alpha_gate)
        dicts = dict(zip(fieldnames_test, data))
        write.writerow(dicts)
        print(alpha_gate,gate)
# f_test = open(os.path.join(file_path,'time_now+best_paper_ablations_alpha0.0-gate0.0_post','pedestrian_summary.txt'),'r')
# csvfile = open('test.csv',mode='w',newline='')
# line_test = f_test.readlines()
# fieldnames_test = line_test[0].split(' ')
# fieldnames_test.insert(0, 'gate')
# fieldnames_test.insert(0, 'alpha_gate')

# write = csv.DictWriter(csvfile,fieldnames=fieldnames_test)
# write.writeheader()

# for i in gate:
#     for j in gate:
#         txt_file = os.path.join(file_path,'time_now+best_paper_ablations_alpha%s-gate%s_post'%(str(i),str(j)),'pedestrian_summary.txt')
#         f = open(txt_file,'r')
#         line = f.readlines()
#         data = line[1].split(' ')
#         data.insert(0,j)
#         data.insert(0,i)
#         dicts = dict(zip(fieldnames_test, data))
#         write.writerow(dicts)
#         print(i)

