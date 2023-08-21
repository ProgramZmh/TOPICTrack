import os
import csv
gate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
dataset = 'BEE'
file_path = 'results/trackers/' + dataset + '-val'

f_test = open(os.path.join(file_path,'bee_test_det_thresh0.0_post','pedestrian_summary.txt'),'r')
csvfile = open('bee_det_thresh.csv',mode='w',newline='')
line_test = f_test.readlines()
fieldnames_test = line_test[0].split(' ')
fieldnames_test.insert(0, 'det_thresh')
# fieldnames_test.insert(0, 'alpha_gate')

write = csv.DictWriter(csvfile,fieldnames=fieldnames_test)
write.writeheader()

for j in gate:
    txt_file = os.path.join(file_path,'bee_test_det_thresh%s_post'%(str(j)),'pedestrian_summary.txt')
    f = open(txt_file,'r')
    line = f.readlines()
    data = line[1].split(' ')
    data.insert(0,j)
    
    dicts = dict(zip(fieldnames_test, data))
    write.writerow(dicts)

