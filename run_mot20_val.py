import os
import csv
dataset = 'MOT20'
file_path = 'results/trackers/' + dataset + '-val/'+'mot20_val_7_1'
examp_file = '2023-07-01_12-25-21+best_paper_ablations_alpha0.0-gate0.0_post'
listFiles = os.listdir(file_path)

# f_test = open(os.path.join(file_path,examp_file,'pedestrian_summary.txt'),'r')
# csvfile = open('bee.csv',mode='w',newline='')
# line_test = f_test.readlines()
# fieldnames_test = line_test[0].split(' ')
# fieldnames_test.insert(0, 'gate')
# fieldnames_test.insert(0, 'alpha_gate')

# write = csv.DictWriter(csvfile,fieldnames=fieldnames_test)
# write.writeheader()

for i in listFiles:
    if 'post' in i:
        

        print(i)