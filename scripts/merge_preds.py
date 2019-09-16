import numpy as np, os, pickle

submissions_input = open('sample_submission.csv')
submissions_output = open('sample_submits_READY.csv', 'w+')
submissions_output.write('id,scalar_coupling_constant\n')
numbondsdict = {'1JHC':1, '1JHN':1, '2JHH':2,
          '2JHN':2, '2JHC':2, '3JHH':3,
          '3JHC':3, '3JHN':3}

duds = open('no_pred_observed.csv', 'w+')

ids, preds = [], []
for key in numbondsdict:
    with open('%s_preds.csv'%key) as current_preds:
        for line in current_preds:
            ids.append(line.split(',')[0])
            preds.append(line.strip().split(',')[1])

submissions_input.readline()
submissions_master_ids = []
for line in submissions_input:
    submissions_master_ids.append(line.split(',')[0])

preds = [x for _,x in sorted(zip(ids, preds))]
ids = sorted(ids)
for i in range(0,len(submissions_master_ids)):
    if submissions_master_ids[i] != ids[i]:
        duds.write(submissions_master_ids[i] + '\n')
        ids.insert(i,submissions_master_ids[i])
        preds.insert(i,'5')
    if i % 100000 == 0:
        print(i)


for i in range(0, len(ids)):
    _ = submissions_output.write(''.join([ids[i], ',',
                                          preds[i], '\n']))

submissions_output.close()
submissions_input.close()
duds.close()
