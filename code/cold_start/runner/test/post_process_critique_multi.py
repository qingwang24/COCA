import json
import os
import numpy as np
from tqdm import tqdm
output_dir = '/train21/cog8/permanent/qkchang/shliu19/critique/experiments/gen_critique_revision/qwen2_vl_7b_t1/mathvista/'

files = os.listdir(output_dir)
# records = []
from multiprocessing import Pool, cpu_count
def process_file(file):
    file_path = os.path.join(output_dir, file)
    records=[]
    with open(file_path, 'r', encoding='utf-8') as fi:
        lines = fi.readlines()
        record=''
        flag=False
        for line in lines:
            if record=='' and line=='{\n':                
                flag = True
            if line=='}\n':
                flag = False
            # if record!='' and '{\n' == line.replace(' ',''):
            #     record=''
            # else:
            #     record+=line
            record+=line
            if not flag:
                records.append(json.loads(record))
                record = ''
    return records

files = [f for f in files if 'rank' in f]

records_all = []
with Pool(20) as pool:
    for raw_output in tqdm(pool.imap(process_file, files), total=len(files)):
        records_all += raw_output
# for file in tqdm(files):
#     records_all+=process_file(file)
print(len(records_all))

with open(os.path.join(output_dir, files[0]).split('_rank')[0]+'.json', 'w', encoding='utf-8') as fi:
    json.dump(records_all, fi, ensure_ascii=False)
