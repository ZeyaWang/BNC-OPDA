import sys
import os
import pickle as pk

def get_files_with_substring_and_suffix(directory, substring, suffix):
    files = []
    all_files = os.listdir(directory)
    files = [file for file in all_files if substring in file and file.endswith(suffix)]
    return files

subff = open('submit.py','w')
subff.write('import os\n')


# domain = {'visda': ['train', 'validation']}


domain = {'office': ['amazon', 'dslr', 'webcam'],
          'officehome': ['Art', 'Clipart', 'Product', 'Real_World']}





# source_target = {
#     'visda': [[0,1]]
# }
source_target = {
    'office': [[0,1],[0,2],[1,2],[2,1],[1,0],[2,0]],
    'officehome': [[0, 1], [0, 2], [0, 3], [1, 0], [1, 2], [1, 3], [2, 0], [2, 1], [2, 3], [3, 0],
                                     [3, 1], [3, 2]]
}

target_type = 'OPDA'
#target_type = 'PDA'
target_type = 'OSDA'
outline = []
intervals = [1]
#intervals = [5]
#balances = [0.001, 0.01, 0.1, 1.0]
#balances = [0.001, 0.01]
lambdavs = [0.0]
alphas = [100.0, 200.0, 400.0]
balances = [0.01]
lr_scales = [0.1]
max_k = 100
#max_k = 3
# lrs = [0.001]#[0.01, 0.001, 0.0005, 0.0001]
# lrs = [0.001, 0.0005]
# KKs = [5,10,50]#[5, 30, 100]
KKs = [5]
covs = [0.001]#[0.01, 0.001]
#scs = ['cos', 'entropy']
scs = ['cos']
#clf = [False, True] # [False, True]
clf = [True]
target_types = ['OPDA']#, 'OSDA']
for target_type in target_types:
    for ds, st in source_target.items():
        if ds == 'office':
            #lrs = [0.01, 0.1]
            lrs = [0.01]
            #lrs = [0.0001]
            #KKs = [5]
        elif ds == 'officehome':
            #lrs = [0.001, 0.0001]
            lrs = [0.0001]
        for src, tar in st:
            for interval in intervals:
                for balance in balances:
                    for alpha in alphas:
                        for lr in lrs:
                            for lr_scale in lr_scales:
                                for KK in KKs:
                                    for cov in covs:
                                        for sc in scs:
                                            for cl in clf:
                                                #CUDA_VISIBLE_DEVICES=2,3,5,6,7
                                                cmd = ('python /home/zwa281/UDA/BNC-OPDA/source_free.py --total_epoch 10 --batch_size 64 --target_type {} --dataset {} --source {} --target {} --balance {} --lr {} '
                                                       '--lr_scale {} --iter_factor {} --alpha {} --max_k {} --KK {} --covariance_prior {} --score {} ').format(target_type, ds, src, tar, balance, lr, lr_scale, interval, alpha, max_k, KK, cov, sc)
                                                if cl:
                                                    cmd += '--classifier \n'
                                                else:
                                                    cmd += '\n'
                                                outcsv = 'exp_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(target_type, domain[ds][src], domain[ds][tar], balance, lr, lr_scale, interval, alpha, max_k, KK, cov, sc, cl)
                                                if not os.path.isfile(outcsv):
                                                    outline.append(cmd)
                                                else:
                                                    print('======{} exists======'.format(outcsv))
nn = 6
split_lists = [[] for _ in range(nn)]
for i, element in enumerate(outline):
    split_lists[i % nn].append(element)
cuda_list = [1,2,3,6,7]
cuda_list = [2,3,4,5,6,7]
#cuda_list = [4,5,6,7]
for ii in range(nn):
    job = 'UDA_{}'.format(ii)
    jobName=job + '.sh'
    outf = open(jobName,'w')
    outf.write('#!/bin/bash\n')
    outf.write('\n')
    outf.write('#SBATCH --partition=wang\n')
    outf.write('#SBATCH --gpus-per-node=rtxa5500:1\n')
    outf.write('#SBATCH --nodes=1 --mem=25G --time=168:00:00\n')
    outf.write('#SBATCH --ntasks=1\n')
    outf.write('#SBATCH --cpus-per-task=5\n')
    outf.write('#SBATCH --output=slurm-%A.%a.out\n')
    outf.write('#SBATCH --error=slurm-%A.%a.err\n')
    outf.write('\n')
    outf.write('module load cuda/11.7\n')
    outf.write('conda info --envs\n')
    outf.write('eval $(conda shell.bash hook)\n')
    outf.write('source ~/miniconda/etc/profile.d/conda.sh\n')
    outf.write('conda activate myenvs\n')
    for l in split_lists[ii]:
        outf.write('CUDA_VISIBLE_DEVICES={} '.format(cuda_list[ii])+l)
    outf.close()
    subff.write('os.system("sbatch %s")\n' % jobName)
subff.close()
