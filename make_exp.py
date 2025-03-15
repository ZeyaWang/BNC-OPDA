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


domain = {'visda': ['train', 'validation']}

source_target = {
    'visda': [[0,1]]
}
outline = []
intervals = [1]
#balances = [0.001, 0.01, 0.1, 1.0]
#balances = [0.001, 0.01]
lambdavs = [0.0]
balances = [0.001, 0.01]
lr_scales = [0.1]
max_k = 100
lrs = [0.001, 0.0005, 0.0001]
KKs = [5, 30]
covs = [0.01, 0.001]
#scs = ['cos', 'entropy']
scs = ['cos']
clf = [False] # [False, True]
for ds, st in source_target.items():
    for src, tar in st:
        for interval in intervals:
            for balance in balances:
                for lambdav in lambdavs:
                    for lr in lrs:
                        for lr_scale in lr_scales:
                            for KK in KKs:
                                for cov in covs:
                                    for sc in scs:
                                        for cl in clf:
                                            cmd = ('python /home/zwa281/UDA/BNC-OPDA/source_free.py --dataset {} --source {} --target {} --balance {} --lr {} '
                                                   '--lr_scale {} --interval {} --lambdav {} --max_k {} --KK {} --covariance_prior {} --score {} ').format(ds, src, tar, balance, lr, lr_scale, interval, lambdav, max_k, KK, cov, sc)
                                            if cl:
                                                cmd += '--classifier \n'
                                            else:
                                                cmd += '\n'
                                            outcsv = 'exp_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(domain[ds][src], domain[ds][tar], balance, lr, lr_scale, interval, lambdav, max_k, KK, cov, sc, cl)
                                            if not os.path.isfile(outcsv):
                                                outline.append(cmd)
                                            else:
                                                print('======{} exists======'.format(outcsv))
nn = 5# 7
split_lists = [[] for _ in range(nn)]
for i, element in enumerate(outline):
    split_lists[i % nn].append(element)

for ii in range(nn):
    job = 'DA_{}'.format(ii)
    jobName=job + '.sh'
    outf = open(jobName,'w')
    outf.write('#!/bin/bash\n')
    outf.write('\n')
    outf.write('#SBATCH --partition=wang\n')
    outf.write('#SBATCH --gpus-per-node=rtxa5500:1\n')
    outf.write('#SBATCH --nodes=1 --mem=32G --time=168:00:00\n')
    outf.write('#SBATCH --ntasks=1\n')
    outf.write('#SBATCH --cpus-per-task=6\n')
    outf.write('#SBATCH --output=slurm-%A.%a.out\n')
    outf.write('#SBATCH --error=slurm-%A.%a.err\n')
    outf.write('\n')
    outf.write('module load cuda/11.7\n')
    outf.write('conda info --envs\n')
    outf.write('eval $(conda shell.bash hook)\n')
    outf.write('source ~/miniconda/etc/profile.d/conda.sh\n')
    outf.write('conda activate myenvs\n')
    for l in split_lists[ii]:
        outf.write(l)
    outf.close()
    subff.write('os.system("sbatch %s")\n' % jobName)
subff.close()
