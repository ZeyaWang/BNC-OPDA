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


domain = {'office-train-config.yaml': ['amazon', 'dslr', 'webcam'],
          'officehome-train-config.yaml': ['Art', 'Clipart', 'Product', 'Real_World']}

source_target = {
    'office-train-config.yaml': [[0,1],[0,2],[1,2],[2,1],[1,0],[2,0]],
    'officehome-train-config.yaml': [[0, 1], [0, 2], [0, 3], [1, 0], [1, 2], [1, 3], [2, 0], [2, 1], [2, 3], [3, 0], [3, 1], [3, 2]]
}

#intervals = [1, 10]
#intervals = [1, 10]
intervals = [1]
#balances = [0.001, 0.01, 0.1, 1.0]
#balances = [0.001, 0.01]
alphas = [0.0]
balances = [0.001, 0.01]
lr_scales = [10.0, 1.0, 0.1]
#alphas = [0.0]

outline = []

for ds, st in source_target.items():
    for src, tar in st:
        for interval in intervals:
            for balance in balances:
                for alpha in alphas:
                    for lr_scale in lr_scales:
                        outcsv = 'exp_{}_{}_{}_{}_{}_{}.csv'.format(domain[ds][src], domain[ds][tar], balance, interval, alpha, lr_scale)
                        if not os.path.isfile(outcsv):
                            outline.append(
                                #'python /home/zwa281/UDA/UDA5/source_free_clean.py --config {} --source {} --target {} --balance {} --lambdav {} --interval {} \n'.format(ds, src, tar, balance, alpha, interval))
                                'python /home/zwa281/UDA/UDA5/source_free_clean.py --config {} --source {} --target {} --balance {} --lr_scale {} --interval {} --lambdav {}\n'.format(ds, src, tar, balance, lr_scale, interval, alpha))

                        else:
                            print('======{} exists======'.format(outcsv))
nn = 6 # 7
split_lists = [[] for _ in range(nn)]
for i, element in enumerate(outline):
    split_lists[i % nn].append(element)

for ii in range(nn):
    job = 'UDA_{}'.format(ii)
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
