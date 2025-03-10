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

intervals = [1, 10, 20]
balances = [0.001, 0.01, 0.1, 1]

for ds, st in source_target.items():
    for src, tar in st:
        for interval in intervals:
            for balance in balances:
                outcsv = 'exp_{}_{}_{}_{}.csv'.format(domain[ds][src], domain[ds][tar], balance, interval)
                job = 'UDA_{}_{}_{}_{}_{}'.format(ds.split('.')[0].split('-')[0], src, tar, balance, interval)
                jobName=job + '.sh'
                outf = open(jobName,'w')
                outf.write('#!/bin/bash\n')
                outf.write('\n')
                outf.write('#SBATCH --partition=wang\n')
                outf.write('#SBATCH --gpus-per-node=rtxa5500:1\n')
                outf.write('#SBATCH --nodes=1 --mem=32G --time=24:00:00\n')
                outf.write('#SBATCH --ntasks=1\n')
                outf.write('#SBATCH --cpus-per-task=8\n')
                outf.write('#SBATCH --output=slurm-%A.%a.out\n')
                outf.write('#SBATCH --error=slurm-%A.%a.err\n')
                outf.write('\n')
                outf.write('module load cuda/12.1\n')
                outf.write('conda info --envs\n')
                outf.write('eval $(conda shell.bash hook)\n')
                outf.write('source ~/miniconda/etc/profile.d/conda.sh\n')
                outf.write('conda activate myenv\n')
                outf.write('python3 /home/zwa281/UDA/UDA2/source_free_train_detect_joint.py --config {} --source {} --target {} --balance {} --interval {} \n'.format(ds, src, tar, balance, interval))
                outf.close()
                if not os.path.isfile(outcsv):
                    subff.write('os.system("sbatch %s")\n' % jobName)
subff.close()
