from toolbox import output_gs_name


gs_name = output_gs_name()

for i in gs_name:
    with open(f"{i}.py", "w") as f_py:
        f_py.write(
            f"""from toolbox import plot_mixed_thining_ksd
plot_mixed_thining_ksd(model_name='{i}')
"""
        )

    with open(f"{i}.sh", "w") as f_sh:
        f_sh.write(
            f"""#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH -t 2-00:00:00

source /mnt/nfs/home/c2029946/Software/miniconda3/bin/activate stein-q-thinning
module load GCC/11.3.0

python {i}.py

echo Finishing job
exit 0
"""
        )

with open("batch_run.sh", "w") as f_batch:
    for j in gs_name:
        f_batch.write(f"sbatch {j}.sh\n")
