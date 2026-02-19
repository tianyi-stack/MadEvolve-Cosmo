#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=10:00:00
#SBATCH --job-name StdRec
#SBATCH --output=HaloStdRec_output_%j.txt
#SBATCH --mail-type=FAIL

export OMP_NUM_THREADS=8
source $HOME/ENV/nbodykit/bin/activate

Para='fiducial'
for ((i=340;i<400;i++))
do
    Simu_Start=$(($i*5))
    for ((j=0;j<1;j++))
    do
        cd /scratch/p/pen/zangsh/Quijote_Simulations/MLR
#         DenPATH='/scratch/p/pen/zangsh/Quijote_Simulations/Density/'$Para'/'
        (python -u RecCal_multi.py $Para $(($Simu_Start+0)) $j) &
        (python -u RecCal_multi.py $Para $(($Simu_Start+1)) $j) &
        (python -u RecCal_multi.py $Para $(($Simu_Start+2)) $j) &
        (python -u RecCal_multi.py $Para $(($Simu_Start+3)) $j) &
        (python -u RecCal_multi.py $Para $(($Simu_Start+4)) $j) &
        wait
        
    done
done
