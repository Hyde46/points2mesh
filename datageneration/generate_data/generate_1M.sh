#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "usage: <ids to process>"
    exit 1
fi

#cd /home/groh/git/cgtuebingen/Flex-Convolution-Dev/data_generation
cd /home/heid/Documents/master/pc2mesh/datageneration/generate_data

#source activate gflex
source activate py2
# conda info
for l in "$@"
do
    echo "$l"

    parallel python generate_gt.py --label $l --dir /graphics/scratch/datasets/ShapeNetCorev2/data/100k/ --num_parts 20 --samples 100000 --mode train --part_id ::: {0..19}
    python generate_gt.py --merge --label $l --dir /graphics/scratch/datasets/ShapeNetCorev2/data/100k/ --num_parts 20 --samples 100000 --mode train

    #parallel python generate_gt.py --label $l --dir /graphics/scratch/datasets/ShapeNetCorev2/10 --num_parts 20 --samples 10 --mode test --part_id ::: {0..19}
    #python generate_gt.py --merge --label $l --dir /graphics/scratch/datasets/ShapeNetCorev2/10 --num_parts 20 --samples 10 --mode test
done

# cd /graphics/scratch/datasets/ModelNet40/advanced/1M
# rm -rf *.parts
# rm -rf *-lock
