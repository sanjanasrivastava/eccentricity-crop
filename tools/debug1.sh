#!/bin/bash
for file in log/slurm-13525023*
do
    if [ ! -z $(grep ":)" "$file") ]; then
        :
    else
        echo $file
    fi
done


