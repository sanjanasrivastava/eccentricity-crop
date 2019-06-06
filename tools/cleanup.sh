#!/bin/bash
endmessage='InvalidArgumentError'

for filename in log/slurm-$1*
    do
        if  grep -Fq $endmessage $filename
        then
            jobid=$(echo $filename | cut -c 20- | rev | cut -c 5- | rev)
            echo $jobid
        else
            :
        fi
done 

