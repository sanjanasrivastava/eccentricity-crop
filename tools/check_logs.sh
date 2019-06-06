#!/bin/bash
endmessage=':)'
nanmessage='InvalidArgumentError'
for filename in log/slurm-$1*
    do
        if grep -Fxq $endmessage $filename 
        then
            :
        # elif grep -Fq $nanmessage $filename
        # then
        #     :
        else
            echo $filename
            # vim $filename
        fi
done 
