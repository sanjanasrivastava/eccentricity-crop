#!/bin/bash
endmessage=':)'
for filename in log/*
    do
        if  grep -Fxq $endmessage $filename
        then
            :
        else
            echo $filename
        fi
done 
