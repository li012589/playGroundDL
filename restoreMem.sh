#!/bin/bash
if [ -d "./membackup/$1" ]
then
    cp -r ./membackup/$1 ./saved_networks/
else
    echo "No such folder"
fi
