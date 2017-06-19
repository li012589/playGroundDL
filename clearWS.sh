#!/bin/bash
if [ $1 == 'qlearning' ]
then
    echo 'clearing qlearning'
    rm -rf ./savedQnetwork/*
    rm -rf ./cartpole_q/*
    rm -rf ./picDir/*
    touch ./savedQnetwork/placeholder
    touch ./picDir/placeholder
elif [ $1 == 'pg' ]
then
    echo 'pg'
elif [ $1 == 'flappybird' ]
then
    echo 'clearing flappybird'
    rm -rf ./saved_networks/*
    touch ./saved_networks/placeholder
else
    echo 'NA'
fi
