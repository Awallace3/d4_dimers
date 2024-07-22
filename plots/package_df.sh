#!/usr/bin/bash

# This script will ensure a pkl file is split into N files sized less than 49MB to store on GitHub
# Need to get a file argument
if [ -z "$1" ]; then
    echo "Please provide a file to split"
    exit 1
fi

# splits with 49MB size and append _N to the file name 
# get output file base name before extension
base=$(basename $1 .pkl)
split -b 49M -d $1 $base
