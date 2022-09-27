#!/bin/bash

dl_scheduer_conf=$2
work_space=$1
work_space_path=./trace/$work_space
log_file=$work_space_path/$work_space.log

cp tmp.log $log_file
cp $dl_scheduer_conf $work_space_path/
python3 -i scheduler_trace_analyzer.py -o $work_space_path -i $log_file