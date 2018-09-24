#!/bin/bash

for k in 128 512
do
	for i in "RNN" "LSTM" 
	do
		for j in 30 20 15 10 5
		do
	    	python train.py --input_length $j --model_type $i --batch_size $k
		done
	done
done
