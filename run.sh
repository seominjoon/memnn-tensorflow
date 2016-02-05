#!/bin/bash
for i in `seq 1 20`; do
  python main.py --task $i --eval_period 20 --progress False
done
