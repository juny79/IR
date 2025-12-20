#!/bin/bash
cd /root/IR
python main.py > eval_[7,4,2]_test.log 2>&1
echo "평가 완료"
