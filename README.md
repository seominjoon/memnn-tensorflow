# End-to-end memory networks in TensorFlow
TensorFlow implementation of Facebook's end-to-end memory networks. The original source code is implemented in MatLab (bAbI QA) and Torch (LM).

Paper: http://arxiv.org/abs/1503.08895

Original source code: https://github.com/facebook/MemNN

The paper presents results on two tasks; bAbI question answering and language modeling.
Currently, this implementation runs bAbI QA only.

## Requirements
pip install these packages:

tensorflow, progressbar

## Quick start
To train and test on task 1, run:
```bash
python main.py --task 1
```

For more info and other options, run:
```bash
python main.py --help
```

## Results
On a single run, I got:

Task 1 test error: 0.40%

Task 2 test error: 10.70%

Task 3 test error: 61.90%

Task 4 test error: 18.00%

Task 5 test error: 19.00%

Task 6 test error: 48.30%

Task 7 test error: 25.80%

Task 8 test error: 17.60%

Task 9 test error: 31.00%

Task 10 test error: 37.50%

Task 11 test error: 1.30%

Task 12 test error: 0.50%

Task 13 test error: 0.70%

Task 14 test error: 10.10%

Task 15 test error: 0.00%

Task 16 test error: 52.00%

Task 17 test error: 49.80%

Task 18 test error: 12.00%

Task 19 test error: 91.20%

Task 20 test error: 0.00%

Note that yes/no question task is performing poorly. Please pull request if you find solution to this!

## Implementation details
The implementation does not use random noise (RN); reading the paper only, it wasn't clear to me. 
Also, I haven't tested joint learning, but you can simply run it by modifying the main.py file only.
