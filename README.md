# TICC-POMCP

## Prerequisites

You will need to install [`pypy`](https://pypy.org/download.html).

## Instruction for basic execution

To execute TICC-POMCP solver, run
```
pypy3 driver.py <seed>
```

To execute POMCP solver without capability models, run
```
pypy3 'standard POMCP\driver.py' <seed>
```

## Instruction for reproducing experimental results

To reproduce experimental results for TICC-POMCP solver, run
```
bash job_launcher.sh
```

To reproduce experimental results for POMCP solver without capability models, run
```
bash std_job_launcher.sh
```

Redirect `sys.stdout` to write output to log files.

## Experimental setups
### Setup 1 - Varying number of samples
In this setup, the number of shopping lists is fixed at 10 and number of shopping item types is fixed at 5. The number of search samples is varied from 5000 to 50000. Vary `num_iter` in `driver.py` based on number of search samples. 

The shopping lists and the actual capabilities setup are as follows:

_Shopping lists_

Item 1 | Item 2 | Item 3 | Item 4 | Item 5
--- | --- | --- | --- | ---
4 | 3 | 0 | 2 | 3 
1 | 4 | 0 | 7 | 1 
2 | 3 | 2 | 3 | 3 
5 | 4 | 2 | 0 | 2 
0 | 3 | 3 | 4 | 3 
3 | 3 | 0 | 3 | 3 
6 | 3 | 0 | 1 | 2 
2 | 3 | 4 | 1 | 2 
1 | 1 | 2 | 4 | 4 
0 | 3 | 2 | 5 | 2

_Human's capability_

Item 1 | Item 2 | Item 3 | Item 4 | Item 5
--- | --- | --- | --- | ---
0\% | 100\% | 10\% | 0\% | 100\%

_Robot's capability_

Item 1 | Item 2 | Item 3 | Item 4 | Item 5
--- | --- | --- | --- | ---
100\% | 0\% | 100\% | 100\% | 10\% 


### Setup 2 - Varying number of shopping item types
In this setup, the number of shopping lists is fixed at 2 and number of search samples is fixed at 50000. The number of shopping item types is varied from 2 to 5. Vary `reward_space` in `driver.py` accordingly.

The shopping lists and the actual capabilities setup are as follows:

#### 2 item types

_Shopping lists_

Item 1 | Item 2 
--- | --- 
1 | 12
2 | 10

_Human's capability_

Item 1 | Item 2 
--- | --- 
50\% | 100\%

_Robot's capability_

Item 1 | Item 2 
--- | --- 
100\% | 50\%

#### 3 item types

_Shopping lists_

Item 1 | Item 2 | Item 3 
--- | --- | --- 
8 | 5 | 0
2 | 5 | 6

_Human's capability_

Item 1 | Item 2 | Item 3 
--- | --- | --- 
0\% | 100\% | 10\%

_Robot's capability_

Item 1 | Item 2 | Item 3 
--- | --- | --- 
100\% | 0\% | 100\%

#### 4 item types

_Shopping lists_

Item 1 | Item 2 | Item 3 | Item 4
--- | --- | --- | ---
4 | 4 | 2 | 3
3 | 5 | 0 | 5

_Human's capability_

Item 1 | Item 2 | Item 3 | Item 4
--- | --- | --- | --- 
0\% | 100\% | 10\% | 100\%

_Robot's capability_

Item 1 | Item 2 | Item 3 | Item 4
--- | --- | --- | --- 
100\% | 0\% | 100\% | 10\%

#### 5 item types

_Shopping lists_

Item 1 | Item 2 | Item 3 | Item 4 | Item 5
--- | --- | --- | --- | ---
2 | 3 | 2 | 3 | 3
5 | 4 | 2 | 0 | 2

_Human's capability_

Item 1 | Item 2 | Item 3 | Item 4 | Item 5
--- | --- | --- | --- | ---
0\% | 100\% | 10\% | 0\% | 100\% 

_Robot's capability_

Item 1 | Item 2 | Item 3 | Item 4 | Item 5
--- | --- | --- | --- | ---
100\% | 0\% | 100\% | 100\% | 10\%

### Setup 3 - Varying number of shopping lists
In this setup, the number of shopping items is fixed at 5 and number of search samples is fixed at 50000. The number of shopping lists is varied from 5 to 10. 

For an n-shopping list setup, use the first n shopping lists used for Setup 1. Vary `reward_space` in `driver.py` accordingly. The actual capability setup is also the same as Setup 1. 