# GA-based dynamic workflow scheduling in fog computing

## Overview

This project is mainly designed to use YAFS simulator to inplement the MAPE dynamic framework and to evaluate our algorithm's performance.

## Porject Structure


```plaintext
├── src/example/DynamicWorkload                  
│   ├── data           
│   ├── results         
│   └── main_hyperparameter.py
│   └── main_experiment_app.py
│   └── main_experiment_lambda.py
│   └── main_experiment_network.py
│   └── readme.md
├── src/yafs                 
│   ├── GA.py             
│   └── GA_static.py   
│   └── ApdatationCost.py
│   └── core.py
└── ...   

```

markdown
Copy code

### src/example/DynamicWorkload

This is where all the source code about the MAPE framework implementation and exoeriment evaluation. 

- `main_hyperparameter`: This is the experiment about GA hyperparameter fine tunning.
- `main_experiment_app.py`: This is the experiment abouut increasing number of applications.
- `main_experiment_lambda.py`: This is the experiment about varying workload.
- `main_experiment_network.py`: This is the experiment about increasing size of network.
- `data`: This file contains the json files which is about the initialization of yafs simulator.
- `results`: This file contains the output of simulator.
- `readme.md`: This file is readme.

### src/yafs

This folder contains the GA implementation and YAFS core class for the project.

- `GA.py`: This is the GA implementation of dyanmic scheduling.
- `GA_static.py`: This is the GA implementation of static scheduling.
- `AdaptationCost.py`: This is a Util class, which calculate the cost of adaptation.
- `core.py`: This is YAFS's core file.
