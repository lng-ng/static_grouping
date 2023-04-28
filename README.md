# Static Grouping

## Requirements
Python 3.7+  
For the other dependencies, run ``pip install -r requirements.txt``.

## Generating necessary data from datasets
1. Run ``python parser.py [twitter OR reddit] [path to dataset]`` to generate the user profiles.   
For more customization of the arguments, use ``python parser.py -h`` to find out.
2. Run ``python batch_generation.py [twitter OR reddit]`` to generate the batch data.  
Use ``-s [batch_size]`` if you want to change the minimum batch threshold. 
By default, the batch size is 5000 users.

## Running the experiments
All configurations for the experiments are put into a single config file.  
As an example, the config file `test_conf.json` is given.  

Run ``python main.py [path to config file]`` to start the experiments.  
Logs are saved in the `results` folder.

## About the configuration file
* `expr_conf = "overhead" OR "anonymity"` specififies the type of experiment.  
  * `input_conf` contains information about the input for the experiment, namely the profiles
  and batch data generated from the steps before.  
  * Set `dataset` and `min_batch_size` so that the correct user profile and batch data are used.  
  * `batch_num` specifies which batch to use from the batch data.

* `learn_conf` contains the parameters for the learning phase.  
  * `learn_duration`: duration of the learning phase  
  * `n_clusters`: number of clusters for kmodes  
  * `min_cluster_size`: minimum cluster size  

* `overhead_conf` contains the parameters for the online phase, when **evaluating overhead**.  
This means this is only meaningful when `expr_conf = "overhead"`.
  * `max_msg_num`: maximum number of messages in a packet
  * `act_thresholds:`: list of all the activity thresholds. For each threshold, overhead evaluation is conducted
  * `dyna_sched`: enable dynamic scheduling

* `anonymity_conf` contains the parameters for the online phase, when **evaluating anonymity**.  
This means this is only meaningful when `expr_conf = "anonymity"`.
  * `act_threshold`: activity threshold
  * `churn_rate`: list of all the churn rates.
  * `max_waits`: list of all the maximum number of required rounds that the system waits a user for. If max_wait is 0, user is given no chance.
  * For each pair of `(churn_rate, max_waits)`, anonymity evaluation is conducted
  * `(min_offline, max_offline)`: randomization range of the number of rounds each user will be offline for
* `misc`
  * `cache_clusters`: enable the caching of clustering data for the learning phase
  * `cluster_cacheread`: enable reading the clustering data from a cached file so for faster simulation
  * `cluster_data_path`: path to the cached file for reading, if `cluster_cacheread` is enabled