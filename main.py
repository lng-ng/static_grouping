import ast
import copy
import os
import json
import argparse
from datetime import datetime
import numpy as np

from classes.Learner import Learner
from classes.AnonymityEvaluator import AnonymityEvaluator
from classes.Client import StaticGroupingClient
from classes.OverheadEvaluator import OverheadEvaluator
from util import create_schedule, find_group, save_json


def init(conf):
    all_timeseries = []
    clients = []
    round_num = -1

    dataset = conf["input_conf"]["dataset"]
    batch_num = conf["input_conf"]["batch_num"]

    dataset_folder = os.path.join('dataset', dataset, '')
    data_path = os.path.join(dataset_folder, f'{dataset}_profiles.json')
    batch_path = os.path.join(dataset_folder, f'batchdata_{dataset}_b{conf["input_conf"]["min_batch_size"]}.json')

    with open(batch_path, 'r') as f:
        data = json.load(f)
        user_base_size = data['user_base_size']
        assert data['dataset'] == dataset
        batch = data['batch_data'][batch_num]
        batch_idxs = ast.literal_eval(batch['idx'])
        batch_cids = ast.literal_eval(batch['id'])
        batch_size = len(batch_cids)
        learning_start = batch['learningStart']
        online_start = learning_start + conf["system_conf"]["learn_conf"]["learn_duration"]
        if not batch['valid']:
            print("Batch invalid. Not enough rounds left to start learning phase.")
            return
    with open(data_path, 'r') as f:
        data = json.load(f)
        assert user_base_size == len(data['clientData'])
        objects = data["clientData"]
        client_user_map = {}
        j = 0
        tstmax = -1
        for i, obj in enumerate(objects):
            if len(batch_cids) == 0:
                break
            try:
                idx = batch_idxs.index(i)
            except ValueError:
                continue
            cid = obj["client"]
            start = obj["start"] - 1
            queue = ast.literal_eval(obj['queue'])
            msg_map = ast.literal_eval(obj['msg_map'])
            timeseries = ast.literal_eval(obj["timeSeries"])
            round_num = max(len(timeseries), round_num)
            #total_msgs = len(np.where(queue == 1)[0])
            all_timeseries.append(timeseries)
            total_msgs = len(np.where(np.asarray(queue) == 1)[0])
            client = StaticGroupingClient(j, j, timeseries, msg_map, queue,total_msgs)
            client.remove_old_messages(online_start)
            #print(client.msg_map)
            for tpl in client.msg_map:
                if tpl[0] >= online_start:
                    tstmax = max(tpl[0], tstmax)
                    break
            assert cid == batch_cids[idx]
            assert start < batch['learningStart']
            clients.append(client)
            batch_idxs.pop(idx)
            batch_cids.pop(idx)
            assert obj["client"] not in client_user_map
            client_user_map[obj["client"]] = j
            j += 1
        print(f"\n{len(clients)} users saved in database")
        assert len(clients) == batch_size
    all_timeseries = np.asarray(all_timeseries)
    clients = np.asarray(clients)
    online_clients = ast.literal_eval(data["online_clients"])
    online_lst = []
    for i, lst in enumerate(online_clients):
        online_lst.append([])
        for cid in lst:
            if cid in client_user_map:
                online_lst[i].append(client_user_map[cid])
    assert len(online_lst) == len(online_clients)
    print(f"Batch size: {batch_size}")

    # For logging purposes
    dataset_log = {
        "dataset": conf["input_conf"]["dataset"],
        "msg_threshold": data["msgThreshold"],
        "user_base_size": len(data["clientData"]),
        "batch_threshold": conf["input_conf"]["min_batch_size"],
        "actual_batch_size": batch_size,
        "num_rounds": round_num,
        "cover_msg_ratio": data['ratio'],
        "round_length": data["roundLength"],
    }
    return [clients,
            online_lst,
            learning_start,
            online_start,
            all_timeseries,
            round_num], dataset_log



def main(conf):
    rnd = 30
    np.random.seed(rnd)

    batch_threshold = conf["input_conf"]["min_batch_size"]
    batch_num = conf["input_conf"]["batch_num"]
    dataset = conf["input_conf"]["dataset"]

    current_time = datetime.utcnow().isoformat()
    print(f"Current time: {current_time}")
    print("Initializing..")
    init_result, dataset_log = init(conf)
    print("Initialization finished")
    print("")
    print("Begin clustering..")
    k, t, learning_duration, clusters, cost, learning_matrix = [None] * 6
    if conf['misc']['cluster_cacheread']:
        print(f"Reading cluster data from {conf['misc']['cluster_data_path']}")
        with open(conf["misc"]["cluster_data_path"]) as f:
            cached_data = json.load(f)
            k = cached_data['k']
            t = cached_data['t']
            learning_duration = cached_data['learn_duration']
            clusters = cached_data['clusters']
            cost = cached_data['cost']
            learning_matrix = np.asarray(cached_data['learning_matrix'])
    else:
        k = conf["system_conf"]["learn_conf"]["n_clusters"]
        t = conf["system_conf"]["learn_conf"]["min_cluster_size"]
        learning_duration = conf["system_conf"]["learn_conf"]["learn_duration"]
        learner = Learner(learning_duration=learning_duration)
        clusters, cost, learning_matrix = learner.batch_clustering(num_clusters=k,
                                                                   cluster_size_threshold=t,
                                                                   all_timeseries=init_result[4],
                                                                   learning_start=init_result[2])
    print("Clustering finished")
    if conf["misc"]["cache_clusters"]:
        cache_folder = os.path.join('cache', f'{dataset}_{batch_threshold}',
                                    str(batch_num), current_time.replace(':', '_'), '')
        cluster_data = {
            "k": k,
            "t": t,
            "learn_duration": learning_duration,
            "clusters": clusters,
            "cost": cost,
            "learning_matrix": learning_matrix.tolist(),
        }
        os.makedirs(cache_folder, exist_ok=True)
        cluster_path = os.path.join(cache_folder, f'clusters_k{k}_t{t}.json')
        save_json(cluster_data, cluster_path)
        print(f"Cluster data is cached to {cluster_path}")
    experiment_type = conf["expr_conf"]
    sys_conf = conf["system_conf"]

    # For logging purposes
    learn_log = copy.deepcopy(sys_conf["learn_conf"])
    result_folder = os.path.join('results', f'{dataset}_{batch_threshold}', f'batch{batch_num}',
                                 f'k{k}', current_time.replace(':', '_') , '')
    os.makedirs(result_folder, exist_ok=True)

    print("")
    print("Starting experiments")
    print(f"Experiment type: {experiment_type}")
    if experiment_type == "overhead":
        activity_thresholds = sys_conf["overhead_conf"]["act_thresholds"]
        for activity_threshold in activity_thresholds:
            print(f"Activity threshold: {activity_threshold}")
            schedules = [create_schedule(learning_matrix[cluster], activity_threshold) for cluster in clusters]

            learn_log["activity_threshold"] = activity_threshold
            learn_log["cluster_info"] = [
                (len(cluster), schedule.tolist()) for cluster,schedule in zip(clusters, schedules)
            ]

            online_log = {
                "dyna_sched": sys_conf["overhead_conf"]["dyna_sched"],
                "max_msg_num": sys_conf["overhead_conf"]["max_msg_num"]
            }

            evaluator_conf = {
                'k': k,
                't': t,
                'activity_threshold': activity_threshold,
                'dyna_sched': sys_conf["overhead_conf"]["dyna_sched"],
                'learn_duration': learning_duration,
                'max_msg_num': sys_conf["overhead_conf"]["max_msg_num"],
            }
            evaluator = OverheadEvaluator(init_result, evaluator_conf, clusters=clusters, cluster_schedules=schedules)
            bandwidth, pad, avg_latency, cluster_msg_latency = evaluator.overhead_eval()
            user_group = [find_group(client) for client in init_result[0]]
            result_log = {
                "expr": experiment_type,
                "bandwidth": bandwidth,
                "avg_band_width": np.mean(bandwidth),
                "user_group": user_group,
                "pad": pad,
                "avg_latency": avg_latency,
                "cluster_msg_latency": cluster_msg_latency,
            }
            full_log = {
                "dataset_info": dataset_log,
                "learn_info": learn_log,
                "online_info": online_log,
                "result": result_log,
            }
            log_name = None
            if isinstance(activity_threshold, float):
                log_name = f'overhead_at_p{int(activity_threshold * 100)}.json'
            elif isinstance(activity_threshold, int):
                log_name = f'overhead_at_a{activity_threshold}.json'
            log_path = os.path.join(result_folder, log_name)
            save_json(full_log, log_path)

    elif experiment_type == "anonymity":
        wait_times = sys_conf["anonymity_conf"]["max_waits"]
        crs = sys_conf["anonymity_conf"]["churn_rate"]
        min_offline = sys_conf["anonymity_conf"]["min_offline"]
        max_offline = sys_conf["anonymity_conf"]["max_offline"]
        activity_threshold = sys_conf["anonymity_conf"]["act_threshold"]

        for max_wait in wait_times:
            for churn_rate in crs:
                # Anonymity evaluations
                print(f"Max wait {max_wait}, churn rate {churn_rate}")
                analyzer = AnonymityEvaluator(churn_percentage=churn_rate, online_start=init_result[3], end=init_result[5] - 1)
                analyzer.set_chance_param(max_wait=max_wait,
                                          min_offline=min_offline,
                                          max_offline=max_offline)
                traces = []
                schedules = [create_schedule(learning_matrix[cluster], activity_threshold) for cluster
                             in clusters]

                learn_log["activity_threshold"] = activity_threshold
                learn_log["cluster_info"] = [
                    (len(cluster), schedule.tolist()) for cluster, schedule in zip(clusters, schedules)
                ]

                for cluster, schedule in zip(clusters, schedules):
                    trace = analyzer.analyze(cluster, schedule, chances=max_wait > 0)
                    traces.append(trace.tolist())

                online_log = {
                    "churn_rate": churn_rate,
                    "max_wait": max_wait,
                }
                result_log = {
                    "size_over_time": traces,
                    "avg_size": np.mean(traces, axis=0).tolist()
                }

                full_log = {
                    "expr": experiment_type,
                    "dataset_info": dataset_log,
                    "learn_info": learn_log,
                    "online_info": online_log,
                    "result": result_log,
                }
                log_name = f'anonymity_c{int(churn_rate * 100)}w{max_wait}.json'
                log_path = os.path.join(result_folder, log_name)
                save_json(full_log, log_path)
    print("Experiments finished")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="config file")
    args = parser.parse_args()
    conf_path = args.config
    with open(conf_path) as f:
        conf = json.load(f)
    main(conf)

