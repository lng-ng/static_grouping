import ast
import json
import argparse
import os

from tqdm import tqdm
import numpy as np


dump_intermediate = False


# 1000 - 1351745295 - 1354337697
# 10000 - 1351742468 - 1354337954
# 100000 - 1351742400 - 1354337979
# 1000000 - 1351742400 - 1354337999
# Reddit - 1632896852 - 1635488851
def getTimestamps(dataset):
    """
    Returns the earliest timestamp and the latest timestamp in the dataset

    Dataset must be SORTED in ascending order beforehand.
    """
    with open(dataset, 'r') as f:
        endTimestamp = -1
        startTimestamp = float('inf')
        print("First pass")

        for i, line in enumerate(tqdm(f)):
            data = ast.literal_eval(line)
            timestamp = data['timestamp']
            if i == 0:
                startTimestamp = min(timestamp, startTimestamp)
            endTimestamp = max(timestamp, endTimestamp)
    return startTimestamp, endTimestamp

def augment(dataset, ratio, end_timestamp):
    users = {}
    with open(dataset, 'r') as f:
        print("Second pass")
        for line in tqdm(f):
            data = ast.literal_eval(line)
            timestamp = data['timestamp']
            user_id = data['user_id']
            if user_id not in users:
                users[user_id] = {} # keys should already be sorted
            user = users[user_id]
            if timestamp not in user:
                user[timestamp] = []
            for i in range (0,ratio):
                # Generate a random timestamp larger than the initial timestamp
                rand_timestamp = np.random.randint(timestamp + 1, end_timestamp + 2)
                assert rand_timestamp > timestamp
                user[timestamp].append(rand_timestamp)
    output = {
        'user_timestamps': users,
        'cover_msg_ratio': ratio,
    }
    # Dump the timestamps data
    if dump_intermediate:
        with open(f'timestamps_{len(users.keys())}_r{ratio}.json', 'w') as f:
            json.dump(output, f)
    print(f"Number of users found: {len(users.keys())}")
    return output

def create_profiles(inp, start_time, msg_threshold, round_length, dataset_name):
    round_num = -1
    clients = []
    data = inp['user_timestamps']
    ratio = inp['cover_msg_ratio']
    print("Generating user profiles")
    for i, user in enumerate(tqdm(data.keys())):
        timestamps = data[user]
        if len(timestamps.keys()) < msg_threshold:
            continue
        queue = []
        for real_msg in timestamps.keys():
            queue.append((int(real_msg), 1))
        for cover_msgs in timestamps.values():
            for cover_msg in cover_msgs:
                queue.append((int(cover_msg), 0))
        queue.sort(key=lambda tp: tp[0])
        round_map = list(map(lambda tp: int((tp[0] - start_time) / round_length), queue))
        round_num = max(round_map[-1], round_num)
        round_sent = {}
        for ele in round_map:
            if ele in round_sent:
                round_sent[ele] += 1
            else:
                round_sent[ele] = 1
        binaryqueue = list(map(lambda tp: tp[1], queue))
        compressed = list(np.packbits(np.asarray(binaryqueue)))
        client = {
            'start': round_map[0] + 1,  # startRound starts at 1
            'client': user,
            'rounds_sent': round_sent,
            'queue': str(binaryqueue),
            'compressed_queue': str(compressed),
            'total_msgs': len(binaryqueue),
            'msg_map': str([(k, v) for k, v in round_sent.items()])
        }
        clients.append(client)
    #ridx = np.random.choice(len(clients), 1000, replace=False)
    #tmp = []
    #for i in range(len(clients)):
     #   if i in ridx:
      #      tmp.append(clients[i])
   #clients = tmp
    online_lst = [[] for i in range(0, round_num + 1)]
    print("Creating online_lst")
    for client in tqdm(clients):
        round_sent = client['rounds_sent']
        ts = np.zeros(round_num + 1, dtype='uint8')
        ts[list(round_sent.keys())] = np.asarray(list(round_sent.values()))
        client['timeSeries'] = str(np.where(ts > 0,1,0).tolist())
        for cround in round_sent.keys():
            lst = online_lst[cround]
            assert client['client'] not in lst
            lst.append(client['client'])
        client.pop('rounds_sent')
    output = {
        'dataset': dataset_name,
        'clientData': clients,
        'roundLength': round_length,
        'startTime': start_time,
        'msgThreshold': msg_threshold,
        'online_clients': str(online_lst),
        'round_num': round_num + 1,
        'ratio': ratio,
    }
    result_folder = os.path.join('dataset', dataset_name, '')
    os.makedirs(result_folder, exist_ok=True)

    with open(os.path.join(result_folder, f'{dataset_name}_profiles.json'), 'w') as f:
        json.dump(output, f)
    print(f"Dataset name: {dataset_name}")
    print(f"Number of rounds: {round_num + 1}")
    print(f"Number of unfiltered users: {len(data.keys())}")
    print(f"Each user has to send more than {msg_threshold} real messages")
    print(f"Number of filtered users: {len(clients)}")
    print(f"For each real message {ratio} cover messages are generated")


if __name__ == '__main__':
    msg_threshold = -1
    ratio = -1

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices={'twitter','reddit'}, help="dataset name to create the folder for storing files")
    parser.add_argument("input", help="dataset file path")
    parser.add_argument("-t", "--threshold", type=int, default=10, help="Total number of messages each user have to at least send")
    parser.add_argument("-r", "--cover", type=int, default=0, help="Ratio of real:cover")
    parser.add_argument("-l", "--rlength", type=int, default=3600, help="Length of a round in seconds")
    args = parser.parse_args()
    path = args.input

    print("Parser started")
    startTimestamp, endTimestamp = getTimestamps(path)
    print(startTimestamp, endTimestamp)
    intermediate_output = augment(path, args.cover, endTimestamp)
    create_profiles(intermediate_output, startTimestamp, args.threshold, args.rlength, args.dataset)
    print("Parser is done")







