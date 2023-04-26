from classes.Learner import Learner
from util import create_schedule
from tqdm import tqdm
import numpy as np
import copy


class OverheadEvaluator():
    __slots__ = ['learning_start','online_start', 'round_num', 'clients', 'online_lst', 'all_timeseries',
                 'conf',
                 'clusters','cluster_schedules','schedule_recorder']

    def __init__(self, init_result, conf, clusters, cluster_schedules):
        self.clients = copy.deepcopy(init_result[0])
        self.online_lst = copy.deepcopy(init_result[1])
        self.learning_start = init_result[2]
        self.online_start = init_result[3]
        self.all_timeseries = copy.deepcopy(init_result[4])
        self.round_num = init_result[5]

        self.conf = conf

        self.clusters = clusters
        self.cluster_schedules = cluster_schedules
        self.schedule_recorder = []


    def start_online_phase_nochurn(self, packet_limit=1, scheduling_user_threshold=0):
        assert self.clusters is not None
        assert self.clients is not None
        self.schedule_recorder = []
        dynamic_scheduling = self.conf['dyna_sched']
        learning_duration = self.conf['learn_duration']
        max_msg_num = self.conf['max_msg_num']
        # Initialization
        for current_round in tqdm(range(self.online_start, self.round_num)):
            # Put new messages into client queues
            if current_round < self.round_num:
                online_clients = self.online_lst[current_round]
                for cid in online_clients:
                    client = self.clients[cid]
                    assert client.timeseries[current_round]
                    assert client.msg_map[0][0] == current_round, f"{client.msg_map}"
                    client.update_msgs_available()

            # For logging purposes
            new_schedule_lst = []
            change_arr = []
            cluster_sizes = []

            for i in range (len(self.clusters)):
                cluster = self.clusters[i]

                # Update schedules every 24 rounds
                if (current_round - self.online_start) % learning_duration == 0 and dynamic_scheduling:
                    learning_data = self.all_timeseries[cluster]
                    learning_data = learning_data[:,current_round-learning_duration:current_round]
                    assert learning_data.shape[1] == learning_duration, f"{learning_data.shape}"
                    new_schedule = create_schedule(learning_data, scheduling_user_threshold)
                    new_schedule_lst.append(new_schedule)
                    #schedule_diff = np.count_nonzero(new_schedule == self.cluster_schedules[i])
                    #pred = schedule_diff >= 5 and np.any(new_schedule)
                    pred = np.count_nonzero(new_schedule) >= 2
                    change_arr.append(pred)
                    if pred:
                        self.cluster_schedules[i] = new_schedule

                # Clusters send out messages
                # Cluster does not send if the at current round the schedule is 0
                cluster_schedule = self.cluster_schedules[i]
                if cluster_schedule[current_round % len(cluster_schedule)]:
                    for cid in cluster:
                        client = self.clients[cid]
                        assert client.num_msgs_available >= 0
                        num_packet = 0
                        while num_packet == 0 or (
                                num_packet < packet_limit and client.num_msgs_available):
                            num_msgs = min(max_msg_num, client.num_msgs_available)
                            packet = client.generate_packet(num_msgs, max_msg_num, current_round)
                            for round_generated in packet[1:]:
                                if round_generated != -1:
                                    send_latency = packet[0] - round_generated
                                    publish_latency = current_round + 1 - packet[0]
                                    client.record_message_latency(send_latency, publish_latency)
                            num_packet += 1
                cluster_sizes.append(len(self.clusters[i]))

            if (current_round - self.online_start) % learning_duration == 0 and dynamic_scheduling:
                self.schedule_recorder.append(
                    {
                        "table" : new_schedule_lst,
                        "round" : current_round,
                        "change": np.asarray(change_arr, dtype='uint32').tolist(),
                    }
                )


    def overhead_eval(self):
        activity_threshold = self.conf['activity_threshold']
        self.start_online_phase_nochurn(packet_limit=1,scheduling_user_threshold=activity_threshold)
        return self.get_metrics()


    def get_metrics(self):
        cluster_msg_latency, pad, avg_latency, bandwidth = [], [], [], []

        # Calculate avg msg latency per cluster
        for cluster, schedule in zip(self.clusters, self.cluster_schedules):
            cluster_total_latency = 0
            num_msgs = 0
            for cid in cluster:
                cluster_total_latency += self.clients[cid].total_latency
                num_msgs += self.clients[cid].num_published_real_msgs
            cluster_msg_latency.append(np.divide(cluster_total_latency, num_msgs))

        for client in self.clients:
            bandwidth.append(client.num_dummy_blocks)
            avg_latency.append(np.divide(client.total_latency, client.num_published_real_msgs))
            pad.append(client.num_pad_messages)

        return bandwidth, pad, avg_latency, cluster_msg_latency
