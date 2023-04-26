
import numpy as np


class Client:
    __slots__ = ['cid', 'pseudonym', 'start',
                 'timeseries','client_queue','end','msg_map','total_msgs',
                 'num_msgs_available',
                 'min_latency','max_latency','total_latency',
                 'min_publish_latency', 'max_publish_latency','total_publish_latency',
                 'min_send_latency', 'max_send_latency','total_send_latency',
                 'num_published_real_msgs',
                 'num_pad_messages',
                 'num_dummy_blocks',
                 'message_latency',
                 ]

    def __init__(self, cid, pseudonym, timeseries, msg_map, queue, total_msgs):
        self.cid = cid
        self.pseudonym = pseudonym
        self.total_msgs = total_msgs
        self.end = msg_map[-1][0]
        self.start = msg_map[0][0]
        self.client_queue = queue
        self.timeseries = timeseries
        self.msg_map = msg_map
        self.num_msgs_available = 0
        self.num_pad_messages = 0
        self.num_dummy_blocks = 0

        self.min_send_latency, self.max_send_latency, self.total_send_latency = float('inf'), -1, 0
        self.min_publish_latency, self.max_publish_latency, self.total_publish_latency = float('inf'), -1, 0
        self.min_latency, self.max_latency, self.total_latency = float('inf'), -1, 0
        self.num_published_real_msgs = 0
        self.message_latency = []

    def record_message_latency(self, send_latency, publish_latency):
        assert send_latency >= 0
        assert publish_latency >= 0
        self.num_published_real_msgs += 1
        self.min_send_latency = min(self.min_send_latency, send_latency)
        self.max_send_latency = min(self.max_send_latency, send_latency)
        self.total_send_latency += send_latency

        self.min_publish_latency = min(self.min_publish_latency, publish_latency)
        self.max_publish_latency = min(self.max_publish_latency, publish_latency)
        self.total_publish_latency += publish_latency

        total_latency = send_latency + publish_latency
        self.min_latency = min(self.min_latency, total_latency)
        self.max_latency = max(self.max_latency, total_latency)
        self.total_latency += total_latency

        self.message_latency.append(total_latency)
        assert len(self.message_latency) == self.num_published_real_msgs

    def __str__(self):
        return f"{self.cid},{self.timeseries},{self.msg_map},{self.client_queue}"

    def update_msgs_available(self):
        tpl = self.msg_map.pop(0)
        num_msgs = tpl[1]
        round_generated = tpl[0]
        self.client_queue[self.num_msgs_available:self.num_msgs_available+num_msgs] *= (round_generated + 1)
        self.num_msgs_available += num_msgs

    def generate_packet(self, num_msgs, packet_size, packet_generation_time):
        assert num_msgs <= packet_size
        assert num_msgs <= self.num_msgs_available
        packet = [packet_generation_time + 1]
        packet.extend(self.client_queue[:num_msgs])
        while len(packet) - 1 < packet_size:
            packet.append(-1)
        assert len(packet) - 1 == packet_size
        for msg_generation_time in packet[1:]:
            if msg_generation_time != -1:
                assert msg_generation_time <= packet[0]
        self.client_queue = np.delete(self.client_queue, slice(num_msgs))
        self.num_msgs_available -= num_msgs
        if num_msgs == 0:
            self.num_dummy_blocks += 1
        else:
            self.num_pad_messages += packet_size - num_msgs
        return packet


class StaticGroupingClient(Client):
    def remove_old_messages(self, a_round):
        while len(self.msg_map):
            if self.msg_map[0][0] >= a_round:
                break
            else:
                self.client_queue = np.delete(self.client_queue, slice(self.msg_map[0][1]))
                self.msg_map.pop(0)