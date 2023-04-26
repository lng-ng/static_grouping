import numpy as np


class AnonymityEvaluator:
    __slots__ =  ['churn_percentage','online_start','max_wait','min_offline','max_offline','end']

    def __init__(self, churn_percentage, online_start, end):
        self.churn_percentage = churn_percentage
        self.online_start = online_start
        self.end = end
        self.max_wait = None
        self.min_offline = None
        self.max_offline = None

    def set_chance_param(self, max_wait=None, min_offline=None, max_offline=None):
        if max_wait is not None:
            self.max_wait = max_wait
        if min_offline is not None:
            self.min_offline = min_offline
        if max_offline is not None:
            self.max_offline = max_offline

    def analyze(self, cluster, schedule, chances=False):
        cluster_size = len(cluster)
        cluster_trace = np.asarray([cluster_size for i in range(self.online_start, self.end + 1)])
        required_rounds = np.where(schedule)[0]
        if not len(required_rounds):
            return cluster_trace
        sample = np.random.choice(required_rounds, size=int(self.churn_percentage * cluster_size),
                                  replace=True)
        if not len(sample):
            return cluster_trace
        lower_bounds = np.ceil((self.online_start - sample) / len(schedule)).astype(int)
        upper_bounds = np.floor((self.end - sample) / len(schedule)).astype(int)
        coefficients = np.random.randint(lower_bounds, upper_bounds)
        result = None
        if chances:
            assert self.max_wait is not None
            assert self.min_offline is not None
            assert self.max_offline is not None
            offline_times = np.random.randint(self.min_offline, self.max_offline, len(coefficients))
            a_filter = offline_times > self.max_wait
            result = []
            for val, coeff in zip(sample[a_filter], coefficients[a_filter]):
                pos = np.where(required_rounds == val)[0]
                assert len(pos) == 1
                pos = pos[0]
                new_pos = (pos + self.max_wait) % len(required_rounds)
                coeff_inc = int((pos + self.max_wait) / len(required_rounds))
                result.append(required_rounds[new_pos] + (coeff_inc + coeff) * len(schedule))
            result = np.asarray(result)
        else:
            result = sample + coefficients * len(schedule)
        """
        print("normalized result")
        print((result - self.online_start) % 24)
        """
        for val in result:
            assert schedule[val % len(schedule)], f"{val}, {val % len(schedule)}"
        for r in result:
            if r > self.end:
                continue
            cluster_trace[(r - self.online_start):] -= 1
        return cluster_trace



