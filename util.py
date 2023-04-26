import numpy as np
import json
import jsbeautifier
def save_json(data, data_path):
    options = jsbeautifier.default_options()
    options.indent_size = 4
    with open(data_path, "w") as f:
        res = jsbeautifier.beautify(json.dumps(data), options)
        f.write(res)


def find_group(client):
    if client.total_msgs > 1000:
        return ">1000"
    if client.total_msgs > 100:
        return ">100"
    if client.total_msgs >= 10:
        return ">=10"
    return "<10"


def create_schedule(cluster_timeseries, activity_threshold):
    schedule = None
    if isinstance(activity_threshold, int):
        schedule = np.where(np.sum(cluster_timeseries, axis=0) > 0, activity_threshold, 0)
    elif isinstance(activity_threshold, float):
        schedule = np.where(np.sum(cluster_timeseries, axis=0) > activity_threshold * len(cluster_timeseries), 1, 0)
    assert schedule.size == len(cluster_timeseries[0])
    return schedule


def transform_series(series, learning_duration):
    transformed = [0 for i in range(learning_duration)]
    assert len(transformed) == learning_duration
    for i in range(len(series)):
        transformed_index = i % learning_duration
        transformed[transformed_index] += series[i]
    return transformed


def create_learning_matrix(timeseries_lst, learning_duration):
    learning_matrix = []
    for row in timeseries_lst:
        learning_matrix.append(transform_series(row, learning_duration))
    learning_matrix = np.asarray(learning_matrix, dtype='uint8')
    learning_matrix = learning_matrix > 0
    return learning_matrix.astype(int)