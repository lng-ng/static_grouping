import ast
import json
import argparse
import os

from tqdm import tqdm

clients = []
round_num = -1
round_length = -1

def batch_generation_init(data_path):
    global clients, round_num, round_length
    with open(data_path, "r") as f:
        print(f"Data used: {f.name}")
        data = json.load(f)
        objects = data["clientData"]
        round_num = data["round_num"]
        round_length = data["roundLength"]
    for j, obj in enumerate(objects):
        client = {  # definition of a user
            "id": obj['client'],
            "start": int(obj["start"]) - 1,  # round in which user joins the system
            "grouped": False,
            "learningStart": -1,
            "idx": j,
        }
        clients.append(client)
        del obj


def generate_batch_data(dataset_name, batch_size):
    global clients, round_num, round_length
    assert len(clients)
    assert round_num != -1
    assert round_length != -1
    batches = []
    for i in tqdm(range(0, round_num)):
        lst = list(filter(lambda client: (client["start"] <= i and client["learningStart"] == -1), clients))
        if len(lst) < batch_size:
            continue
        else:
            batch_ids = list(map(lambda c: c['id'], lst))
            batch_idx = list(map(lambda c: c['idx'], lst))
            batches.append({
                'learningStart': i + 1, # begin learning at NEXT round, after enough users accumulated
                'id': str(batch_ids),
                'idx': str(batch_idx),
                'valid': True if round_num - (i+1) + 1 >= 24 else False # Indicates whether there is enough rounds left in the communication for learning phase
            })
            for c in lst:
                c["learningStart"] = i + 1
    output = {
        'dataset': dataset_name,
        'user_base_size': len(clients),
        'batch_threshold': batch_size,
        'num_batches': len(batches),
        'batch_data': batches,
    }
    return output



def test_batch_data(batchdata_path):
    global clients
    print(f"Testing batch data at {batchdata_path}..")
    with open(batchdata_path, 'r') as f:
        data = json.load(f)
        batch_threshold = data["batch_threshold"]
        num_batches = data["num_batches"]
        batch_data = data["batch_data"]
        print(f"Batch threshold: {batch_threshold}")
        print(f"Total number of batches: {num_batches}")
        for batch in batch_data:
            uids = ast.literal_eval(batch["id"])
            idxs = ast.literal_eval(batch["idx"])
            assert len(uids) == len(idxs)
            print(f"Batch size: {len(uids)}")
            print(f"Learning start at: {batch['learningStart']}")
            assert len(uids) > batch_threshold


def main(args):
    clients = []
    print("Initializing batch generation")
    print(f"Dataset: {args.dataset}")
    data_path = os.path.join('dataset', args.dataset, f'{args.dataset}_profiles.json')
    batch_generation_init(data_path)
    for client in clients:
        client['learningStart'] = -1
    batch_size = args.size
    print(f"Generating batch data with size {batch_size}..")
    result = generate_batch_data(args.dataset, batch_size)
    output_path = os.path.join('dataset',args.dataset, f'batchdata_{args.dataset}_b{batch_size}.json' )
    with open(output_path, 'w') as f:
        json.dump(result, f)
    print("Finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="dataset name")
    parser.add_argument("-s", "--size", type=int, default=5000, help="Batch size")
    args = parser.parse_args()
    main(args)
