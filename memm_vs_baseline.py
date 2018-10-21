import data_reader
import memm_prob
import baseline

timestamp = "1021122311"
valid_ind_path = '''data_ind_split/simple_valid_portion_0.1_seed_1021122311.pickle'''
'''feature = length'''

reader = data_reader.DataReader()
reader.read_file()
ret = reader.split_train_valid_by_valid_ind(valid_ind_path)

train = ret["train"]
valid = ret["valid"]

memm = memm_prob.MEMMProb(None)
memm.train(train, max_iter=10, length=True, timestamp=timestamp)
memm_perf = memm.valid_performance(valid)

base = baseline.Baseline()
base.train(train)
base_perf = base.valid_performance(valid)

print(memm_perf, base_perf)