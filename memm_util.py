import numpy as np


__bio_cat2ind__ = {
  'MISC'  : 0,
  'LOC'   : 1,
  'ORG'   : 2,
  'PER'   : 3,
}
__bio_tag2val__ = {
  "B": 2,
  "I": 1,
}
BIO_VEC_LEN = len(__bio_cat2ind__)


def bio2vec(bio):
  result = np.zeros((BIO_VEC_LEN, ))
  if bio == "O":
    return result
  else:
    tag, cat = bio.split("-")
  result[__bio_cat2ind__[cat]] = __bio_tag2val__[tag]
  return result

if __name__ == "__main__":
  import data_reader
  import random

  reader = data_reader.DataReader()
  reader.read_file()
  ret = reader.split_train_valid()
  train = ret["train"]
  valid = ret["valid"]

  selected_ind = random.sample(range(len(train)), 3)
  for i in selected_ind:
    sample = train[i]
    bios = sample[2]
    for bio in bios.split():
      print(f"{bio}, {bio2vec(bio)}")

  ## simply use "x.split()" is enough to split tokens/poss/bios
  # for i, sample in enumerate(train):
  #   if i % 100 == 0:
  #     print(i)
  #   tok_cnt = len(sample[0].split())
  #   pos_cnt = len(sample[1].split())
  #   bio_cnt = len(sample[2].split())
  #   if tok_cnt != pos_cnt or pos_cnt != bio_cnt or bio_cnt != tok_cnt:
  #     print(f"tok: {tok_cnt}, pos: {pos_cnt}, bio: {bio_cnt}, {sample}")

  ## all non "O" tags has exactly ONE "-"
  # for i, sample in enumerate(train):
  #   bio = sample[2]
  #   for tag in bio.split():
  #     if tag != "O" and tag.count("-") != 1:
  #       print(f"{tag.count('-')}, {sample}, {tag}")

  ## all non "O" tags has subtag in {'MISC', 'LOC', 'ORG', 'PER'}
  # bio_set = set()
  # for i, sample in enumerate(train):
  #   bio = sample[2]
  #   for tag in bio.split():
  #     if tag != "O":
  #       bio_set.add(tag.split("-")[1])
  # print(bio_set)