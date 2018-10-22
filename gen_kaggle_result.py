def gen_kaggle_result(result_list, filepath):

  d = {
    "PER": [],
    "LOC": [],
    "ORG": [],
    "MISC":[],
  }

  last_tag = "O"
  start = 0

  cnt = 0
  for ind, tag in enumerate(result_list):
    tag = tag[2:] if len(tag) > 1 else tag

    if last_tag == "O" and tag != "O":
      start = ind
    elif last_tag != "O" and tag == "O":
      d[last_tag].append(f"{start}-{ind-1}")

    last_tag = tag

  result = '''Type,Prediction\n'''
  for tag, l in d.items():
    result += f"{tag},{' '.join(l)}" + "\n"

  with open(filepath, "w") as fout:
    fout.write(result)

  return result
