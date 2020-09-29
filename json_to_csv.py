import json
import pandas as pd
import sys

inp_file = sys.argv[1]
op_file = sys.argv[2]

row_example = {
    "level": "",
    "ancestors": "",
    "topic_index": "",
    "topic_word_indicator": "",
    "phrases": ""
}
final_data = []
with open(inp_file) as f:
    data = json.load(f)
    for parent in data:
        for ancestor in data[parent]:
            for topic_index in data[parent][ancestor]:
                for topic_word in data[parent][ancestor][topic_index]:
                    temp = row_example.copy()
                    temp["level"] = parent
                    temp["ancestors"] = ancestor
                    temp["topic_index"] = topic_index
                    temp["topic_word_indicator"] = topic_word
                    # print(data[parent][ancestor])
                    temp["phrases"] = data[parent][ancestor][topic_index][topic_word]
                    final_data.append(temp)

finalest = []
for row in final_data:
    vals = list(row.values())
    # print(vals)
    temp = vals[-1]
    vals = vals[:-1]
    vals.extend(temp)
    finalest.append(vals)


# print(finalest[0])
df = pd.DataFrame(finalest)
df.to_csv(op_file)

