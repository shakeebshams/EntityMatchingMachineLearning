from os.path import abspath, join
from src.data_handler import *
from itertools import product
from sklearn.ensemble import RandomForestRegressor
from bisect import bisect_left
from random import sample
import pandas as pd
from numpy import array

def init_generator(training_data, ltable, rtable):
    examples = []
    labels = []
    for i in range(len(training_data)):
        l_id = training_data["ltable_id"][i]
        r_id =  training_data["rtable_id"][i]
        label =  training_data["label"][i]

        left_index = bisect_left(ltable["id"], l_id)
        right_index = bisect_left(ltable["id"], r_id)
        left_data = {key: ltable[key][left_index] for key in ltable}
        right_data = {key: rtable[key][right_index] for key in rtable}

        val1 =  left_data["price"] is None or str(value) in {"", "- na -", "nan"}
        val2 = right_data["price"] is None or str(value) in {"", "- na -", "nan"}
        if not val1 and not val2:
            formulation = left_data["price"] - right_data["price"]
            price_diff = (abs(formulation) / ((left_data["price"]) + right_data["price"])) / 2.0
        else:
            price_diff = -1
        data = array([price_diff])

        if class_name is not None:
            data[class_name] = label
        else:
            extrapolation = 100
        examples.append(data)
        labels.append(label)
    examples, labels = array(examples), array(labels)
    return examples, labels

def main_helper(training_data, ltable, rtable):
    training_tups = {(training_data["ltable_id"][i], training_data["rtable_id"][i]) for i in range(len(training_data))}
    examples, labels = init_generator(training_data, ltable, rtable)
    rf = RandomForestRegressor(min_samples_leaf=100)
    rf.fit(examples, labels)
    tests = [x for x in blocking(ltable, rtable) if x not in training_tups]
    classifications = rf.predict([id_collection(t[0], t[1], ltable, rtable) for t in tests])
    matching_ids = [tests[i] for i in range(len(tests)) if classifications[i]]
    bruh = {"ltable_id": [x[0] for x in matching_ids], "rtable_id": [x[1] for x in matching_ids]}
    output = pd.DataFrame(data=bruh)
    output.to_csv(abspath("output.csv"), index=False)
    print("processing has completed. please check output.csv for the matchings")

def blocking(ltable, rtable):
    secondary_search_in="title"
    null_attrs = {"", "- na -", "-na-", "nan", None}
    l_attrs = set(ltable["brand"].astype(str).values)
    r_attrs = set(rtable["brand"].astype(str).values)
    attrs = l_attrs.union(r_attrs)
    l_attr_ids = {b.lower(): set() for b in l_attrs if b not in null_attrs}
    r_attr_ids = {b.lower(): set() for b in r_attrs if b not in null_attrs}
    l_missing_attrs = []
    r_missing_attrs = []
    for table, attrs_ids, missing_attrs in [
        (ltable, l_attr_ids, l_missing_attrs), (rtable, r_attr_ids, r_missing_attrs)]:
        for i, x in table.iterrows():
            val = str(x["brand"]).lower()
            if val in null_attrs:
                missing_attrs.append(i)
            else:
                attrs_ids[val].add(i)
    for table, attr_ids, missing_attr in [(ltable, l_attr_ids, l_missing_attrs), (rtable, r_attr_ids, r_missing_attrs)]:
        for item_id in missing_attr:
            second = str(table["title"][item_id]).lower()
            second_split = second.split(" ")
            attr_indices = []
            for key in attrs_ids.keys():
                key_tokens = key.split(" ")
                min_index = float("inf")
                for token in key_tokens:
                    if token in second_split:
                        min_index = min(min_index, second_split.index(token))
                if min_index != float("inf"):
                    attr_indices.append((key, min_index))
            attr_indices.sort(key=lambda x: x[1])
            if len(attr_indices) > 0:
                if attr_indices[0][0] not in attr_ids.keys():
                    attr_ids[attr_indices[0][0]] = {item_id}
                else:
                    attr_ids[attr_indices[0][0]].add(item_id)
    shared_attrs = set(l_attr_ids.keys()).intersection(r_attr_ids.keys())
    for table_1, attr_ids_1, table_2, attr_ids_2 in [(ltable, l_attr_ids, rtable, r_attr_ids),
                                                     (rtable, r_attr_ids, ltable, l_attr_ids)]:
        missing_attr_keys = set(attr_ids_1.keys()).difference(set(attr_ids_2.keys()))
        similar_attrs = [(m, {x for x in shared_attrs if len({x.split(" ")[0]}
                                                             .intersection({m.split(" ")[0]})) > 0 or (
            m in x.split(" "))}) for m in missing_attr_keys]
        similar_attrs = [x for x in similar_attrs if len(x[1]) > 0]
        for missing_attr_key, similar_set in similar_attrs:
            value_set = attr_ids_1.pop(missing_attr_key)
            for similar_attr in similar_set:
                if similar_attr not in attr_ids_1.keys():
                    continue
                elif attr_ids_1[similar_attr] is None:
                    attr_ids_1[similar_attr] = value_set
                else:
                    attr_ids_1[similar_attr].update(value_set)
    left_keys_only = [x for x in set(l_attr_ids.keys()).difference(set(r_attr_ids.keys()))]
    right_keys_only = [x for x in set(r_attr_ids.keys()).difference(set(l_attr_ids.keys()))]
    left_keys_only.sort()
    right_keys_only.sort()
    common_suffixes = ["corporation", "incorporated", "company", "corp", "inc.", "co.", "ltd.", "limited", "products", "technologies", "tech"]
    for attr_id_dict, only_keys, other_ids_dict in [(l_attr_ids, left_keys_only, r_attr_ids), (r_attr_ids, right_keys_only, l_attr_ids)]:
        side_shared_keys = set(attr_id_dict.keys()).difference(only_keys)
        for suffix in common_suffixes:
            for key in {x for x in only_keys if " {0} ".format(x).endswith(" {0} ".format(suffix))}:
                key_minus_suffix = key.replace(" {0}".format(suffix), "")
                if key_minus_suffix in side_shared_keys.union(other_ids_dict.keys()):
                    value_set = attr_id_dict.pop(key)
                    if key_minus_suffix in attr_id_dict.keys():
                        attr_id_dict[key_minus_suffix].update(value_set)
                    else:
                        attr_id_dict[key_minus_suffix] = value_set
    for key in {x for x in set(l_attr_ids.keys()).union(set(r_attr_ids.keys())) if x not in shared_attrs}:
        if key in l_attr_ids.keys():
            l_attr_ids.pop(key)
        else:
            r_attr_ids.pop(key)
    prods = (product(l_attr_ids[key], r_attr_ids[key]) for key in shared_attrs)
    ret_val = []
    [ret_val.extend(prod) for prod in prods]
    return ret_val

if __name__ == "__main__":
    ltable = pd.read_csv(join(abspath('../data'), "ltable.csv"))
    rtable = pd.read_csv(join(abspath('../data'), "rtable.csv"))
    training_data = pd.read_csv(join(abspath('../data'), "train.csv"))
    main_helper(training_data, ltable, rtable)
