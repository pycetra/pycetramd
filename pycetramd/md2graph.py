import numpy as np
import pandas as pd

from pycetramd import datatype as datatype_new


def main(text, config):
    dataset = {
        "node_feature": np.zeros(0, dtype=datatype_new.getDtype("node_feature_np")),
        "link_feature": np.zeros(0, dtype=datatype_new.getDtype("link_feature_np")),
        "layer_feature": np.zeros(0, dtype=datatype_new.getDtype("layer_feature_np")),
        "layer_node_feature": np.zeros(
            0, dtype=datatype_new.getDtype("layer_node_feature_np")
        ),
        "layer_link_feature": np.zeros(
            0, dtype=datatype_new.getDtype("layer_link_feature_np")
        ),
    }
    if "".join(text.split()) == "":
        return dataset

    df = text2dataframe(text, config)
    df = df[df["md"] != ""].reset_index(drop=True)

    df["link_flag"] = df["md"].apply(lambda x: 1 if x.count("@") == 2 else 0)

    df["level"] = df["md"].apply(
        lambda x: len(x.split("-")[0].split(config["tab"])) - 1
    )
    df["id"] = range(1001, 1001 + len(df))
    df["text"] = df["md"].apply(lambda x: x.split("- ")[-1])
    df["text"] = df["text"].apply(
        lambda x: x.replace(" ", "") if x.startswith("#") else x
    )
    df["sortkey"] = getSortKey(df)
    df["region"] = getRegion(df.to_records(index=False))
    node_f_md = df

    size = len(node_f_md)
    node_f = np.zeros(size, datatype_new.getDtype("node_feature_np"))

    node_f["stylecls"] = [[""] for i in range(size)]
    node_f["position"] = [[0] for i in range(size)]
    node_f["id"] = node_f_md["id"]
    node_f["text"] = node_f_md["text"]
    node_f["level"] = node_f_md["level"]
    node_f["sortkey"] = node_f_md["sortkey"]
    node_f["region"] = node_f_md["region"]
    stdsymbol = node_f_md["text"].apply(
        lambda x: x.split("@")[-1] if x.count("@") == 1 else ""
    )
    node_f["symbol"] = stdsymbol
    node_f["stylecls"] = appendStyleCls(
        node_f["stylecls"], getStyleClsPositionparam(df)
    )
    node_f["stylecls"] = appendStyleCls(node_f["stylecls"], getStyleClsMarkdownpath(df))
    symbol2id_dict = getIdDict(stdsymbol, node_f["id"])

    index1 = node_f["region"] == "position"
    index2 = node_f["text"] != "#position"
    target_index = index1 & index2

    position_array = getPositionParam(node_f, target_index)
    position_dict = {}
    for i in position_array:
        if symbol2id_dict.get(i["symbol"]):
            position_dict[symbol2id_dict[i["symbol"]]] = [i["x"], i["y"]]

    node_f["symbol"] = [
        position_array["symbol"][n] if i["symbol"] == "" else i["symbol"]
        for n, i in enumerate(node_f)
    ]
    node_f["position"] = [
        [i["x"], i["y"]] if i["symbol"] != "" else [0] for i in position_array
    ]
    node_f["position"] = [
        position_dict[i["id"]] if position_dict.get(i["id"]) else i["position"]
        for i in node_f
    ]
    stylecls_position = [
        "position"
        if len(i["position"]) == 2 and "position_param" not in i["stylecls"]
        else ""
        for i in node_f
    ]
    node_f["stylecls"] = appendStyleCls(node_f["stylecls"], stylecls_position)
    node_f["stylecls"] = [[j for j in i if j != ""] for i in node_f["stylecls"]]

    # markdownをgraphへ変換する
    fromto_list = getFromTo(df, symbol2id_dict)
    size = len(fromto_list)
    link_f = np.zeros(size, datatype_new.getDtype("link_feature_np"))
    link_f["stylecls"] = [[""] for i in range(size)]
    link_f[["from", "to", "stylecls"]] = fromto_list
    link_f["id"] = [i + 2000 for i in range(size)]

    dataset["node_feature"] = node_f
    dataset["link_feature"] = link_f

    return dataset


def getPositionParam(node_f, target_index):
    result = np.zeros(
        len(node_f), dtype=[("symbol", "U100"), ("x", np.int64), ("y", np.int64)]
    )
    result[["x", "y"]] = -1
    for n, i in enumerate(node_f):
        if target_index[n]:
            csv_data = i["text"].split(",")
            result[n]["symbol"] = csv_data[0]
            result[n]["x"] = csv_data[1]
            result[n]["y"] = csv_data[2]
    return result


def appendStyleCls(stylecls, newcls):
    result = []
    for n, i in enumerate(stylecls):
        tmp = i
        tmp.append(newcls[n])
        result.append(tmp)
    return result


def getStyleClsPositionparam(df):
    result = []
    for i in df.to_dict("records"):
        rec = ""
        if i["region"].startswith("position"):
            rec = "position_param"
        result.append(rec)
    return result


def getStyleClsMarkdownpath(df):
    result = []
    for i in df.to_dict("records"):
        rec = ""
        if i["text"].count("@") == 2:
            rec = "markdownpath"
        result.append(rec)
    return result


def text2dataframe(text, config):
    result = []
    for i in text.split(config["sep"]):
        result.append(i)
    df = pd.DataFrame({"md": result})
    return df


def getSortKey(df):
    from collections import defaultdict

    node_sortkey = defaultdict(lambda: 0)
    sortkey_list = []
    for i in df.to_dict("records"):
        sortkey_list.append(node_sortkey[i["level"]])
        node_sortkey[i["level"]] += 1
    return sortkey_list


def getFromTo(df, symbol2id_dict):
    node_max = {}
    fromto_list = []
    for i in df.to_dict("records"):
        if i["link_flag"] == 0:
            node_max[i["level"]] = i["id"]
        else:
            parent_node = i["text"].split("->")[0].replace("@", "")
            child_node = i["text"].split("->")[1].replace("@", "")
            if symbol2id_dict.get(parent_node) and symbol2id_dict.get(child_node):
                fromto_list.append(
                    (
                        symbol2id_dict[parent_node],
                        symbol2id_dict[child_node],
                        ["markdownpath"],
                    )
                )

        if i["level"] == 0:
            continue
        elif i["level"] > 0:
            parent_node = node_max[i["level"] - 1]
            fromto_list.append((parent_node, i["id"], [""]))
        else:
            raise KeyError(i["level"])
    return fromto_list


def getIdDict(symbol, id):
    index = symbol != ""
    return dict(zip(symbol[index], id[index]))


def formatted(s):
    return "".join([ch for ch in s if ch.isalnum()])


def formattedSymbol(s):
    return formatted(s)


def formattedRegion(s):
    return formatted(s.replace("#", ""))


def getRegion(node_f):
    result = [""] * len(node_f)
    for n, i in enumerate(node_f):
        if i["text"].startswith("#") and not i["text"].startswith("##"):
            result[n] = formattedRegion(i["text"])
    return result
