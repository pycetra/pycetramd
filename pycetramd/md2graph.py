import numpy as np
import pandas as pd


def main(text, config={"sep": "\n", "tab": "  "}):
    if "".join(text.split()) == "":
        return gd.GraphData()

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
    node_f = np.zeros(
        size,
        [
            ("id", np.int64),
            ("text", "<U200"),
            ("level", np.int64),
            ("sortkey", np.int64),
            ("x", np.float64),
            ("y", np.float64),
            ("selecter", np.int64),
            ("region", "<U200"),
            ("stylecls", "O"),
            ("position", "O"),
        ],
    )
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
    node_f["stylecls"] = appendStyleCls(node_f["stylecls"], getStyleClsMarkdownpath(df))
    symbol2id_dict = getIdDict(stdsymbol, node_f["id"])

    fromto_list = getFromTo(df, symbol2id_dict)
    size = len(fromto_list)
    link_f = np.zeros(
        size,
        [
            ("id", np.int64),
            ("from", np.int64),
            ("to", np.int64),
            ("x_from", np.float64),
            ("y_from", np.float64),
            ("x_to", np.float64),
            ("y_to", np.float64),
            ("selecter", np.int64),
            ("region", "<U200"),
            ("stylecls", "O"),
        ],
    )
    link_f[["from", "to", "stylecls"]] = fromto_list
    link_f["id"] = [i + 2000 for i in range(size)]

    return (link_f, node_f)


def appendStyleCls(stylecls, newcls):
    result = []
    for n, i in enumerate(stylecls):
        tmp = i
        tmp.append(newcls[n])
        result.append(tmp)
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
