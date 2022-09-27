import networkx as nx
import pandas as pd


def main(link_f, node_f):
    """
    graph構造をmarkdown形式に変換する
    """
    link_f = pd.DataFrame(link_f)
    node_f = pd.DataFrame(node_f)

    # ・閉路を持つgraph構造からtree構造へ変換しておく.
    # というのも、markdownはtree構造だが、symbolic-pathによって閉路を含む構造になっているから.
    # さらにいうと、閉路をtreeにする処理はどのpathを除外するかという複数の選択肢を持つ難しい問-
    # 題であるためmainの最初に実行しておく
    use_index = link_f["stylecls"].apply(lambda x: "markdownpath" not in x)
    link_f = link_f.loc[use_index, :]

    # ・"MarkDownBox"を作るための前処理
    # ・非連結グラフ(=forest)のなかの連結成分(=tree)にカテゴリ(root_category)を振っておく
    # というのも、markdownの章や節というまとまりに分割できるようにしたいから
    link_f_r2 = addLevel(link_f, node_f[["id", "level"]])
    if link_f_r2.shape[0] > 0:
        root_category = getRootCategory(link_f_r2, node_f)
        link_root_category = link_f_r2["from"].apply(lambda x: root_category[x])
        link_f_r2["root_category"] = link_root_category
        link_f_r3 = link_f_r2
    else:
        link_f_r2["root_category"] = ""
        link_f_r3 = link_f_r2

    # markdownを生成する際の処理を最もよく表しているデータ
    MarkDownBox = fusionLinkAndNode(
        link_f_r3[["from", "to", "root_category"]], node_f[["id", "level", "region"]]
    )
    # return MarkDownBox,link_f,node_f
    # グラフ構造をmarkdownの出力形式に合わせなければならない
    # というのも、link_featureのレコードの並び順番は、markdownと同じ並び順番ではない. よって深さ優先探索によって揃える.
    MarkDownBox_sorted = sortByDepthFirstSearchOfEdges(MarkDownBox)

    # return MarkDownBox_sorted, link_f_r3,node_f
    markdown_text = getMarkDownText(MarkDownBox_sorted, node_f)
    return markdown_text


def addLevel(link_f, node_f):
    if link_f.shape[0] > 0:
        link_f_r2 = pd.merge(
            link_f, node_f[["id", "level"]], left_on="from", right_on="id"
        )
        link_f_r2 = link_f_r2[["from", "to", "level"]]
    else:
        link_f_r2 = link_f
    return link_f_r2


def getRootCategory(link_f, node_f):
    root_category = {}
    for i in node_f.to_dict("records"):
        if int(i["level"]) == 0:
            root_category[i["id"]] = i["id"]

    for lv in range(max(link_f["level"]) + 1):
        for i in link_f.to_dict("records"):
            if int(i["level"]) == lv and root_category.get(i["from"]):
                root_category[i["to"]] = root_category[i["from"]]
    return root_category


def fusionLinkAndNode(link_f, node_f):
    new_link_list = []
    for i in node_f[node_f["level"] == 0].to_dict("records"):
        root_id = i["id"]
        index1 = link_f["root_category"] == root_id
        df_init = pd.DataFrame(
            [
                {
                    "from": None,
                    "to": root_id,
                    "root_category": root_id,
                    "region": i["region"],
                }
            ]
        )
        df = pd.concat([df_init, link_f[index1]], axis=0)
        new_link_list.append(df)
    return pd.concat(new_link_list, axis=0)


def sortByDepthFirstSearchOfEdges(link_f):
    result = []
    for i, df in link_f.groupby(["root_category"], sort=False):
        if df.shape[0] > 1:
            df_over2 = df[1:]
            root_node = df["root_category"].values[0]
            edges = df_over2.rename(columns={"from": "source", "to": "target"})
            G = nx.from_pandas_edgelist(edges, create_using=nx.DiGraph())
            # 幅優先探索(nx.bfs_edges)を使わない理由: markdownの上から描画する順番は深さ優先探索と一致するため
            link_list = [tuple(df[["from", "to"]].values[0])] + list(
                nx.edge_dfs(G, source=root_node)
            )
        else:
            link_list = [tuple(df[["from", "to"]].values[0])]
        result.append(link_list)
    result2 = []
    for i in result:
        result2 += i
    sort_index = (
        pd.DataFrame(result2)
        .rename(columns={0: "from", 1: "to"})
        .reset_index()
        .rename(columns={"index": "sort_number"})
    )
    link_f_new = pd.merge(
        link_f, sort_index, left_on=["from", "to"], right_on=["from", "to"]
    ).sort_values(["sort_number"])
    return link_f_new


def getMarkDownText(link_f, node_f):
    node_dict = dict(zip(node_f["id"], node_f.to_dict("records")))

    def printMdText(node_f_record, tab="  ", symbol="- "):
        space = node_f_record["level"] * tab
        text = node_f_record["text"]
        record = space + symbol + text
        return record

    markdown_text = ""
    region_name = ""
    for i in link_f.to_dict("records"):
        if i["from"] == -1 and i["to"] == -1:
            markdown_text += "\n"
            continue
        if region_name != i["region"] and type(i["region"]) == str:
            markdown_text += "\n"
            region_name = i["region"]
        md_str = printMdText(node_dict[i["to"]])
        markdown_text += md_str + "\n"
    return markdown_text
