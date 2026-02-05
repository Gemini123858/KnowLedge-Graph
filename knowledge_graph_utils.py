# knowledge_graph_utils.py

import networkx as nx
import plotly.graph_objects as go
import plotly.express as px  # 新增导入
from typing import Dict, Any
import json
from data_persistence_utils import get_cache_dir
import os
import logging
import colorsys
import re
import jieba
import jieba.analyse


def generate_colors(n):
    """智能颜色生成：小规模用HSV分布，大规模用预设色板"""
    if n <= 50:
        # 黄金分割分布算法
        HSV_tuples = [(i * 0.618 % 1.0, 0.8, 0.8) for i in range(n)]
        return ['rgb({},{},{})'.format(*map(lambda x: int(x*255), colorsys.hsv_to_rgb(*hsv))) 
                for hsv in HSV_tuples]
    else:
        # 使用Plotly预设色板
        return px.colors.qualitative.Alphabet[:n]
    
def load_graph_from_json(file_hash: str) -> nx.Graph:
    cache_file = os.path.join(get_cache_dir(), f"{file_hash}_graph_data.json")
    print(f"Attempting to load graph data from: {cache_file}")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            G = nx.Graph()

            # 添加节点
            for node in data['nodes']:
                G.add_node(node['id'], type=node['type'], **node['properties'])

            # 修改关系处理部分
            for rel in data['relationships']:
                try:
                    # 直接使用 source 和 target
                    source_id = rel['source'] if isinstance(rel['source'], str) else str(rel['source'])
                    target_id = rel['target'] if isinstance(rel['target'], str) else str(rel['target'])

                    # 如果还是包含 id 标记，则尝试提取
                    if "id='" in source_id:
                        source_id = re.search(r"id='([^']*)'", source_id)
                        source_id = source_id.group(1) if source_id else source_id
                    if "id='" in target_id:
                        target_id = re.search(r"id='([^']*)'", target_id)
                        target_id = target_id.group(1) if target_id else target_id

                    # 添加边
                    G.add_edge(source_id, target_id, label=rel['type'], **rel['properties'])
                except Exception as e:
                    print(f"Warning: Could not add relationship: {rel}. Error: {str(e)}")
                    continue

            print(f"Successfully loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G
        except Exception as e:
            print(f"Error loading graph data: {str(e)}")
            logging.exception("Error details:")
    else:
        print(f"File not found: {cache_file}")
    return None


def find_relevant_subgraph(G: nx.Graph, question: str, max_depth: int = 2) -> nx.Graph:
    # 1. 对问题进行分词
    keywords = jieba.lcut(question)
    key_terms = [word for word in keywords if len(word) >= 2]  # 只保留长度>=2的词

    # 2. 找到相关节点
    relevant_nodes = []
    for node in G.nodes:
        node_text = G.nodes[node].get('text', '').lower()
        if any(term.lower() in node_text for term in key_terms):
            relevant_nodes.append(node)

    # 3. 构建子图
    subgraph = nx.Graph()
    for start_node in relevant_nodes:
        nodes_to_explore = [(start_node, 0)]
        explored = set()

        while nodes_to_explore:
            current_node, depth = nodes_to_explore.pop(0)
            if current_node in explored or depth > max_depth:
                continue

            explored.add(current_node)
            if not subgraph.has_node(current_node):
                subgraph.add_node(current_node, **G.nodes[current_node])

            # 探索相邻节点
            for neighbor in G.neighbors(current_node):
                if neighbor not in explored and depth < max_depth:
                    if not subgraph.has_node(neighbor):
                        subgraph.add_node(neighbor, **G.nodes[neighbor])
                    subgraph.add_edge(current_node, neighbor, **G.edges[current_node, neighbor])
                    nodes_to_explore.append((neighbor, depth + 1))

    # 4. 如果子图太小，扩展搜索
    if subgraph.number_of_nodes() < 3:
        for node in list(subgraph.nodes):
            for neighbor in G.neighbors(node):
                if not subgraph.has_node(neighbor):
                    subgraph.add_node(neighbor, **G.nodes[neighbor])
                subgraph.add_edge(node, neighbor, **G.edges[node, neighbor])

    return subgraph


def prepare_graph_data(graph, cypher_query):
    results = graph.run(cypher_query)

    G = nx.Graph()

    print("Debug: Cypher Query:", cypher_query)

    for i, record in enumerate(results):
        print(f"Debug: Record {i}:", dict(record))

        start_node = record['n']
        end_node = record.get('related')
        relationship = record.get('r')

        start_id = start_node.identity

        # 添加起始节点
        if not G.has_node(start_id):
            G.add_node(start_id,
                       label=list(start_node.labels)[0] if start_node.labels else 'Unknown',
                       title=start_node.get('text', '')[:50])

        # 如果有相关节点和关系，添加它们
        if end_node and relationship:
            end_id = end_node.identity
            if not G.has_node(end_id):
                G.add_node(end_id,
                           label=list(end_node.labels)[0] if end_node.labels else 'Unknown',
                           title=end_node.get('text', '')[:50])
            G.add_edge(start_id, end_id, label=type(relationship).__name__)

    print(f"Debug: Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def generate_colors(n):
    HSV_tuples = [(x * 1.0 / n, 0.7, 0.7) for x in range(n)]
    return ['rgb' + str(tuple(int(x * 255) for x in colorsys.hsv_to_rgb(*hsv))) for hsv in HSV_tuples]


def create_knowledge_graph(G: nx.Graph) -> go.Figure:
    # 使用 Fruchterman-Reingold 布局算法
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    all_types = (
        list(nx.get_node_attributes(G, 'type').values()) +
        list(nx.get_edge_attributes(G, 'label').values())
    )
    unique_types = list(set(all_types))
    colors = generate_colors(len(unique_types))
    color_map = dict(zip(unique_types, colors))

    # 生成节点类型的颜色映射
    # node_types = set(nx.get_node_attributes(G, 'type').values())
    # if not node_types:
    #     # 如果没有节点类型，使用默认颜色
    #     default_color = '#888888'
    #     color_map = {'default': default_color}
    # else:
    #     colors = generate_colors(len(node_types))
    #     color_map = dict(zip(node_types, colors))
    #     default_color = colors[0] if colors else '#888888'

    # 边信息
    # edge_x, edge_y, edge_text = [], [], []
    # for edge in G.edges(data=True):
    #     x0, y0 = pos[edge[0]]
    #     x1, y1 = pos[edge[1]]
    #     edge_x.extend([x0, x1, None])
    #     edge_y.extend([y0, y1, None])
    #     edge_text.append(edge[2].get('label', ''))
    edge_x, edge_y, edge_text, edge_colors = [], [], [], []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        rel_type = edge[2].get('label', 'UNKNOWN')
        edge_text.append(rel_type)
        edge_colors.append(color_map.get(rel_type, '#888'))  # 关系类型颜色

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        mode='lines+text',
        text=edge_text,
        # textposition='middle center'
        textposition='middle right',  # 调整文本位置
        textfont=dict(color='black')  # 确保文字可见
    )

    # 节点信息
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_type = data.get('type', 'Unknown')
        # node_text.append(f"{node}<br>Type: {node_type}")
        node_text.append(
            f"{data.get('text', node)}<br>"
            f"Type: {node_type}<br>"
            f"Properties: { {k:v for k,v in data.items() if k not in ['type','text']} }"
        )
        # node_color.append(color_map.get(node_type, default_color))
        node_color.append(color_map.get(node_type, '#666'))
        # 根据节点的连接数调整大小
        node_size.append(10 + len(list(G.neighbors(node))))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            # showscale=True,
            # colorscale=[[i / (len(color_map) - 1), color] for i, color in enumerate(color_map.values())] if len(
            #     color_map) > 1 else [[0, default_color], [1, default_color]],
            size=node_size,
            color=node_color,
            # line_width=2
            line=dict(width=2, color='DarkSlateGrey')
        ),
        text=[node for node in G.nodes()],
        textposition="top center",
        hovertext=node_text
    )

    # 创建图例
    legend_traces = []
    for node_type, color in color_map.items():
        legend_traces.append(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                legendgroup=node_type,
                showlegend=True,
                name=node_type
            )
        )

    # 创建图形
    fig = go.Figure(data=[edge_trace, node_trace] + legend_traces,
                    layout=go.Layout(
                        title=dict(
                            text='Knowledge Graph',
                            font=dict(size=16)
                        ),
                        showlegend=True,
                        hovermode='closest',
                        clickmode='event+select',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    ))

    return fig


def build_dynamic_cypher_query(relevant_info: Dict[str, Any], question: str) -> str:
    nodes = relevant_info["nodes"]
    relations = relevant_info["relations"]

    # 使用APOC进行模糊匹配
    find_nodes_query = f"""
    MATCH (n)
    WHERE apoc.text.fuzzyMatch(n.text, '{question}') > 0.5
    WITH n
    ORDER BY apoc.text.fuzzyMatch(n.text, '{question}') DESC
    LIMIT 5
    """

    # 第二步：探索这些节点的关系
    explore_relations_query = """
    OPTIONAL MATCH (n)-[r]-(related)
    WHERE NOT (related:Document)  // 排除 Document 类型的节点，因为它们可能是重复的内容
    """

    # 过滤条件（如果需要的话）
    filter_conditions = []
    if nodes:
        node_filter = " OR ".join([f"'{node}' IN labels(n) OR '{node}' IN labels(related)" for node in nodes])
        filter_conditions.append(f"({node_filter})")
    if relations:
        relation_filter = " OR ".join([f"type(r) = '{rel}'" for rel in relations])
        filter_conditions.append(f"({relation_filter})")

    filter_query = " AND ".join(filter_conditions)
    if filter_query:
        filter_query = f"WHERE {filter_query}"

    # 返回结果
    return_query = """
    RETURN DISTINCT n, r, related
    LIMIT 50
    """

    full_query = find_nodes_query + explore_relations_query + filter_query + return_query
    return full_query
