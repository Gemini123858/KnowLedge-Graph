import json
import os
import networkx as nx
from typing import Dict, Any
import hashlib


def generate_file_hash(file_content: bytes) -> str:
    return hashlib.md5(file_content).hexdigest()


def get_cache_dir():
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir


def save_processed_data(data: Dict[str, Any], file_name: str):
    cache_file = os.path.join(get_cache_dir(), file_name)
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_processed_data(file_name: str) -> Dict[str, Any]:
    cache_file = os.path.join(get_cache_dir(), file_name)
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def graph_to_dict(G: nx.Graph) -> Dict[str, Any]:
    """将NetworkX图转换为可序列化的字典"""
    return {
        'nodes': [
            {
                'id': node,
                'label': G.nodes[node].get('label', 'Unknown'),
                'title': G.nodes[node].get('title', ''),
                'properties': {k: v for k, v in G.nodes[node].items() if k not in ['label', 'title']}
            } for node in G.nodes()
        ],
        'edges': [
            {
                'source': edge[0],
                'target': edge[1],
                'label': G.edges[edge].get('label', ''),
                'properties': {k: v for k, v in G.edges[edge].items() if k != 'label'}
            } for edge in G.edges()
        ]
    }


def dict_to_graph(graph_dict: Dict[str, Any]) -> nx.Graph:
    """将字典转换回NetworkX图"""
    G = nx.Graph()
    for node in graph_dict['nodes']:
        G.add_node(node['id'], label=node['label'], title=node['title'], **node['properties'])
    for edge in graph_dict['edges']:
        G.add_edge(edge['source'], edge['target'], label=edge['label'], **edge['properties'])
    return G
