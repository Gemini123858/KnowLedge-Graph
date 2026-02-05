import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
import numpy as np
import requests
from config import EMBEDDING_CONFIG,SIMILARITY_THRESHOLD
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def dict_to_graphdoc(data: dict) -> GraphDocument:
    # 首先创建所有节点
    nodes_dict = {node_data["id"]: Node(id=node_data["id"], type=node_data["type"]) 
                 for node_data in data["nodes"]}
    
    # 然后创建关系，查找对应的节点对象
    relationships = []
    for rel_data in data["relationships"]:
        source_node = nodes_dict[rel_data["source_node_id"]]
        target_node = nodes_dict[rel_data["target_node_id"]]
        
        relationship = Relationship(
            source=source_node,
            target=target_node,
            type=rel_data["type"],
            properties={"provenance": rel_data["provenance"]}  # 确保属性字典中有provenance
        )
        relationships.append(relationship)
    
    source = data["source"]
    if isinstance(source, str):
        source = Document(page_content=source)  
    
    # 创建GraphDocument对象
    return GraphDocument(
        nodes=list(nodes_dict.values()),
        relationships=relationships,
        source=source
    )

import os
from openai import OpenAI
from typing import List
import logging

logger = logging.getLogger(__name__)

class BailianEmbeddings:
    """百炼平台OpenAI兼容模式嵌入服务"""
    def __init__(self, api_key: str = None, model: str = "text-embedding-v4"):
        """
        初始化百炼嵌入服务
        
        参数:
            api_key: 百炼API Key（从环境变量DASHSCOPE_API_KEY读取或直接传入）
            model: 使用的模型名称
        """
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    def embed_documents(self, texts: List[str], dimensions: int = 2048) -> List[List[float]]:
        """为文档列表生成嵌入向量（添加分批处理）"""
        if not texts:
            return []
        
        # 百炼平台单次请求最多支持10个文本
        batch_size = 10
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                logger.debug(f"处理批次 {batch_num}/{total_batches}，包含 {len(batch)} 个文本")
                
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=dimensions,
                    encoding_format="float"
                )
                
                # 提取嵌入向量并添加到结果列表
                batch_embeddings = [embedding.embedding for embedding in response.data]
                embeddings.extend(batch_embeddings)
                
                logger.debug(f"批次 {batch_num} 处理成功")
                
            except Exception as e:
                logger.error(f"百炼嵌入请求失败（批次 {batch_num}）: {str(e)}")
                # 对于失败批次，返回空向量占位符
                embeddings.extend([[]] * len(batch))
                logger.warning(f"已为批次 {batch_num} 添加空嵌入占位符")
        
        return embeddings
    
    def embed_query(self, text: str, dimensions: int = 1024) -> List[float]:
        """为单个查询生成嵌入向量"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text],
                dimensions=dimensions,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"百炼查询嵌入失败: {str(e)}")
            raise

class LocalEmbeddings:
    """封装本地嵌入服务的类"""
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.embedding_endpoint = f"{base_url}/embeddings"
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档列表生成嵌入向量"""
        if not texts:
            return []
        
        try:
            # 准备请求数据
            payload = {"input": texts}
            headers = {"Content-Type": "application/json"}
            
            # 发送请求到本地嵌入服务
            response = requests.post(self.embedding_endpoint, json=payload, headers=headers)
            response.raise_for_status()  # 检查HTTP错误
            
            # 解析响应
            result = response.json()
            # print(result)
            embeddings = [item["apis"] for item in result["data"]]
            
            return embeddings
        
        except requests.exceptions.RequestException as e:
            print(e)
            raise
        except (KeyError, ValueError) as e:
            print(e)
            raise
    def embed_query(self, text: str) -> List[float]:
        url = f"{self.base_url}/embeddings"
        response = requests.post(url, json={"input": text, "model": self.model})
        print(response.json())
        return response.json()["data"][0]["apis"]

def generate_relationship_signature(rel: Relationship) -> str:
    """生成关系的语义签名字符串"""
    source_str = f"{rel.source.id}(type:{rel.source.type})"
    target_str = f"{rel.target.id}(type:{rel.target.type})"
    # 将properties字典按key排序后拼接
    if rel.properties:
        prop_items = [f"{k}:{v}" for k, v in sorted(rel.properties.items())]
        prop_str = " | ".join(prop_items)
    else:
        prop_str = ""
    signature = f"{source_str} {rel.type} {target_str}"
    if prop_str:
        signature += f" | {prop_str}"
    return signature

def embed_relationships(graph_doc: GraphDocument, embeddings) -> tuple[list[Relationship], np.ndarray]:
    """为所有关系生成嵌入向量（使用LocalEmbeddings类）"""
    relationships = []
    signatures = []
    
    for rel in graph_doc.relationships:
        # 生成关系的语义签名
        signature = generate_relationship_signature(rel)
        relationships.append(rel)
        signatures.append(signature)
    
    # 使用LocalEmbeddings类生成嵌入向量
    logger.info(f"开始为 {len(signatures)} 个关系生成嵌入向量...")
    embeddings_list = embeddings.embed_documents(signatures)
    logger.info("嵌入向量生成完成")
    
    # 转换为NumPy数组
    embeddings_array = np.array(embeddings_list)
    
    return relationships, embeddings_array

def embed_nodes(graph_doc: GraphDocument, embeddings) -> tuple[list[Node], np.ndarray]:
    """为所有节点生成嵌入向量（使用LocalEmbeddings类）"""
    nodes = []
    signatures = []
    
    for node in graph_doc.nodes:
        # 生成节点的语义签名
        signature = f"{node.id}(type:{node.type})"
        nodes.append(node)
        signatures.append(signature)
    
    # 使用LocalEmbeddings类生成嵌入向量
    logger.info(f"开始为 {len(signatures)} 个节点生成嵌入向量...")
    embeddings_list = embeddings.embed_documents(signatures)
    logger.info("嵌入向量生成完成")
    
    # 转换为NumPy数组
    embeddings_array = np.array(embeddings_list)
    
    return nodes, embeddings_array


def optimize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """优化嵌入向量以提高区分度"""
    # 1. 标准化嵌入向量
    embeddings = (embeddings - np.mean(embeddings, axis=0)) / np.std(embeddings, axis=0)
    
    # 2. 降维保留重要特征
    pca = PCA(n_components=min(128, embeddings.shape[1]))
    embeddings = pca.fit_transform(embeddings)
    
    # 3. 应用UMAP进一步优化
    try:
        import umap
        reducer = umap.UMAP(n_components=min(64, embeddings.shape[1]), random_state=42)
        embeddings = reducer.fit_transform(embeddings)
    except ImportError:
        logger.warning("UMAP未安装，跳过此优化步骤")
    
    return embeddings

def optimize_embeddings_with_umap(embeddings: np.ndarray, n_components: int = 64) -> np.ndarray:
    """使用UMAP优化嵌入向量"""
    try:
        from umap import UMAP
        logger.info(f"使用UMAP降维到 {n_components} 维")
        umap = UMAP(
            n_components=n_components,
            n_neighbors=min(15, len(embeddings)-1),
            min_dist=0.1,
            metric='cosine',
            random_state=500
        )
        return umap.fit_transform(embeddings)
    except ImportError:
        logger.warning("UMAP未安装，使用PCA替代")
        return optimize_embeddings(embeddings)  # 回退到PCA


def find_similar_relationships(relationships: list[Relationship], embeddings: np.ndarray, 
                              similarity_threshold: float = SIMILARITY_THRESHOLD,type:int = 0) -> list[list[Relationship]]:
    """找出语义相似的关系组"""
    logger.info("开始计算相似关系组...")
    
    # embeddings = optimize_embeddings_with_umap(embeddings=embeddings,n_components=64)
    
    # 计算相似度矩阵
    similarity_matrix = cosine_similarity(embeddings)

    similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
    
    # 使用DBSCAN聚类算法找到相似关系组
    # 将相似度转换为距离 (1 - similarity)
    distance_matrix = 1 - similarity_matrix

    distance_matrix = np.maximum(distance_matrix, 0.0)

    # 计算距离矩阵的统计信息
    min_dist = np.min(distance_matrix)
    max_dist = np.max(distance_matrix)
    mean_dist = np.mean(distance_matrix)
    logger.info(f"距离矩阵统计 - 最小值: {min_dist:.6f}, 最大值: {max_dist:.6f}, 平均值: {mean_dist:.6f}")

    if type == 1:
        logger.info("尝试层次聚类...")
            # 设置距离阈值
        distance_threshold = 1 - similarity_threshold

            # 使用层次聚类
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage='average'
        )
        clusters = clustering.fit_predict(distance_matrix)

        logger.info("层次聚类成功完成")


    if type == 0:
        if len(distance_matrix) > 1:
            # 取上三角矩阵（不包括对角线）
            triu_distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
            eps_value = np.percentile(triu_distances,3)  #eps_value 小于%3的距离
        else:
            eps_value = 1-similarity_threshold  # 默认值

        logger.info(f"使用DBSCAN参数: eps={eps_value:.10f}")

        dbscan = DBSCAN(eps=eps_value, min_samples=1, metric='precomputed')
        clusters = dbscan.fit_predict(distance_matrix)

        
    print(f"clusters:{clusters}")
    # 按聚类分组关系
    cluster_groups = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(relationships[idx])
    
    # 返回分组（排除单元素组）
    similar_groups = [group for group in cluster_groups.values() if len(group) >= 1]
    
    logger.info(f"找到 {len(similar_groups)} 个相似关系组")
    return similar_groups

def post_process_similar_groups(similar_groups: List[List[Relationship]], embedding) -> List[List[Relationship]]:
    """后处理相似组，确保组内关系真正相似"""
    processed_groups = []
    
    for group in similar_groups:
        # 1. 计算组内平均相似度
        signatures = [generate_relationship_signature(rel) for rel in group]
        # embeddings = LocalEmbeddings(
        #     base_url=EMBEDDING_CONFIG['local']['base_url'], 
        #     model=EMBEDDING_CONFIG['local']['model']
        # )
        embeddings = embedding.embed_documents(signatures)

        embeddings = optimize_embeddings_with_umap(embeddings)
        
        similarity_matrix = cosine_similarity(embeddings)
        avg_similarity = np.mean(similarity_matrix)
        
        # 2. 如果平均相似度低于阈值，尝试拆分
        if avg_similarity < SIMILARITY_THRESHOLD:
            logger.warning(f"组内平均相似度 {avg_similarity:.4f} 低于阈值 {SIMILARITY_THRESHOLD:.4f}, 尝试拆分")
            
            distance_matrix = 1 - similarity_matrix
            if len(distance_matrix) > 1:
                # 取上三角矩阵（不包括对角线）
                triu_distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
                eps_value = np.percentile(triu_distances,25)  # 25%的距离小于这个值
            else:
                eps_value = 1-SIMILARITY_THRESHOLD  # 默认值

            # logger.info(f"使用DBSCAN参数: eps={eps_value:.6f}")

            dbscan = DBSCAN(eps=eps_value, min_samples=1, metric='precomputed')
            sub_clusters = dbscan.fit_predict(distance_matrix)

            # 创建新组
            sub_groups = {}
            for idx, cluster_id in enumerate(sub_clusters):
                if cluster_id not in sub_groups:
                    sub_groups[cluster_id] = []
                sub_groups[cluster_id].append(group[idx])
            
            # 只保留有多个关系的子组
            for sub_group in sub_groups.values():
                if len(sub_group) > 1:
                    processed_groups.append(sub_group)
        else:
            processed_groups.append(group)
    
    return processed_groups

def process_graph_document(graph_doc: GraphDocument, embeddings) -> dict[str, list[list[Relationship]]]:
    """处理GraphDocument，找到所有相似关系组"""
    # 嵌入所有关系
    relationships, embedding = embed_relationships(graph_doc=graph_doc,embeddings=embeddings)
    
    # 找到相似关系组
    similar_groups = find_similar_relationships(relationships, embedding, type=0)

    # 后处理相似组
    # similar_groups = post_process_similar_groups(similar_groups, embedding=embeddings)
    
    # 组织结果
    result = {
        "similar_relationship_groups": similar_groups,
        "total_relationships": len(relationships),
        "similar_groups_count": len(similar_groups),
        "similar_relationships_count": sum(len(group) for group in similar_groups)
    }
    
    return result



import hashlib

def node_hash(node):
    """
    为节点生成唯一hash值
    """
    return hashlib.md5(f"{node.id}_{node.type}".encode("utf-8")).hexdigest()

def merge_nodes_by_hash(graph_doc: GraphDocument, embeddings, node_similarity_threshold: float = 0.98) -> GraphDocument:
    # 1. 统计所有节点并生成hash
    node_hash_map = {}
    for node in graph_doc.nodes:
        h = node_hash(node)
        node_hash_map[h] = node

    # 2. 关系中关联hash
    rel_with_hash = []
    for rel in graph_doc.relationships:
        source_hash = node_hash(rel.source)
        target_hash = node_hash(rel.target)
        rel_with_hash.append((rel, source_hash, target_hash))

    # 3. 对所有节点hash去重后embedding
    unique_nodes = list(node_hash_map.values())
    node_signatures = [f"{node.id}(type:{node.type})" for node in unique_nodes]
    node_embeddings = embeddings.embed_documents(node_signatures)
    node_embeddings = np.array(node_embeddings)

    # 4. 节点聚类
    similarity_matrix = cosine_similarity(node_embeddings)
    distance_matrix = 1 - similarity_matrix
    distance_matrix = np.maximum(distance_matrix, 0.0)
    if len(distance_matrix) > 1:
        triu_distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        eps_value = np.percentile(triu_distances, 100 * (1 - node_similarity_threshold))
    else:
        eps_value = 1 - node_similarity_threshold
    dbscan = DBSCAN(eps=eps_value, min_samples=1, metric='precomputed')
    clusters = dbscan.fit_predict(distance_matrix)

    # 5. hash到新节点映射
    hash_to_new_node = {}
    for cluster_id in set(clusters):
        cluster_indices = [i for i, cid in enumerate(clusters) if cid == cluster_id]
        rep_node = unique_nodes[cluster_indices[0]]
        new_node = Node(id=f"merged_{cluster_id}", type=rep_node.type, properties=rep_node.properties.copy())
        for idx in cluster_indices:
            h = node_hash(unique_nodes[idx])
            hash_to_new_node[h] = new_node

    # 6. 关系重定向
    new_relationships = []
    for rel, source_hash, target_hash in rel_with_hash:
        new_source = hash_to_new_node.get(source_hash, rel.source)
        new_target = hash_to_new_node.get(target_hash, rel.target)
        if new_source.id == new_target.id:
            continue  # 可选：去除自环
        new_relationships.append(Relationship(
            source=new_source,
            target=new_target,
            type=rel.type,
            properties=rel.properties.copy()
        ))

    # 7. 新节点去重
    unique_new_nodes = list({node.id: node for node in hash_to_new_node.values()}.values())

    return GraphDocument(
        nodes=unique_new_nodes,
        relationships=new_relationships,
        source=graph_doc.source
    )
# ...existing code...


if __name__ == "__main__":

    # 整理Graphdoc

    # with open('民法第一二编关系提取_qwen.json', 'r', encoding='utf-8') as f:
    #     datas = json.load(f)

    # nodes:List[Node] = []
    # relationships:List[Relationship] = []
    # for item in datas:
    #     n = item.get('nodes',[])
    #     rel = item.get('relationships',[])
    #     nodes.extend(n)
    #     relationships.extend(rel)
    
    # # print(json.dumps(nodes, indent=2, ensure_ascii=False))
    # # print(json.dumps(relationships, indent=2, ensure_ascii=False))

    # graph_dict:dict = {}
    # graph_dict["nodes"] = nodes
    # graph_dict["relationships"] = relationships
    # graph_dict["source"] = ""

    # # print(json.dumps(graph_dict, indent=2, ensure_ascii=False))
    
    # with open('民法第一二编关系提取_qwen_整理.json', 'w', encoding='utf-8') as f:
    #     json.dump(graph_dict, f, indent=2, ensure_ascii=False)











    embedded = BailianEmbeddings(api_key="sk-42eefc96379c4058a7f188a3fae46d51")

    # test_embedding = embedding.embed_query(text="test", dimensions=2048)
    # print(f"成功生成嵌入向量，维度: {len(test_embedding)}, {test_embedding}")

    with open('民法第一二编关系提取_qwen_整理.json', 'r', encoding='utf-8') as f:
        graph_dict = json.load(f)

    # print(type(graph_dict))
    graph_doc = dict_to_graphdoc(graph_dict)

    # print(graph_doc)

    # embedded = LocalEmbeddings(base_url=EMBEDDING_CONFIG['local']['base_url'], model=EMBEDDING_CONFIG['local']['model'])
    # test_embedding = embedded.embed_query("test")
    # print(f"成功生成嵌入向量，维度: {len(test_embedding)}")
    # test_embedding = embedded.embed_documents(["test"])

    try:
        result = process_graph_document(graph_doc,embeddings=embedded)
    except Exception as e:
        logger.error(f"处理GraphDocument失败: {e}")
        raise

    logger.info(f"总关系数: {result['total_relationships']}")
    logger.info(f"发现相似关系组数: {result['similar_groups_count']}")
    logger.info(f"涉及相似关系数: {result['similar_relationships_count']}")

    # 保存结果到文件
    output_data = {
        "graph_document_source": graph_doc.source.page_content,
        "similar_relationship_groups": []
    }
    
    for i, group in enumerate(result["similar_relationship_groups"], 1):
        group_info = {
            "group_id": i,
            "relationships": [],
            "representative": generate_relationship_signature(group[0])
        }
        
        for j, rel in enumerate(group, 1):
            rel_info = {
                "signature": generate_relationship_signature(rel),
                "source": rel.source.id,
                "source_type": rel.source.type,
                "target": rel.target.id,
                "target_type": rel.target.type,
                "relation_type": rel.type,
                "provenance": rel.properties.get("provenance", "")
            }
            group_info["relationships"].append(rel_info)
        
        output_data["similar_relationship_groups"].append(group_info)
    
    # 5. 保存结果到JSON文件
    try:
        with open('民法第一二编关系embedding分组.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info("结果已保存到 'similar_relationships_results_qwen.json'")
    except Exception as e:
        logger.error(f"保存结果文件失败: {e}")


        


    

