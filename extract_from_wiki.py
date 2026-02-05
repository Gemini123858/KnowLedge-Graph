from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.language_models import BaseLanguageModel
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional

import time
from structed_relations_generate import Relation_Generate
from rebuild_graph_transformer import GraphTransformer

# computer network Konwnledge graph construction
# relation summary with LLMs
# relation extraction with LLMs
from embedding_deduplication import BailianEmbeddings
import logging
from embedding_deduplication import process_graph_document
from embedding_deduplication import generate_relationship_signature

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NetworkRelation(BaseModel):
    head: str = Field(
        ...,
        description=(
            "网络中的核心概念实体的明确标识"
            "必须使用标准化的术语或技术表述。"
            # "- 设备可以是服务器、路由器、交换机等，协议可以是 TCP/IP、HTTP 等"
            # "- 每个设备、协议或通信方式应当有独立描述，避免笼统表达"
            # "- 网络协议名称需使用常见缩写或完整描述（如 TCP/IP 而非仅 TCP）"
        )
    )
    head_type: str = Field(
        ...,
        description=(
            "头部实体所属类型"
            "必须从预设的实体类别中选择"
        )
    )
    relation_type: str = Field(
        ...,
        description=(
            "网络关系的所属类别"
            "需从预设的关系类别中选择"   
        )
    )
    relation_condition: str = Field(
        ...,
        description=(
            "关系状语：表示该关系存在的前提/时间、空间条件/方式，可以取自上下文的状语。"
            " 'all' 表示无条件成立"
        )
    )
    tail: str = Field(
        ...,
        description=(
            "网络关系的客体对象，要求："
            "必须使用标准化的术语或技术表述。"
            "必须是概念性实体的描述，切勿出现多个并列实体"
            # "- 具体化网络组件或协议的受众（如 '接收数据'、'承载通信' 等）"
            # "- 关联性要明确（如 '支持'、'依赖于'）"
            # "- 如设备间的互联，需明确目标设备"
        )
    )
    tail_type: str = Field(
        ...,
        description=(
            "尾部所属的实体类型"
            "必须从预设的实体类别中选择"
            # "须从预设的设备或协议类别中选择"
            # "- 如：'路由器'、'交换机'、'协议'、'服务器' 等"
        )
    )
    # protocol: str = Field(
    #     ...,
    #     description=(
    #         "涉及的网络协议或通信协议"
    #         "- 如：TCP、UDP、HTTP、FTP、IP 等"
    #         "- 可依据协议的层次结构或协议的特定作用分类"
    #     )
    # )
    # layer: str = Field(
    #     ...,
    #     description=(
    #         "网络协议或设备所在的层次，"
    #         "如：'物理层'、'数据链路层'、'网络层'、'传输层'、'应用层'"
    #         "- 确保每个网络协议或设备根据 OSI 模型或 TCP/IP 模型进行分类"
    #     )
    # )
    source_document: str = Field(
        ...,
        description="抽取该关系的 源语句/段落 的回显"
    )


class without_function_calling_graphDocument(BaseModel):
    relationships: List[NetworkRelation] = Field(
        ...,
        description="提取出的关系列表",
    )

if __name__ == "__main__":
    llm_qwen = ChatOpenAI(
        model="qwen-turbo-latest",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-42eefc96379c4058a7f188a3fae46d51",
    )

    
    with open("../wiki_computer_network_txt/wiki_computer_network/9499.txt", "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file]
    
    print(lines)


    # time.sleep(200)
    source = ""
    text = [lines[3]]
    
    res_nodes = []
    res_relationships = []


    sys_relation_summary_str = """
    您是计算机网络知识图谱架构师，请严格遵循以下规则归纳关系类型：
    【核心规则】  
    1. **抽象层级**：提炼网络实体间的本质关系，避免过于具体的描述。  
    2. **关系类型**：必须是简洁的动词短语（≤6个字），描述实体之间的互动或功能。  
    3. **禁止包含具体协议、设备、网络层或特定数据流**：即不要包含像“使用TCP协议”或“传输数据包”的描述，应简化为“使用”或“传输”这样的一般性关系。  
    4. **可合并相似网络行为**：例如，“支持传输”与“传输支持”应合并为“传输”类关系。  
    5. **忽略修饰词**：如“高带宽”、“低延迟”等描述应忽略，专注于核心关系，不需提及数字、单位或其他量化描述。  
    6. **明确简洁**：关系类型应直观且能准确反映网络实体间的核心功能，避免复杂化。
    请从给定文本中提取出所有符合规则的关系类型，并以列表形式输出。

    """
    sys_extraction_str = """
        实体类型定义：
    - **设备类**：如路由器、交换机、服务器、防火墙等网络设备。
    - **协议类**：计算机网络中使用的通信协议，如 TCP/IP、UDP、HTTP、FTP、DNS 等。
    - **连接类**：网络中设备或节点之间的物理或逻辑连接类型，如有线连接、无线连接、VPN、Wi-Fi 等。
    - **网络层类**：表示协议所在的层次，如物理层、数据链路层、网络层、传输层、应用层等。
    - **带宽类**：表示网络连接的传输速率，如 100 Mbps、1 Gbps 等。
    - **延迟类**：表示数据传输的时间延迟，如 10 ms、100 ms 等。
    - **网络行为类**：网络设备或协议之间的交互或操作行为，如数据传输、数据包丢失、流量管理等。
    - **安全性类**：网络通信中的安全协议或措施，如加密、身份验证、防火墙等。
    特殊要求：
    - **重要**：当一个设备或协议与多个其他设备或协议之间存在关系时，应当分别提取出每一条关系。比如，若“路由器”和“交换机”之间存在多种协议或行为关系，应分别提取出每一条。
    - **详细性**：抽取出的关系数量没有限制，请尽可能详尽地提取出所有可能的网络行为关系，不遗漏任何可能的联系。
    - **简洁性**：每条关系描述应简洁明确，避免复杂的上下文或冗余信息。使用动词短语（如“连接”、“使用”、“支持”等），并确保关系描述短小（≤6字）。
    请严格按照这些规则从给定的文本中抽取所有可能的关系，并清晰列出每一条关系，确保准确且全面。

    """

    relationship_generator = Relation_Generate(
        llm=llm_qwen,
        node_labels=res_nodes,
        function_calling=False,
        extract_prompt_str=sys_relation_summary_str
        )
    
    docs = []
    for content in text:
        doc = Document(page_content=content, metadata={'source': source})
        docs.append(doc)
        relations = relationship_generator.generate_relations(inputs=content,
                                                              relations=list(set(res_relationships)))
        res_relationships.extend(relations)

    print("关系类型：", set(res_relationships))
    transformer = GraphTransformer(
        # llm=llm_local,
        llm=llm_qwen,
        allowed_nodes=res_nodes,
        allowed_relationships=res_relationships,
        function_calling=False,
        pydantic_object=without_function_calling_graphDocument,
        function_calling_pydantic=None,
        prompt_str=sys_extraction_str
    )

    graph_docs: List[GraphDocument]  = []
    for doc in docs:
        graph = transformer.extraction_from_document(doc)
        graph_doc = graph["gpd_list"]
        graph_docs.extend(graph_doc)

    ans_GraphDoc: GraphDocument = GraphDocument(nodes=[], relationships=[], source=None)
    for doc in graph_docs:
        for node in doc.nodes:
            ans_GraphDoc.nodes.append(node)
        for rela in doc.relationships:
            ans_GraphDoc.relationships.append(rela)
    
    print("nodes:", ans_GraphDoc.nodes)
    print("relationships:", ans_GraphDoc.relationships)

    embedding = BailianEmbeddings(api_key="sk-42eefc96379c4058a7f188a3fae46d51")
    
    try:
        result = process_graph_document(ans_GraphDoc, embeddings=embedding)
    except Exception as e:
        logger.error(f"处理GraphDocument失败: {e}")
        raise
        
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
    
    print(json.dumps(output_data, ensure_ascii=False, indent=4))