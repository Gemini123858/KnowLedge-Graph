"""
DataGraphX - A Knowledge Graph-based RAG Application

This project implements a Retrieval-Augmented Generation system using knowledge graphs,
combining LangChain, Neo4j, and large language models for document analysis.

Author: Yu Wang
License: CC BY-NC-SA 4.0
"""
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from typing import List
import re
import jieba
import jieba.analyse
import streamlit as st
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfReader

from langchain_community.vectorstores import Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from py2neo import Graph
from streamlit_plotly_events import plotly_events
from langchain.prompts import ChatPromptTemplate
from relation_generate import (
    create_civil_code_relation_prompt
)
from pdf_chunk import (
    law_chunker,
    preprocess_text
)
from structed_relations_generate import Relation_Generate

from api_utils import (
    LocalEmbeddings,
    test_api_connection,
    test_embeddings
)
from config import (
    API_CONFIG,
    NEO4J_CONFIG,
    EMBEDDING_CONFIG,
    GRAPH_CONFIG,
    DOC_CONFIG,
    system_prompt_str,
    system_summary_rela_str
)
from data_persistence_utils import generate_file_hash
from data_persistence_utils import (
    save_processed_data,
    get_cache_dir
)
from knowledge_graph_utils import load_graph_from_json, find_relevant_subgraph, create_knowledge_graph


class StreamlitHandler(logging.Handler):
    def __init__(self, st_container):
        super().__init__()
        self.st_container = st_container

    def emit(self, record):
        log_entry = self.format(record)
        self.st_container.text(log_entry)


def clean_text(text: str) -> str:
    """
    清除文本中的控制字符，特别是 \x00-\x08, \x0B-\x0C, \x0E-\x1F 这些非法字符
    """
    # 此处将换行符替换为空格，并移除其它控制字符
    text = text.replace("\n", " ")
    return re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', text)


async def process_and_save_pdf_data(transformer, docs, file_name, logger):
    logger.info(f"开始处理 {len(docs)} 个文档")
    all_graph_documents = []
    total_docs = len(docs)
    for i, doc in enumerate(docs):
        graph_doc = await asyncio.to_thread(transformer.convert_to_graph_documents, [doc])
        all_graph_documents.extend(graph_doc)
        logger.info(f"处理进度: {(i + 1)}/{total_docs}")

    # 将图数据转换为可序列化的格式
    serializable_data = []
    for doc in all_graph_documents:
        doc_data = {
            "nodes": [{"id": node.id, "type": node.type, "properties": node.properties} for node in doc.nodes],
            "relationships": [
                {"source": rel.source, "target": rel.target, "type": rel.type, "properties": rel.properties} for rel in
                doc.relationships]
        }
        serializable_data.append(doc_data)

    # 保存数据到 JSON 文件
    cache_file_name = f"{file_name}_graph_data.json"
    save_processed_data(
        {"graph_documents": serializable_data}, cache_file_name)
    logger.info(f"图数据已保存到 {cache_file_name}")

    return all_graph_documents


async def convert_and_add_to_graph(transformer, docs, graph, progress_bar, logger, file_name, file_hash):
    logger.info(f"开始处理 {len(docs)} 个文档")
    all_graph_documents = []
    total_docs = len(docs)
    for i, doc in enumerate(docs):
        graph_doc = await asyncio.to_thread(transformer.convert_to_graph_documents, [doc])
        all_graph_documents.extend(graph_doc)
        progress = (i + 1) / total_docs
        progress_bar.progress(0.7 + progress * 0.15)  # 70% 到 85% 的进度
        logger.info(f"处理进度: {(i + 1)}/{total_docs}")

    # 将图数据转换为可序列化的格式
    serializable_data = {
        "nodes": [],
        "relationships": []
    }
    for doc in all_graph_documents:
        for node in doc.nodes:
            serializable_data["nodes"].append({
                "id": str(node.id),  # 确保id是字符串
                "type": node.type,
                # 将所有属性值转换为字符串
                "properties": {k: str(v) for k, v in node.properties.items()}
            })
        for rel in doc.relationships:
            serializable_data["relationships"].append({
                "source": str(rel.source),  # 确保source是字符串
                "target": str(rel.target),  # 确保target是字符串
                "type": rel.type,
                # 将所有属性值转换为字符串
                "properties": {k: str(v) for k, v in rel.properties.items()}
            })

    # 保存数据到 JSON 文件
    cache_file_name = f"{file_hash}_graph_data.json"
    save_processed_data(serializable_data, cache_file_name)
    logger.info(f"图数据已保存到 {cache_file_name}")

    logger.info(f"开始添加 {len(all_graph_documents)} 个图形文档到数据库")
    await asyncio.to_thread(graph.add_graph_documents, all_graph_documents, include_source=True)
    logger.info("图形文档添加完成")


def get_relevant_schema_from_neo4j(graph: Neo4jGraph, question: str, max_nodes: int = 5) -> Dict[str, Any]:
    query = """
    CALL db.index.fulltext.queryNodes("entity_index", $question) YIELD node, score
    WHERE score > 0.5
    WITH node, score
    ORDER BY score DESC
    LIMIT $max_nodes
    MATCH (node)-[r]-(related)
    RETURN 
        labels(node) AS node_labels, 
        properties(node) AS node_properties,
        type(r) AS relationship_type,
        labels(related) AS related_labels
    """
    results = graph.query(
        query, {"question": question, "max_nodes": max_nodes})

    schema = {}
    for result in results:
        node_label = result["node_labels"][0]
        if node_label not in schema:
            schema[node_label] = {"properties": [], "relationships": []}

        # 添加属性
        for prop in result["node_properties"].keys():
            if prop not in schema[node_label]["properties"]:
                schema[node_label]["properties"].append(prop)

        # 添加关系
        related_label = result["related_labels"][0]
        relationship = {
            "type": result["relationship_type"],
            "target": related_label
        }
        if relationship not in schema[node_label]["relationships"]:
            schema[node_label]["relationships"].append(relationship)

    return schema


def build_dynamic_cypher_query(relevant_info: Dict[str, Any], question: str) -> str:
    nodes = relevant_info["nodes"]
    relations = relevant_info["relations"]

    # 基础查询：找到包含问题关键词的节点
    base_query = f"""
    MATCH (n)
    WHERE n.text CONTAINS '{question}'
    """

    # 扩展查询：探索相关节点和关系
    expand_query = """
    MATCH (n)-[r]-(related)
    """

    # 过滤条件
    filter_conditions = []
    if nodes:
        node_filter = " OR ".join(
            [f"'{node}' IN labels(n) OR '{node}' IN labels(related)" for node in nodes])
        filter_conditions.append(f"({node_filter})")
    if relations:
        relation_filter = " OR ".join(
            [f"type(r) = '{rel}'" for rel in relations])
        filter_conditions.append(f"({relation_filter})")

    filter_query = " AND ".join(filter_conditions)
    if filter_query:
        filter_query = f"WHERE {filter_query}"

    # 返回结果
    return_query = """
    RETURN DISTINCT n, r, related
    LIMIT 50
    """

    full_query = base_query + expand_query + filter_query + return_query
    return full_query


def prepare_llm_input(question: str, schema: Dict[str, Any], cypher_query: str,
                      query_result: List[Dict[str, Any]]) -> str:
    schema_summary = "\n".join(
        [f"- {label}: {props}" for label, props in schema.items()])
    result_summary = "\n".join([str(item)
                               for item in query_result[:5]])  # 限制结果数量

    return f"""
    分析以下信息并回答问题：
    
    问题：{question}
    
    相关模式摘要：
    {schema_summary}
    
    执行的Cypher查询：
    {cypher_query}
    
    查询结果摘要：
    {result_summary}
    
    请提供以下格式的回答：
    1. 解释：[解释查询和结果]
    2. 回答：[基于查询结果的回答]
    """


def get_relevant_nodes_and_relations(graph: Neo4jGraph, question: str, allowed_nodes: List[str],
                                     allowed_relationships: List[str]) -> Dict[str, Any]:
    relevant_nodes = []
    for node in allowed_nodes:
        if node.lower() in question.lower():
            relevant_nodes.append(node)

    relevant_relations = []
    for relation in allowed_relationships:
        if relation.lower() in question.lower():
            relevant_relations.append(relation)

    return {
        "nodes": relevant_nodes,
        "relations": relevant_relations
    }


def build_dynamic_cypher_query(relevant_info: Dict[str, Any], question: str) -> str:
    nodes = relevant_info["nodes"]
    relations = relevant_info["relations"]

    if not nodes and not relations:
        return f"""
        MATCH (n)
        WHERE apoc.text.fuzzyMatch(n.text, '{question}') > 0.5
        RETURN n
        ORDER BY apoc.text.fuzzyMatch(n.text, '{question}') DESC
        LIMIT 5
        """
    if len(nodes) == 1 and not relations:  # 直接查询特定类型的节点
        return f"MATCH (n:{nodes[0]}) RETURN n LIMIT 10"

    node_match = " OR ".join([f"n:{node}" for node in nodes])
    query = f"MATCH (n) WHERE ({node_match})"

    if relations:
        relation_match = " OR ".join(
            [f"type(r) = '{rel}'" for rel in relations])
        query += f" OPTIONAL MATCH (n)-[r]->(m) WHERE {relation_match}"
        return_clause = "RETURN n, r, m"
    else:
        query += " OPTIONAL MATCH (n)-[r]->(m)"
        return_clause = "RETURN n, r, m"

    query += f" {return_clause} LIMIT 10"
    return query


async def process_question(prompt: str, graph: Neo4jGraph, graph_config: dict, embeddings: Embeddings, llm: ChatOpenAI):
    """改进的问题处理函数"""
    logger = logging.getLogger('graphy')

    try:
        # 1. 提取关键词
        keywords = jieba.lcut(prompt)
        key_terms = [word for word in keywords if len(word) >= 2]  # 只保留长度>=2的词

        results = []

        # 2. 基于关键词的精确匹配
        for term in key_terms:
            term_query = """
            MATCH (n)
            WHERE n.text IS NOT NULL 
            AND toLower(n.text) CONTAINS toLower($term)
            RETURN n.text as content, 1.0 as score
            LIMIT 3
            """

            term_results = await asyncio.to_thread(
                graph.query,
                term_query,
                {"term": term}
            )
            results.extend(term_results)

        # 3. 基于整体语义的向量搜索
        vector_query = """
        CALL db.index.vector.queryNodes(
            'vector_index',
            5,
            $apis
        ) 
        YIELD node, score
        WHERE node.text IS NOT NULL
        AND score > 0.3
        RETURN node.text as content, score
        ORDER BY score DESC
        """

        try:
            question_embedding = await asyncio.to_thread(embeddings.embed_query, prompt)
            vector_results = await asyncio.to_thread(
                graph.query,
                vector_query,
                {"apis": question_embedding}
            )
            results.extend(vector_results)
        except Exception as e:
            logger.error(f"向量搜索出错: {str(e)}")

        # 4. 如果问题包含关系词,添加关系搜索
        relation_keywords = ["关系", "联系", "作用", "影响", "如何"]
        if any(word in prompt for word in relation_keywords):
            # 找出主要实体之间的关系
            terms_str = "|".join(key_terms)
            relation_query = f"""
            MATCH (n)-[r]-(m)
            WHERE n.text IS NOT NULL 
            AND m.text IS NOT NULL
            AND (
                ANY(term IN split($terms, '|') WHERE 
                    toLower(n.text) CONTAINS toLower(term)
                    OR toLower(m.text) CONTAINS toLower(term)
                )
            )
            RETURN n.text as source, type(r) as relation, m.text as target
            LIMIT 5
            """

            try:
                relation_results = await asyncio.to_thread(
                    graph.query,
                    relation_query,
                    {"terms": terms_str}
                )
                results.extend(relation_results)
            except Exception as e:
                logger.error(f"关系搜索出错: {str(e)}")

        # 5. 整理结果
        context = []
        seen_contents = set()

        # 处理文本结果
        for r in results:
            if 'content' in r and r['content']:
                content = r['content'].strip()
                if content and content not in seen_contents:
                    score_text = f"(相关度: {r['score']:.2f})" if 'score' in r else ""
                    context.append(f"- {content} {score_text}")
                    seen_contents.add(content)

        # 处理关系结果
        for r in results:
            if all(k in r for k in ['source', 'relation', 'target']):
                source = r['source'].strip()
                target = r['target'].strip()
                relation_text = f"- {source} --[{r['relation']}]--> {target}"
                if relation_text not in seen_contents:
                    context.append(relation_text)
                    seen_contents.add(relation_text)

        # 6. 生成回答
        if not context:
            return "未找到相关信息。", results, []

        llm_prompt = f"""
        基于以下检索到的信息回答问题：

        问题：{prompt}
        关键词：{', '.join(key_terms)}

        检索到的信息：
        {chr(10).join(context)}

        请按以下格式回答：
        1. 总结：总结检索到的信息要点
        2. 分析：分析信息的完整性和相关性
        3. 回答：基于检索到的信息回答问题
        """

        response = await llm.ainvoke(llm_prompt)
        return response.content, results, context

    except Exception as e:
        logger.error(f"处理问题时出错: {str(e)}")
        return str(e), None, None


def validate_config():
    assert NEO4J_CONFIG['url'], "Neo4j URL not configured"
    assert API_CONFIG['deepseek']['base_url'], "DeepSeek API base URL not configured"
    assert API_CONFIG['openai']['base_url'], "OpenAI API base URL not configured"
    print("Configuration loaded successfully")


async def main():
    st.set_page_config(
        layout="wide",
        page_title="DataGraphX",
        page_icon=":graph:"
    )
    st.sidebar.image('logo.png', use_container_width=True)
    with st.sidebar.expander("项目介绍"):
        st.markdown("""
    此应用程序允许您上传PDF文件，将其内容提取到Neo4j图形数据库中，并使用自然语言执行查询。
    它利用LangChain和DeepSeek的模型生成Cypher查询，实时与Neo4j数据库交互。
    """)
    st.title("DataGraphX：Langchain学习版")

    # 设置日志（新增）
    logger = logging.getLogger('graphy')
    logger.setLevel(logging.INFO)
    log_container = st.empty()
    handler = StreamlitHandler(log_container)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # 设置 API 和嵌入模型
    if 'API_CONFIGURED' not in st.session_state:
        st.sidebar.subheader("API 设置")
        api_option = st.sidebar.selectbox(
            "选择 API 类型", ["OpenAI", "DeepSeek", ])
        api_key = st.sidebar.text_input(
            f"输入您的 {api_option} API 密钥：", "sk-42eefc96379c4058a7f188a3fae46d51", type='password')
        # QcyDAgAEAAYExiABBj5BzIGCAE
        # [修改] 统一的模型输入
        model_name = st.sidebar.text_input(f"输入 {api_option} 模型名称:",
                                           value=API_CONFIG['deepseek']['default_model'] if api_option == "DeepSeek"
                                           else API_CONFIG['openai']['default_model'])
        # [修改] API 测试按钮
        if st.sidebar.button("测试 API 连接"):
            with st.spinner("正在测试 API 连接..."):
                success, message = test_api_connection(
                    api_option, api_key, model_name)
                if success:
                    st.sidebar.success(f"{api_option} API 连接成功: {message}")
                else:
                    st.sidebar.error(f"{api_option} API 连接失败: {message}")

        # [保持不变] 嵌入模型设置部分
        st.sidebar.subheader("嵌入模型设置")
        embed_option = st.sidebar.selectbox("选择嵌入模型", ["本地", "OpenAI"])
        if embed_option == "本地":
            embed_base_url = st.sidebar.text_input("本地嵌入 API URL:",
                                                   value=EMBEDDING_CONFIG['local']['base_url'])
            embed_model = st.sidebar.text_input("嵌入模型名称:",
                                                value=EMBEDDING_CONFIG['local']['model'])
        # [保持不变] 嵌入模型测试按钮
        if st.sidebar.button("测试嵌入模型"):
            with st.spinner("正在测试嵌入模型..."):
                if embed_option == "本地":
                    success, message = test_embeddings(
                        "本地", base_url=embed_base_url, model=embed_model)
                else:
                    success, message = test_embeddings(
                        "OpenAI", api_key=api_key)

                if success:
                    st.sidebar.success(f"嵌入模型测试成功: {message}")
                else:
                    st.sidebar.error(f"嵌入模型测试失败: {message}")

        # [修改] 确认设置按钮
        if st.sidebar.button("确认设置"):
            try:
                os.environ['OPENAI_API_KEY'] = api_key
                base_url = API_CONFIG[api_option.lower()]['base_url']
                # from langchain_community.llms.tongyi import Tongyi
                llm = ChatOpenAI(model=model_name, base_url=base_url, streaming=True)
                # llm = Tongyi(model=model_name, api_key=api_key)
                if embed_option == "本地":
                    embeddings = LocalEmbeddings(
                        base_url=embed_base_url, model=embed_model)
                else:
                    embeddings = OpenAIEmbeddings()

                st.session_state['embeddings'] = embeddings
                st.session_state['llm'] = llm
                st.session_state['API_CONFIGURED'] = True
                st.sidebar.success("API 和嵌入模型设置成功。")
            except Exception as e:
                st.error(f"设置 API 或嵌入模型时出错：{str(e)}")
                st.error("详细错误信息：")
                st.exception(e)
    else:
        embeddings = st.session_state['embeddings']
        llm = st.session_state['llm']

    # Neo4j 连接设置
    if 'neo4j_connected' not in st.session_state:
        st.sidebar.subheader("连接到Neo4j数据库")
        neo4j_url = st.sidebar.text_input(
            "Neo4j URL:", value=NEO4J_CONFIG['url'])
        neo4j_username = st.sidebar.text_input("Neo4j 用户名:", value="neo4j")
        neo4j_password = st.sidebar.text_input(
            "Neo4j 密码:", "test_pass", type='password')
        connect_button = st.sidebar.button("连接")
        if connect_button and neo4j_password:
            try:
                # 保持原有的 Neo4jGraph 连接
                graph = Neo4jGraph(
                    url=neo4j_url,
                    username=neo4j_username,
                    password=neo4j_password
                )
                # 添加 py2neo 的 Graph 连接
                py2neo_graph = Graph(neo4j_url, auth=(
                    neo4j_username, neo4j_password))

                st.session_state['graph'] = graph
                st.session_state['py2neo_graph'] = py2neo_graph
                st.session_state['neo4j_connected'] = True
                st.session_state['neo4j_url'] = neo4j_url
                st.session_state['neo4j_username'] = neo4j_username
                st.session_state['neo4j_password'] = neo4j_password
                st.sidebar.success("已成功连接到Neo4j数据库。")
            except Exception as e:
                st.error(f"连接到Neo4j失败：{str(e)}")
    else:
        graph = st.session_state['graph']
        py2neo_graph = st.session_state['py2neo_graph']
        neo4j_url = st.session_state['neo4j_url']
        neo4j_username = st.session_state['neo4j_username']
        neo4j_password = st.session_state['neo4j_password']

    # 确保在继续之前已建立Neo4j连接和配置API
    if 'API_CONFIGURED' in st.session_state and 'neo4j_connected' in st.session_state:
        uploaded_file = st.file_uploader("请选择一个PDF文件。", type="pdf")

        if uploaded_file is not None:
            # 获取上传文件的名称和内容
            file_name = uploaded_file.name
            file_content = uploaded_file.getvalue()

            # 计算文件内容的MD5哈希值
            file_hash = generate_file_hash(file_content)

            # 检查是否需要重新处理PDF
            cache_file_name = f"{file_hash}_graph_data.json"
            cache_file_path = os.path.join(get_cache_dir(), cache_file_name)

            # 检查Neo4j数据库中是否已存在此文件的处理结果
            check_query = f"MATCH (d:Document {{hash: '{file_hash}'}}) RETURN d"
            result = graph.query(check_query)

            if result:
                st.success(f"{file_name} 的处理结果已存在于数据库中。")
                need_processing = False
            else:
                need_processing = True

            if need_processing:
                st.info(f"开始处理 {file_name}...")
                progress_bar = st.progress(0)

                # 保存上传的文件到临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file_content)
                    tmp_file_path = tmp_file.name

                # 清理缓存
                cache_file = 'graph_documents_cache.pkl'
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    logger.info(f"缓存文件 {cache_file} 已删除")

                # 清空数据库
                await asyncio.to_thread(graph.query, "MATCH (n) DETACH DELETE n;")
                logger.info("数据库已清空")

                progress_bar.progress(10)

                # 加载PDF

                # loader = PyPDFLoader(tmp_file_path)
                # pages = await asyncio.to_thread(loader.load)
                # logger.info(f"PDF加载完成，页数: {len(pages)}")
                # progress_bar.progress(30)

                # # 分割文本
                # text_splitter = RecursiveCharacterTextSplitter(
                #     chunk_size=DOC_CONFIG['chunk_size'],
                #     chunk_overlap=DOC_CONFIG['chunk_overlap']
                # )
                # with ThreadPoolExecutor() as executor:
                #     split_docs = list(executor.map(
                #         lambda page: text_splitter.split_text(
                #             page.page_content if hasattr(page, 'page_content') else page[1]),
                #         pages
                #     ))
                # docs = [Document(page_content=split_doc, metadata={'source': file_name}) for page_splits in split_docs
                #         for split_doc in page_splits]

                pdf_reader = PdfReader(tmp_file_path)
                logger.info(f"PDF共有 {len(pdf_reader.pages)} 页")
                # 分批处理以避免内存问题
                raw_text = ""
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        raw_text += page_text + "\n"
                    else:
                        logger.warning(f"第 {i+1} 页文本提取为空")
                logger.info(f"提取的原始文本长度: {len(raw_text)} 字符")
                logger.info("正在预处理文本...")
                processed_text = preprocess_text(raw_text)
                # 分块处理
                logger.info("正在提取法律条文...")
                chunk_result = law_chunker(processed_text)
                logger.info(
                    f"处理完成! 共提取 {chunk_result['total_entries']} 条法律条文，分为 {len(chunk_result['chunks'])} 个块")
                chunks_to_save = chunk_result.get("chunks", [])

                # docs = [Document(page_content=split_doc, metadata={'source': file_name}) for page_splits in split_docs
                #         for split_doc in page_splits]

                docs = []
                text = ""
                for chunk in chunks_to_save:
                    for items in chunk["entries"]:
                        text += items["content"] + "\n"
                    doc = Document(page_content=text, metadata={
                                   'source': file_name})
                    docs.append(doc)

                logger.info(f"文本分割完成，共 {len(docs)} 个文档片段")
                progress_bar.progress(40)

                # relation_generator = DynamicRelationGenerator(llm=llm)
                relation_generator = Relation_Generate(
                    llm=llm, node_labels=GRAPH_CONFIG["allowed_nodes"], extract_prompt_str=system_summary_rela_str)
                # dynamic_relations = []

                docs_num = len(docs)
                i = 1

                for doc in docs:
                    relations = relation_generator.generate_relations(
                        relations=GRAPH_CONFIG['allowed_relationships'], inputs=doc.page_content)
                #    logger.info(json.dumps(relations, indent=2, ensure_ascii=False))
                    logger.info("生成关系成功")
                    print(relations)
                    progress = 40 + 20 * i/docs_num
                    i += 1
                    progress_bar.progress(progress/100)
                    GRAPH_CONFIG['allowed_relationships'].extend(relations)

                # custom_prompt = ChatPromptTemplate.from_messages([
                #     ("system",
                #     "作为《民法典》知识图谱构建专家，你需要严格遵循以下规则：\n"
                #     "1. 节点类型只能是：民事主体、法律行为、民事客体、民事义务、法人、自然人、民事权利\n"
                #     "3. 禁止添加规则外的类型"),
                #     ("human",
                #      "分析以下文本，提取相关实体和关系：\n{input}"
                #      "- properties字段保留原始文本片段\n"
                #      ),
                # ])
                GRAPH_CONFIG['allowed_relationships'] = list(
                    set(GRAPH_CONFIG['allowed_relationships']))
                custom_prompt = create_civil_code_relation_prompt(
                    node_labels=GRAPH_CONFIG['allowed_nodes'],
                    rel_types=GRAPH_CONFIG['allowed_relationships'],
                    relationship_type="string",
                    system_prompt=system_prompt_str
                )

                logger.info(list(set(GRAPH_CONFIG['allowed_relationships'])))
                # print("Prompt模板变量：", prompt.input_variables)
                transformer = LLMGraphTransformer(
                    llm=llm,
                    allowed_nodes=GRAPH_CONFIG['allowed_nodes'],
                    allowed_relationships=list(
                        set(GRAPH_CONFIG['allowed_relationships'])),
                    #    allowed_relationships = [],
                    prompt=custom_prompt,
                    #    input_key = "text",
                    #    node_properties=True,
                    #    relationship_properties=True
                    strict_mode=False,
                    ignore_tool_usage=True
                )

                await convert_and_add_to_graph(transformer, docs, graph, progress_bar, logger, file_name,
                                               file_hash)  # 修改这里，添加 file_name 和 file_hash
                logger.info("图形转换完成")
                progress_bar.progress(85)

                # 先删除已存在的向量索引
                try:
                    await asyncio.to_thread(
                        graph.query,
                        "CALL db.index.vector.deleteIndex('vector_index')"
                    )
                except Exception as e:
                    logger.info(f"删除向量索引时出错（可能不存在）: {str(e)}")

                # 删除全文索引
                try:
                    await asyncio.to_thread(
                        graph.query,
                        "CALL db.index.fulltext.drop('entity_index')"
                    )
                except Exception as e:
                    logger.info(f"删除全文索引时出错（可能不存在）: {str(e)}")

                # 创建新的向量索引
                await asyncio.to_thread(
                    Neo4jVector.from_existing_graph,
                    embedding=embeddings,
                    url=neo4j_url,
                    username=neo4j_username,
                    password=neo4j_password,
                    database="neo4j",
                    node_label="研究内容",  # 指定了具体的节点标签
                    text_node_properties=["id", "text"],
                    embedding_node_property="apis",
                    index_name="vector_index",
                    keyword_index_name="entity_index",  # 添加了关键词索引名称
                    search_type="hybrid"  # 使用混合搜索类型
                )
                logger.info("向量索引创建完成")
                progress_bar.progress(100)

                # 添加文档节点到Neo4j，标记处理完成
                doc_query = f"""
                CREATE (d:Document {{name: '{file_name}', hash: '{file_hash}', processed: true}})
                """
                await asyncio.to_thread(graph.query, doc_query)

                st.success(f"{file_name} 处理完成并已添加到数据库。")

            # 更新session state
            st.session_state['current_file'] = file_name
            st.session_state['current_file_hash'] = file_hash
            st.session_state['file_processed'] = True

        # 聊天界面
        if 'file_processed' in st.session_state and st.session_state['file_processed']:
            st.subheader("问答系统")

            # 初始化聊天历史
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # 显示聊天历史
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # 聊天输入
            if prompt := st.chat_input("输入您的问题"):
                st.session_state.messages.append(
                    {"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()

                    try:
                        # 使用新的处理函数
                        response, vector_results, query_results = await process_question(
                            prompt,
                            graph,
                            GRAPH_CONFIG,
                            st.session_state['embeddings'],
                            st.session_state['llm']
                        )

                        # 显示回答
                        message_placeholder.markdown(response)

                        # 显示知识图谱
                        st.subheader("相关知识图谱")
                        try:
                            file_hash = st.session_state.get(
                                'current_file_hash')
                            if file_hash:
                                full_graph = load_graph_from_json(file_hash)
                                if full_graph:
                                    # 先尝试找相关子图
                                    relevant_subgraph = find_relevant_subgraph(
                                        full_graph, prompt)

                                    # 如果相关子图太小或没找到，就使用完整图谱
                                    if not relevant_subgraph or relevant_subgraph.number_of_nodes() < 3:
                                        relevant_subgraph = full_graph
                                        st.info("显示完整知识图谱")
                                    else:
                                        st.info("显示相关知识子图")

                                    fig = create_knowledge_graph(
                                        relevant_subgraph)
                                    selected_points = plotly_events(
                                        fig, click_event=True)

                                    # 添加图谱统计信息
                                    st.info(f"知识图谱包含 {relevant_subgraph.number_of_nodes()} 个节点和 "
                                            f"{relevant_subgraph.number_of_edges()} 个关系")

                                    # 显示选中节点的详细信息
                                    if selected_points:
                                        selected_node = selected_points[0]['pointNumber']
                                        node_info = list(relevant_subgraph.nodes(data=True))[
                                            selected_node]
                                        st.write(f"选中的节点: {node_info[0]}")
                                        st.write(
                                            f"节点内容: {node_info[1].get('text', '')}")
                                else:
                                    st.warning("无法加载知识图谱数据")

                        except Exception as e:
                            st.error(f"生成知识图谱时出错: {str(e)}")
                            logger.error(f"知识图谱生成错误: {str(e)}")
                            logger.exception("详细错误信息")

                        st.session_state.messages.append(
                            {"role": "assistant", "content": response})

                    except Exception as e:
                        error_message = f"处理问题时出错: {str(e)}"
                        message_placeholder.error(error_message)
                        logger.error(error_message)
                        logger.exception("详细错误信息")
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_message})


# 保持 main 函数的调用不变
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
