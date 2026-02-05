from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate
from typing import List, Optional, Tuple, Union
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import json
import re
import logging
from pdf_chunk import process_pdf
from langchain.schema import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.prompts import ChatPromptTemplate


from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_community.llms.tongyi import Tongyi
from structed_relations_generate import Relation_Generate
from typing import Type


class LegalRelation(BaseModel):
    head: str = Field(
        ...,
        description=(
            "法律主体或法律要素的明确标识"
            "必须使用法律文本中出现的规范表述"
            # "- 自然人使用完整身份描述（'未成年被监护人'而非'小孩'）"
            # "- 法律行为需动词化（'行使撤销权'而非'撤销权'）"
            # "- 客体要素完整（'婚前财产'而非'财产'）"
            # "- 不一定用条款原文，可以根据内容自己生成"
            "- 每个法律主体应当独立，如（任何组织或者个人）应当分为两个主体（任何组织）和（个人）"
        )
    )
    head_type: str = Field(
        ...,
        description=(
            "头部实体所属的主体类型"
        )
    )
    relation: str = Field(
        ...,
        description=(
            "法律关系，需体现法律特征"
            # "- 关系不一定用条款原文，可以根据内容自己生成"
        )
    )
    tail: str = Field(
        ...,
        description=(
            "法律关系的接收方要素，要求："
            "- 权利义务的客体需明确（'婚前财产归属'而非'财产'）"
            "- 法律效果需具体化（'合同解除'而非'解除'）"
            "- 法律主体不一定用条款原文，可以根据内容自己生成"
            # "- 当文本中一个主体与多个主体存在关系如('民事主体', '依法享有', '人身权利、财产权利以及其他合法权益') 应当生成两个关系 ('民事主体', '依法享有', '人身权利') 和 ('民事主体', '依法享有', '财产权利')"
        )
    )
    tail_type: str = Field(
        ...,
        description=(
            "尾部所属的主体类型，需符合关系逻辑："
            "须从预设的类中选择"
        )
    )
    provenance: str = Field(
        ...,
        description="抽取出该关系的源法律条文,必须是来自于当前给出的条文，即条文源内容"
    )


def create_civil_code_relation_prompt(
    node_labels: Optional[List[str]] = None,
    rel_types: Optional[Union[List[str], List[Tuple[str, str, str]]]] = None,
    relationship_type: Optional[str] = "tuple",
    additional_instructions: Optional[str] = "",
    function_calling:bool = False,
    system_prompt_str: str = "",
    without_function_call_pydantic_object: Optional[Type[BaseModel]] = None,
    examples: Optional[List[dict]] = None,
) -> ChatPromptTemplate:
    """
    LLMGraphTransformer 专用 Prompt生成函数

    参数：
    node_labels: 实体类型列表
    rel_types: 关系类型列表，支持三元组或字符串格式
    relationship_type: 关系格式类型，"tuple" 或 "string"
    """

    
    node_labels = node_labels
    node_labels_str = "、".join(node_labels)

    # 关系类型处理
    relation_examples = []
    if rel_types:
        if relationship_type == "tuple":
            # 提取唯一关系类型并生成示例
            unique_relations = list(
                {item[1] for item in rel_types if len(item) > 1})
            schema_examples = "\n".join(
                [f"- ({s}, {r}, {t})" for (s, r, t) in rel_types])
        else:
            unique_relations = rel_types
            schema_examples = ""
        rel_types_str = "、".join(unique_relations)
    else:
        rel_types_str = ""
        schema_examples = ""

    # 系统提示构建
    system_prompt_parts = [
        "您是一个知识图谱构建专家，请根据以下要求进行关系提取：",
        "1. 从文本中识别实体及其关系，实体类型：[" + node_labels_str + "]",
        "2. 关系类型使用：" + rel_types_str if rel_types else "",
        system_prompt_str,
        additional_instructions
    ]
    system_prompt = "\n".join(filter(None, system_prompt_parts))

    # 用户提示模板
    human_prompt_parts = [
        "仅从以下内容分析：",
        "{input}",
        "" if function_calling else "输出要求：{format_instructions}",
        "示例：" if examples is not None else "", "{examples}\n",
        "特别注意：",
        "必须严格做到生成的关系全都出自以上输入， 禁止生成以上内容没有提到的内容，禁止从除了以上内容之外的地方提取内容",
        "禁止重复相同的关系",
        additional_instructions
    ]

    # 创建消息模板
    system_message = SystemMessage(content=system_prompt)
 

    if function_calling:
        human_prompt = PromptTemplate(
            template="\n".join(filter(None, human_prompt_parts)),
            input_variables=["input"],
            partial_variables={
            # "format_instructions": parser.get_format_instructions(),
            # "node_labels": node_labels_str,
            # "rel_types": schema_examples if relationship_type == "tuple" else rel_types_str,
            # "examples": legal_examples,
            },
        )

    else:
        parser = JsonOutputParser(pydantic_object=without_function_call_pydantic_object)
        human_prompt = PromptTemplate(
            template="\n".join(filter(None, human_prompt_parts)),
            input_variables=["input"],
            partial_variables={
            "format_instructions": parser.get_format_instructions(),
            # "node_labels": node_labels_str,
            # "rel_types": schema_examples if relationship_type == "tuple" else rel_types_str,
            "examples": json.dumps(examples, ensure_ascii=False, indent=2) if examples else "",
            },
        )

    return ChatPromptTemplate.from_messages([
        system_message,
        HumanMessagePromptTemplate(prompt=human_prompt)
    ])


"""
代码测试
"""
if __name__ == "__main__":
    # 初始化日志
    logging.basicConfig(level=logging.DEBUG)

    # 初始化 LLM
    # 118.195.161.49:11303  Qwen3-14B-AWQ
    llm = ChatOpenAI(
        model="qwen-plus-latest",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-42eefc96379c4058a7f188a3fae46d51",
        # streaming=True
    )
    # "http://118.195.161.49:27984/v1"

    llm_local = ChatOpenAI(
        model = "qwen-local",
        base_url="http://118.195.161.49:27984/v1",
        api_key="QcyDAgAEAAYExiABBj5BzIGCAE",
        # max_completion_tokens=6000,
        # max_tokens = 8000,
    )

    # 初始化DashScope模型
    # tongyi_llm = Tongyi(
    #     model="qwen3-14b",
    #     api_key="sk-42eefc96379c4058a7f188a3fae46d51",
    #     streaming=True
    # )

    res_node = [
        "民事主体类"
        "权利类",
        "义务类",
        "法律行为",
        "法律活动/程序",
        "法律制度类",
        "法律关系客体类",
        "法律效力类"
    ]

    system_summary_rela_str = "您是民法典知识图谱架构师，请严格遵循以下规则归纳关系类型：" +    \
        "【核心规则】"+ \
        "1. 抽象层级：提炼法律关系本质"+    \
        "2. 关系类型必须是 动词短语（≤6字）"+   \
        "3. 关系类型中禁止包含任何具体权利/义务或关系客体，即"+ \
        " 禁止生成：'享有名誉权' '承担违约责任' '设立抵押权'"+  \
        " 应该生成：'享有' '承担' '设立'"+  \
        "4. 可以合并相似法律行为"+  \
        "5. 忽略时间/数量等修饰词"
    generator = Relation_Generate(llm=llm_local, node_labels=res_node, extract_prompt_str=system_summary_rela_str)

    result = process_pdf("民法典第一编.pdf")
    chunks_to_save = result.get("chunks", [])
    docs = []
    text = ""
    res_relation = []
    for chunk in chunks_to_save:
        for items in chunk["entries"]:
            text += items["content"] + "\n"
        doc = Document(page_content=text, metadata={'source': "民法典第一编.pdf"})
        docs.append(doc)
        relations = generator.generate_relations(inputs=text,relations=res_relation)
        res_relation.extend(relations)
        # print(res_relation)

    # print(json.dumps(res_relation, indent=2, ensure_ascii=False))

    # res_relation = ['通过实施', '依法承担', '终止引起', '指定确定', '根据确定', '分配剩余财产', '通过设立', '不得侵犯', '依法负担', '依法撤销', '决定于', '依法宣告', '保护维护', '公示为', '消灭于', '依据保护', '依照执行', '代表行使', '产生于', '请求履行', '可以申请', '依法代理', '应当遵循', '设立为',
    #                  '依法指定', '导致终止', '依法恢复', '依法保护', '设定为', '由...担任', '请求返还', '代理实施', '分配给', '依照实施', '请求要求', '按照分配', '依法变更', '变更导致', '规定规范', '基于产生', '变更登记', '视为具有', '设立产生', '指定为', '独立实施', '依法申请', '注销登记', '代理执行', '依法享有', '因由引起']
    custom_prompt = create_civil_code_relation_prompt(
        node_labels=res_node, rel_types=res_relation, relationship_type="string")

    # # 初始化Transformer（关键修改：绑定响应格式）
    transformer = LLMGraphTransformer(
        llm=llm_local,
        allowed_nodes=list(set(res_node)),
        allowed_relationships=list(set(res_relation)),
        # node_properties=True,
        # relationship_properties=True
        prompt=custom_prompt,
        strict_mode=False,
        ignore_tool_usage=True
    )

    all_graph_documents = []
    for i, doc in enumerate(docs):
        graph_doc = transformer.convert_to_graph_documents([doc])
        all_graph_documents.extend(graph_doc)
        print(i)
        # print(graph_doc)

    print(all_graph_documents)
    # print("节点:", [(node.id, node.type) for node in all_graph_documents[0].nodes])
    # print("关系:", [(rel.source.id, rel.type, rel.target.id)
    #       for rel in all_graph_documents[0].relationships])

    # 打印结果
    # print("\n生成的关系列表:")
