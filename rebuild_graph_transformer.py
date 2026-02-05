from relation_generate import (
    create_civil_code_relation_prompt
)
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
from langchain_experimental.graph_transformers.llm import create_simple_model
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field
from typing import List, Optional
from pdf_chunk import process_pdf
from structed_relations_generate import Relation_Generate
from langchain_core.output_parsers import JsonOutputParser
from relation_generate import LegalRelation
import time
from config import system_prompt_str
DEFAULT_NODE_TYPE = "Node"

# 定义节点属性模型


class NodeProperty(BaseModel):
    key: str = Field(description="属性键名")
    value: str = Field(description="属性值")

# 定义节点模型


class node(BaseModel):
    id: str = Field(description="Name or human-readable unique identifier.")
    type: str = Field(description="节点类型")
    # properties: Optional[List[NodeProperty]] = Field(description="节点属性列表")

# 定义关系属性模型


class RelationshipProperty(BaseModel):
    key: str = Field(description="属性键名")
    value: str = Field(description="属性值")

# 定义关系模型


class relationship(BaseModel):
    source_node_id: str = Field(description="源节点ID")
    source_node_type: str = Field(description= "Name or human-readable unique identifier of source node")
    target_node_id: str = Field(description="目标节点ID")
    target_node_type: str = Field(description="Name or human-readable unique identifier of target node")
    type: str = Field(description="关系类型")
    provenance: str = Field(description="该关系的源法律条文,包括条文源内容")
    # properties: Optional[List[RelationshipProperty]] = Field(description="关系属性列表")


# 定义完整的图文档模型
class graphDocument(BaseModel):
    nodes: List[node] = Field(description="节点列表")
    relationships: List[relationship] = Field(description="关系列表")

class without_function_calling_graphDocument(BaseModel):
    relationships: List[LegalRelation] = Field(
        ...,
        description="提取出的关系列表",
    )


class GraphTransformer:
    def __init__(
            self,
            llm: BaseLanguageModel,
            allowed_nodes: List[str] = [],
            allowed_relationships: Union[List[str],
                                         List[Tuple[str, str, str]]] = [],
            # prompt: Optional[ChatPromptTemplate] = None,
            node_properties: Union[bool, List[str]] = False,
            relationship_properties: Union[bool, List[str]] = False,
            function_calling:bool = False,
            # 新增参数：用于 non-function-calling 情况下的 pydantic 模型
            pydantic_object: Optional[Type[BaseModel]] = None,
            # 可选：function-calling 模式下单独使用的 pydantic 模型（若未提供将回退到上面那个）
            function_calling_pydantic: Optional[Type[BaseModel]] = None,
            prompt_str: str = ""

            # additional_instructions: str = "",
            # relationship_type: str = "string"

    ):
        self.allowed_nodes = allowed_nodes
        self.allowed_relationships = allowed_relationships
        
        self.function_calling = function_calling

        # 参数类型检查（确保传入的是 BaseModel 的子类）
        if not function_calling:
            if not (isinstance(pydantic_object, type) and issubclass(pydantic_object, BaseModel)):
                raise TypeError("pydantic_object must be a subclass of pydantic.BaseModel")
        else:
            if not (isinstance(function_calling_pydantic, type) and issubclass(function_calling_pydantic, BaseModel)):
                raise TypeError("function_calling_pydantic must be a subclass of pydantic.BaseModel or None")
        if self.function_calling:
            self.prompt = create_civil_code_relation_prompt(node_labels=allowed_nodes,
                                               rel_types=allowed_relationships,
                                               relationship_type="string",
                                               function_calling=True,
                                               system_prompt_str=prompt_str,

                                               )
            structured_llm = llm.with_structured_output(
                schema=function_calling_pydantic, include_raw=True, method="function_calling")
            self.chain = self.prompt | structured_llm
        
        else:
            self.prompt = create_civil_code_relation_prompt(node_labels=allowed_nodes,
                                                            rel_types=allowed_relationships,
                                                            relationship_type="string",
                                                            function_calling=False,
                                                            system_prompt_str=prompt_str,
                                                            without_function_call_pydantic_object=pydantic_object)
            parser = JsonOutputParser(pydantic_object=pydantic_object)
            self.chain = self.prompt | llm | parser

    def get_prompt(self):
        return self.prompt
    
    def extraction_from_document(self,
                                 document:Document
                                 ) -> dict:
        # a = 1
        # graph_data:List[GraphDocument] = []
        
        gpd:List[GraphDocument] = []

        # raw_shame
        doc = document
        text = doc.page_content
        # print(text)
        # print("Prompt 模板参数名：", self.prompt.input_variables)  # 检查参数名
        # print("格式化后的 prompt：", self.prompt.format(input=text))  # 检查填充效果
        raw_shame = self.chain.invoke({"input": text})
        print(raw_shame)
        if self.function_calling:
            nodes: List[Node] = []
            relationships: List[Relationship] = []
            if raw_shame["parsed"]:
                raw_shame = raw_shame["parsed"]
            else: 
                return []
            for node in raw_shame.nodes:
                nodes.append(
                    Node(
                        id=node.id,
                        type=node.type,
                    )
                )
            for rel in raw_shame.relationships:
                # a = 0
                source = Node(id=rel.source_node_id, type=rel.source_node_type)
                target = Node(id=rel.target_node_id, type=rel.target_node_stype)
                properties = {}
                properties["provenance"] = rel.provenance
                relationships.append(
                    Relationship(
                        source=source, target=target, type=rel.type, properties=properties
                                )
                )
            gpd.append(GraphDocument(nodes=nodes, relationships=relationships,source=Document(page_content="")))
        else:
            nodes: List[Node] = []
            relationships: List[Relationship] = []
            for rel in raw_shame["relationships"]:
                source = Node(id=rel["head"], type=rel["head_type"])
                target = Node(id=rel["tail"], type=rel["tail_type"])
                properties = {}
                # properties["provenance"] = rel["provenance"]
                for k,v in rel.items():
                    if k not in ["head", "head_type", "tail", "tail_type", "relation_type"]:
                        properties[k] = v
                relationships.append(
                    Relationship(
                        source=source, target=target, type=rel["relation_type"], properties=properties
                                )
                )
                nodes.append(source)
                nodes.append(target)
            # 字典去重
            unique_nodes_dict = {(n.id, n.type): n for n in nodes}
            unique_nodes = list(unique_nodes_dict.values())
            gpd.append(GraphDocument(nodes=unique_nodes, relationships=relationships, source=doc))

        res = {
            'gpd_list' : gpd,
            'raw_shame' : raw_shame
        }
        return res


def graphdoc_to_dict(graphdoc:GraphDocument):
    return {
        "nodes": [{"id": node.id, "type": node.type} for node in graphdoc.nodes],
        "relationships": [
            {
                "source_node_id": rel.source.id,
                "source_node_type": rel.source.type,
                "target_node_id": rel.target.id,
                "target_node_type": rel.target.type,
                "type": rel.type,
                "provenance": rel.properties["provenance"]
            }
            for rel in graphdoc.relationships
        ],
        "source": str(graphdoc.source)  # 处理原始文档
    }

if __name__ == "__main__":
    # a = 1

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
    res_relation = ['通过实施', '依法承担', '终止引起', '指定确定', '根据确定', '分配剩余财产', '通过设立', '不得侵犯', '依法负担', '依法撤销', '决定于', '依法宣告', '保护维护', '公示为', '消灭于', '依据保护', '依照执行', '代表行使', '产生于', '请求履行', '可以申请', '依法代理', '应当遵循', '设立为',
                    '依法指定', '导致终止', '依法恢复', '依法保护', '设定为', '由...担任', '请求返还', '代理实施', '分配给', '依照实施', '请求要求', '按照分配', '依法变更', '变更导致', '规定规范', '基于产生', '变更登记', '视为具有', '设立产生', '指定为', '独立实施', '依法申请', '注销登记', '代理执行', '依法享有', '因由引起']
    test_text = """
    第一条　为了保护民事主体的合法权益，调整民事关系，维护社会和经济秩序，适应中国特色社会主义发展要求，弘扬社会主义核心价值观，根据宪法，制定本法。
    
    第二条　民法调整平等主体的自然人、法人和非法人组织之间的人身关系和财产关系。

    第三条　民事主体的人身权利、财产权利以及其他合法权益受法律保护，任何组织或者个人不得侵犯。

    第四条　民事主体在民事活动中的法律地位一律平等。

    第五条　民事主体从事民事活动，应当遵循自愿原则，按照自己的意思设立、变更、终止民事法律关系。

    第六条　民事主体从事民事活动，应当遵循公平原则，合理确定各方的权利和义务。

    第七条　民事主体从事民事活动，应当遵循诚信原则，秉持诚实，恪守承诺。

    第八条　民事主体从事民事活动，不得违反法律，不得违背公序良俗。

    第九条　民事主体从事民事活动，应当有利于节约资源、保护生态环境。

    第十条　处理民事纠纷，应当依照法律；法律没有规定的，可以适用习惯，但是不得违背公序良俗。

    第十一条　其他法律对民事关系有特别规定的，依照其规定。

    第十二条　中华人民共和国领域内的民事活动，适用中华人民共和国法律。法律另有规定的，依照其规定。
    """
    llm_local = ChatOpenAI(
        model="qwen-local",
        base_url="http://118.195.161.49:27984/v1",
        api_key="QcyDAgAEAAYExiABBj5BzIGCAE",
        max_tokens=5000,
        frequency_penalty=0.5,
        presence_penalty=0.5,
    )

    llm_qwen = ChatOpenAI(
        model="qwen-turbo-latest",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-42eefc96379c4058a7f188a3fae46d51",
    )

    # res_relation = ['设立为', '消灭于', '申请宣告死亡', '通过设立', '清算注销', '依法变更', '基于产生', '指定确定', '变更导致', '指定', '注销登记', '撤销', '请求撤销', '合并分立', '参照适用', 
    #        '设立建设用地使用权', '代理实施', '代理执行', '依法宣告', '设立物权', '因由引起', '保护维护', '合并分立转移', '变更登记事项', '自始无效', '返还不当利益', '指定临时监护',
    #          '依法恢复', '终止', '通过实施', '按照分配', '遗嘱指定', '直接指示', '公示信息', '依法保护', '承担', '变更登记', '具有法律约束力', '依法代理', '承担法律责任', '依法撤销监护',
    #            '申请宣告', '视为具有', '撤销监护', '依法承担连带责任', '决定于', '可以申请', '依照实施', '确定权利义务', '依法享有', '分配给', '规定规范', '设立产生', '由...担任', '依据保护',
    #              '依法承担', '分配剩余财产', '请求履行', '设定为', '设立', '请求返还财产', '导致终止', '指定为', '依法撤销', '申请认定', '享有', '应当遵循', '恢复', '恢复民事行为能力', '产生于', 
    #              '公示为', '不得侵犯', '请求返还', '终止引起', '依法指定', '协议确定', '请求要求', '临时担任', '变更', '独立实施', '终止民事', '履行职责', '依法负担', '根据确定', '指定监护', 
    #              '给予补偿', '代表行使', '恢复资格', '依法申请', '依照执行']
    # print(res)

  
    res_relation = ['设立', '变更', '终止', '发出要约', '作出承诺', '撤回', '撤销', '完成签署', '执行交易', '发现提议', '建立预约', '追认', '认可', '确定效力', '维持效力', '适用规定',
                     '补充协议', '确定标准', '指定交付时间', '调整价格', '以货币履行', '行使选择权', '确定标的', '分担债务', '追偿债务', '返还债权', '中止履责', '拒绝履职', '恢复履责',
                       '协商', '重新协商', '终止权利义务', '代为履职', '取得', '主张抗辩', '负担费用', '同意转移', '加入债务', '承担从债', '一并转让', '请求确认', '进行提存', '及时通知', 
                       '转移风险', '调整', '抵作', '防止损失扩大', '作出担保', '中止支付', '执行回收', '解除合同', '视为购买', '保留所有权', '请求回赎', '履约', '承担责任', '请求支付', 
                       '及时通知', '披露信息', '定期提交', '检查监督', '终止履行', '协商确定', '继续使用', '登记对抗', '签订协议', '检查监督', '更换', '通知', '履行协助', '提交资料',
                         '办理托运', '禁止携带', '告知事项', '及时通知', '调整服务', '尽力救助', '支付', '提供', '完成协作', '制定计划', '使用经费', '交付成果', '进行投资', '分工参与', 
                         '协作配合', '约定风险分配', '保证合法性', '禁止转交', '禁止使用', '履行返还', '主张抗辩', '验收检查', '发出凭证', '制定格式', '允许转让', '催告处置', '紧急处置', 
                         '请求提取', '收取费用', '催告处置', '提存', '承担责任', '遵循指示', '处理事务', '同意转委托', '选任第三人', '直接约束', '披露信息', '转移财产', '及时通知', 
                         '采取措施', '公开承诺', '定期公开', '配合检查', '告知事项', '续订合同', '催告处置', '提存物品', '履行出资', '协商分配', '披露信息', '请求偿还', '请求补偿', 
                         '承担', '采取措施', '报告', '转交', '追认']
    system_summary_rela_str = "您是民法典知识图谱架构师，请严格遵循以下规则归纳关系类型：" +    \
        "【核心规则】"+ \
        "1. 抽象层级：提炼法律关系本质"+    \
        "2. 关系类型必须是 动词短语（≤6字）"+   \
        "3. 关系类型中禁止包含任何具体权利/义务或关系客体，即"+ \
        " 禁止生成：'享有名誉权' '承担违约责任' '设立抵押权'"+  \
        " 应该生成：'享有' '承担' '设立'"+  \
        "4. 可以合并相似法律行为"+  \
        "5. 忽略时间/数量等修饰词"
    generator = Relation_Generate(llm=llm_qwen, node_labels=res_node,function_calling=False, extract_prompt_str=system_summary_rela_str)
    result = process_pdf("民法典第三编（4）.pdf")
    chunks_to_save = result.get("chunks", [])
    docs = []
    res_relation = []
    for chunk in chunks_to_save:
        text = ""
        for items in chunk["entries"]:
            text += items["content"] + "\n"
        doc = Document(page_content=text, metadata={'source': "民法典第三编（4）.pdf"})
        docs.append(doc)
        relations = generator.generate_relations(inputs=text,relations=list(set(res_relation)))
        res_relation.extend(relations)

    print(res_relation)

    relation_pro: List[str] = ["出处"]

    # prompt = create_civil_code_relation_prompt(node_labels=res_node,
    #                                            rel_types=res_relation,
    #                                            relationship_type="string",
    #                                            function_calling=True
    #                                            )
    
    # print(prompt)

    transformer = GraphTransformer(
        # llm=llm_local,
        llm=llm_qwen,
        allowed_nodes=res_node,
        allowed_relationships=res_relation,
        # prompt=prompt,
        relationship_properties=relation_pro,
        function_calling=False,
        pydantic_object=without_function_calling_graphDocument,
        function_calling_pydantic=None,
        prompt_str=system_prompt_str
    )


    # graphdoc = transformer.extraction_from_document(Document(page_content=test_text))
    

    # graphdoc = []
    # for doc in docs:
    #     graphdoc.extend(transformer.extraction_from_document(doc))
   
    # graphdoc_dict = [graphdoc_to_dict(gpd) for gpd in graphdoc]

    graphdoc_dict = []

    #  生成训练数据
    train_data = []
    prompt = transformer.get_prompt()
    # print(prompt)
    # docs = [Document(page_content=test_text, metadata={'source': "民法典第一编.pdf"})]

    for doc in docs:
        train_item = {
            'instruction' :str(prompt),
            'input' :doc.page_content,
            'output':""
        }
        time.sleep(3)
        gpd_ = transformer.extraction_from_document(doc)
        # print(doc.page_content)
        train_item['output'] = gpd_['raw_shame']
        train_data.append(train_item)
        for gpd in gpd_['gpd_list']:
            gpd_dict = graphdoc_to_dict(gpd)
            graphdoc_dict.append(gpd_dict)
            # train_item['output'].append(gpd_dict)
        

    # print(train_data)
    with open('民法典第三编训练数据（4）.json', 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
            

        


    for g in graphdoc_dict:
        print(json.dumps(g, indent=2, ensure_ascii=False))
    with open('民法第三编关系提取_qwen（4）.json', 'w', encoding='utf-8') as f:
        json.dump(graphdoc_dict, f, indent=2, ensure_ascii=False)

