# from generate_dynamic_relation import Relation_Generate
from langchain_community.llms.tongyi import Tongyi
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import json

"""
用JsonOutputParser构造prompt，实现结构化输出(无需对输出做额外处理)
"""

legal_examples = [
    {
        "text": "民事主体依法享有名誉权、荣誉权等人格权利",
        "relation": "享有",
        "analysis": "将'依法享有名誉权'抽象为'享有'，连接主体与权利"
    },
    {
        "text": "遗嘱执行人不得侵占遗产",
        "relation": "不得侵占",
        "analysis": "将'不得侵占遗产'抽象为'不得侵占'，连接主体与权利"
    },
    {
        "text": "物业服务企业应当定期公布服务事项",
        "relation": "应当公开",
        "analysis": "将'应当公布'抽象为'应当公开'，连接主体与行为"
    }
]
# legal_examples = [
#     {
#         "text": "民事主体依法享有名誉权、荣誉权等人格权利",
#         "relation": "依法享有"
#     },
#     {
#         "text": "遗嘱执行人不得侵占遗产",
#         "relation": "不得侵占"
#     },
#     {
#         "text": "物业服务企业应当定期公布服务事项",
#         "relation": "应当公开"
#     },
#     {
#         "text": "网络虚拟财产受法律保护",
#         "relation": "受保护"
#     },
#     {
#         "text": "婚姻登记机关不得泄露当事人隐私",
#         "relation": "不得泄露"
#     }
# ]


class RelationTypeSummary(BaseModel):
    relations: List[str] = Field(
        ...,
        description="从文本中抽象出新的关系类型集合",
    )

def create_relation_summary_prompt(
    node_labels: List[str],
    relation_labels:List[str],
    legal_examples: Optional[List[dict]] = None,
    additional_instructions: Optional[str] = "",
    extract_instructions: str = ""
) -> ChatPromptTemplate:
    """
    法律条款关系类型归纳Prompt生成函数，可以实现结构化输出
    
    参数：
    node_labels: 法律主体类型列表（用于约束分析范围）
    legal_examples: 示例数据集（可选）
    additional_instructions: 附加指令
    """
    
    # 处理节点标签
    print(1)
    node_labels_str = "、".join(node_labels)
    print(2)
    relation_labels_str = "、".join(relation_labels)
    print(3)
    

    system_prompt_parts =  [
        extract_instructions,
    ]
    system_prompt = "\n".join(filter(None, system_prompt_parts))

    # 用户提示模板
    human_prompt_parts = [
       "请从以下条款/文本中提取关系类型：",
        "{inputs}",
        "约束条件：",
        "1. 预定义顶点类型{node_labels}",
        # "2. 每种关系必须连接两个预定义类型的顶点",
        "3. 已存在的关系类型{relation_labels}",
        "4. 请勿生成已存在的关系类型"
        "3. 输出最具有普遍适用性且重要的几种新的关系类型，若已经充分则输出无",
        "输出格式：",
        "{format_instructions}"
    ]
    
    # 创建消息模板
    system_message = SystemMessage(content=system_prompt)
    parser = JsonOutputParser(pydantic_object=RelationTypeSummary)
    
    human_prompt = PromptTemplate(
        template="\n".join(filter(None, human_prompt_parts)),
        input_variables=["inputs"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "node_labels": node_labels_str,
            "relation_labels":relation_labels_str
        },
    )

    return ChatPromptTemplate.from_messages([
        system_message,
        HumanMessagePromptTemplate(prompt=human_prompt)
    ])

class Relation_Generate:
    def __init__(self, 
                 llm: ChatOpenAI, node_labels: List[str],
                 additional_instructions: Optional[str] = "",
                 function_calling: bool = False,
                 extract_prompt_str: str = ""
    ):
        self.llm = llm
        self.node_labels = node_labels
        self.additional_instructions = additional_instructions
        self.function_calling = function_calling
        self.prompt_str = extract_prompt_str
        if function_calling:
            self.llm = llm.with_structured_output(
            schema=RelationTypeSummary, include_raw=True, method="function_calling")
        # self.prompt = create_relation_summary_prompt(
        #     node_labels=node_labels, 
        #     legal_examples=legal_examples, 
        #     additional_instructions=additional_instructions
        # )
        
        # self.parser = JsonOutputParser(pydantic_object=RelationTypeSummary)
        
        # self.chain = self.prompt | self.llm | self.parser
        

    def generate_relations(self, relations:List[str], inputs: str) -> List[str]:  
        prompt = create_relation_summary_prompt(
            node_labels=self.node_labels,
            relation_labels=relations,
            legal_examples=legal_examples,
            additional_instructions=self.additional_instructions,
            extract_instructions=self.prompt_str,
            )
        
        if self.function_calling :
            chain = prompt | self.llm
            response = chain.invoke({"inputs": inputs})  
            print(response["parsed"])
            if response["parsed"]:
                return response["parsed"].relations
            else:
                return []

        
        else:
            parser = JsonOutputParser(pydantic_object=RelationTypeSummary)
            chain = prompt | self.llm | parser
            response = chain.invoke({"inputs": inputs})  
            print(response)
            return response["relations"]


system_summary_rela_str = "您是民法典知识图谱架构师，请严格遵循以下规则归纳关系类型：" +    \
        "【核心规则】"+ \
        "1. 抽象层级：提炼法律关系本质"+    \
        "2. 关系类型必须是 动词短语（≤6字）"+   \
        "3. 关系类型中禁止包含任何具体权利/义务或关系客体，即"+ \
        " 禁止生成：'享有名誉权' '承担违约责任' '设立抵押权'"+  \
        " 应该生成：'享有' '承担' '设立'"+  \
        "4. 可以合并相似法律行为"+  \
        "5. 忽略时间/数量等修饰词"

if __name__ == "__main__":

    # tongyi_llm = Tongyi(
    #     model="qwen-max",
    #     api_key="",
    #     top_p=1
    # )
    llm_local = ChatOpenAI(
        base_url="http://118.195.161.49:27984/v1",
        api_key="QcyDAgAEAAYExiABBj5BzIGCAE",
        model="qwen-local"
    )
    res_node = [
        "民事主体","法律行为","民事客体","民事义务","法人",
        "自然人","民事权利","法定要求","法律负责","法定期间",
        "非法人组织","法律事实","法律效果"
    ]
    generator = Relation_Generate(llm=llm_local, node_labels=res_node,function_calling=True, extract_prompt_str=system_summary_rela_str)

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
    print(generator.generate_relations(inputs=test_text,relations=[]))