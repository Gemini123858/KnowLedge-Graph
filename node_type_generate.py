from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage

class NodeTypeSummary(BaseModel):
    nodes_type: List[str] = Field(
        ...,
        description="从文本中抽象出新的节点类型集合",
    )

def create_node_type_extract_prompt(
                                    node_types: List[str], 
                                    legal_examples: Optional[List[dict]] = None,
                                    extract_prompt_str: str = ""
                                    ) -> ChatPromptTemplate:
    node_types_str = "、".join(node_types)
    system_prompt_parts = [
        extract_prompt_str,
    ]
    if legal_examples:
        examples_str = "\n".join(
            [f"文本: {ex['text']}\n节点类型: {ex['node_type']}\n解析: {ex['analysis']}" for ex in legal_examples]
        )
    system_prompt = "\n".join(filter(None, system_prompt_parts))

    human_prompt_parts = [
        "请从文本中抽取节点类型：",
        "{input}",
        "约束条件:",
        "1,已有节点类型{node_types_str}, 请勿重复抽取",
        "2,输出最具有普遍适用性且重要的几种新的类型，若已经基本充分则输出空",
        "输出格式：",
        "{format_instructions}"
    ]

    system_message = SystemMessage(content=system_prompt)
    parser = JsonOutputParser(pydantic_object=NodeTypeSummary)
    human_prompt = PromptTemplate(
        template="\n".join(filter(None, human_prompt_parts)),
        input_variables=["input"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "node_types_str": node_types_str,
        },
    )
    return ChatPromptTemplate.from_messages([
        system_message,
        HumanMessagePromptTemplate(prompt=human_prompt)
    ])


class NodeType_Generate:
    def __init__(self,
                 llm: ChatOpenAI,
                 extract_prompt_str: Optional[str] = "",
                 function_calling: bool = False):
        self.llm = llm
        self.extract_prompt_str = extract_prompt_str
        self.function_calling = function_calling
        if function_calling:
            self.llm = llm.with_structured_output(
                schema=NodeTypeSummary,
                include_raw=True,
                method="function_calling"
            )
    
    def generate_node_types(self, inputs: str, node_types: List[str]) -> List[str]:
        prompt = create_node_type_extract_prompt(
            node_types=node_types,
            extract_prompt_str=self.extract_prompt_str,
        )

        if self.function_calling:
            chain = prompt | self.llm
            response = chain.invoke({"input": inputs})
            if response['parsed'] and response['parsed'].nodes_type:
                return response['parsed'].nodes_type
            else:
                return []
        else:
            parser = JsonOutputParser(pydantic_object=NodeTypeSummary)
            chain = prompt | self.llm | parser
            response = chain.invoke({"input": inputs})
            print(response)
            return response["nodes_type"] if response and response.get("nodes_type") else []


                
                 
      