from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
from typing import Type, Optional, List, Union, Tuple, Dict
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage
from time import sleep

class EachDecision(BaseModel):
    node1: str = Field(..., description="第一个节点(引用原文的字符串)")
    node2: str = Field(..., description="第二个节点(引用原文的字符串)")
    can_merge: bool = Field(..., description="这两个节点是否可以合并，是则为True，否则为False")


class MergeDecision(BaseModel):
    results: List[EachDecision] = Field(..., description="每一对节点的合并决策结果列表")


class LLMNodeMerger:
    def __init__(self, 
                 llm:BaseLanguageModel,
                 function_calling:bool = False,
                 pydantic_object: Optional[Type[BaseModel]] = None,
                 function_calling_pydantic: Optional[Type[BaseModel]] = None):
        self.llm = llm
        self.function_calling = function_calling
        if function_calling:
            self.prompt = self.get_prompt(function_calling=True)
            structured_llm = llm.with_structured_output(
                schema=function_calling_pydantic, include_raw=True, method="function_calling")
            self.chain = self.prompt | structured_llm
        else:
            self.prompt = self.get_prompt(function_calling=False, without_function_call_pydantic_object=pydantic_object)
            parser = JsonOutputParser(pydantic_object=pydantic_object)
            self.chain = self.prompt | llm | parser

    def get_prompt(self, 
                   function_calling:bool=False, 
                   without_function_call_pydantic_object: Optional[Type[BaseModel]] = None):
        # 给LLM一些已经提取出来的节点以及它们的关系，让LLM判断这两个节点是否可以合并
        system_prompt_parts = [
            "你是一个知识图谱构建专家，负责判断图谱中的节点是否可以合并。",
            "当两个节点表示相同的实体或概念时，应将它们合并为一个节点。",
            "请根据节点本身以及节点的相关关系来判断它们是否可以合并。",
        ]
        system_prompt = "\n".join(system_prompt_parts)
        system_message = SystemMessage(content=system_prompt)

        human_prompt_parts = [
            "请根据以下若干组节点及其相关关系，判断它们是否可以合并为一个节点。",
            "{inputs}",                                                         # inputs:第一组：node1，rels1；node2，rels2；第二组：...
            "只回答能否合并，不要添加任何额外的信息。",
            "" if function_calling else "输出要求：{format_instructions}",
        ]

        if function_calling:
            human_prompt = PromptTemplate(
                template="\n".join(filter(None, human_prompt_parts)),
                input_variables=["inputs"],
                partial_variables={
                },
            )
        else:
            parser = JsonOutputParser(pydantic_object=without_function_call_pydantic_object)
            human_prompt = PromptTemplate(
                template="\n".join(filter(None, human_prompt_parts)),
                input_variables=["inputs"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions(),
                },
            )

        return ChatPromptTemplate.from_messages([
            system_message,
            HumanMessagePromptTemplate(prompt=human_prompt)
        ])
    
    """
    info:[
        {
            "node1": Node_str,
            "node2": Node_str,
            "relations1": List[Relationship_str],
            "relations2": List[Relationship_str],
        },
        ...
    ]
    results:[
        {
            "node1": Node_str,
            "node2": Node_str,
            "can_merge": bool,
        },
        ...
    ]
    """
    def merge_nodes(self, info:List[Dict[str, Union[str, List[str]]]], block_size:int=5)->List[Dict[str, Union[str, bool]]]:

        # 每block_size个一组进行判断
        results = []
        max_times = (len(info) + block_size - 1) // block_size
        for i in range(0, max_times):
            sleep(0.5)
            block = info[i*block_size : (i+1)*block_size]
            inputs = ""
            for j ,item in enumerate(block):
                node1_str = item["node1"]
                node2_str = item["node2"]
                relations1 = item.get("relations1", [])
                relations2 = item.get("relations2", [])
                inputs += f"第{j+1}组:\n"
                inputs += f"节点1: {node1_str}\n"
                inputs += "节点1的相关关系:\n"
                inputs += " ".join([f"- {rel}" for rel in relations1]) + "\n"
                inputs += f"节点2: {node2_str}\n"
                inputs += "节点2的相关关系:\n"
                inputs += " ".join([f"- {rel}" for rel in relations2]) + "\n"

            raw_shame = self.chain.invoke({"inputs": inputs})
            if self.function_calling:
                raise NotImplementedError
                pass
            else:
                print(f"LLM原始输出: {raw_shame}")
                for result in raw_shame["results"]:
                    results.append({
                        "node1": result["node1"],
                        "node2": result["node2"],
                        "can_merge": result["can_merge"],
                    })

        return results