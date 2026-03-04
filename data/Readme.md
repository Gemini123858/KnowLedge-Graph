#### 数据清单
* 所包含的数据有：
* graph_doc_raw.pkl  --  通过LLM抽取生成的知识图谱数据
* ```python
  # 打开方式
  with open("xxx") as file:
    import pickle
    graph_doc = pickle.load(file) # GraphDocument
  ```
* 图数据可视化：
* 拉取一个neo4j的容器，启动服务后可在其中写入数据进而有可视化展示（具体可见compose.yml)
* 启用服务后运行：
* ```python
  # url = "neo4j://localhost:7688"
  # username = "neo4j"
  # password = "wiki_pass"
  # graph = Neo4jGraph(
  #                     url=url,
  #                     username=username,
  #                     password=password
  #                 )
  # graph.query("MATCH (n) DETACH DELETE n")
  # graph.add_graph_documents(graph_doc)
  # print("图数据已成功添加到 Neo4j 数据库中。")
  ```

* all_models_node_similarity_comparison.xlsx ---- 各个模型对于两个节点相似度的计算值
* 其中："cos_similarity" 为decoder-only模型Qwen2.5-7B + prompt引导利用输出的最后层的最后一个token的向量进行计算
*   "gpt_aim_word_cos_similarity" 为decoder-only模型Qwen2.5-7B 在特定层（10-19）上针对key word的token进行聚合得到的向量进行计算
*   "deberta_base_cos_similarity" "deberta_large_cos_similarity" "modernbert_base_cos_similarity" "modernbert_large_cos_similarity" "googlebert_cos_similarity" 均为几款bert模型在特定层上针对key word的token进行聚合得到的向量进行计算
*   * all_models_node_similarity_comparison.pkl ---- 上述数据的pkl

* 在Qwen2.5-7B模型上利用Salesforce/wikitext中wikitext-2-raw-v1数据为样本进行了ID值的计算（Intrinsic Dimension）ID较低的层可能表示更抽象、更压缩的特征，而ID较高的层可能表示更丰富、更细粒度的特征。实验结果见 "qwen25_wikitext_lines_ids.pkl" 与 "qwen25_wikitext_lines_ids.png"

* all_models_merge_decision_comparison.pkl --- 通过询问LLM的方式获得的几款不同LLM的合并意见，True为可以合并
* all_models_merge_decision_comparison.xlsx --- 上述数据的excel

* 
