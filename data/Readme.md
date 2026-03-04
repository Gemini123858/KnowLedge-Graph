#### 数据清单
* 所包含的数据有：graph_doc_raw.pkl  --  通过LLM抽取生成的知识图谱数据
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
