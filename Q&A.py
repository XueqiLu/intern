#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import jieba
import jieba.analyse
import jieba.posseg
import jieba.posseg as pseg
from neo4j import GraphDatabase
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

uri = "uri"
username = "neo4j"
password = "password"
graph = GraphDatabase.driver(uri, auth=(username, password))
jieba.Tokenizer()
jieba.load_userdict('/content/vocab.txt')
jieba.load_userdict('/content/symptom_vocab.txt')
jieba.load_userdict('/content/disease_vocab.txt')
jieba.load_userdict('/content/complications_vocab.txt')
jieba.load_userdict('/content/alias_vocab.txt')

# discover key words
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = set(line.strip() for line in file)
    return stopwords
stopwords_filepath = '/content/stop_words.utf8'
stopwords = load_stopwords(stopwords_filepath)
jieba.analyse.set_stop_words(stopwords_filepath)

def parts_of_speech(sentence):
    dirs={}
    for word,flag in pseg.cut(sentence):
        if word in entities:
            dirs[word]=flag
    return dirs

def key_word(sentence):
    entities1=jieba.analyse.extract_tags(sentence, topK=20, withWeight=False)
    lists_of_t=jieba.analyse.textrank(sentence, topK=20, withWeight=True)
    entities2 = [t[0] for t in lists_of_t]
    entities = list(set(entities1+entities2))
    return entities

# find labels
def fetch_labels(tx, name):
        query = "MATCH (n {name: $name}) RETURN labels(n) AS labels"
        result = tx.run(query, name=name)
        record = result.single()
        return record["labels"] if record else None

def discover_labels(entities):
    label_dic={}
    for entity in entities:
        with graph.session() as session:
            labels = session.execute_read(fetch_labels, entity)
            if labels!=None:
                if len(labels)>1:
                    return 'situation 1'
                elif labels!=None and labels[0] not in label_dic:
                    label_dic[labels[0]]=[]
                label_dic[labels[0]].append(entity)
    if not label_dic:
        return 'situation 2'
    return label_dic

#end nodes find start nodes
def fetch_related_nodes(tx, end_label, relationship, start_label, end_node_name):
    query = f"""MATCH (start:{start_label})-[r:{relationship}]->(end:{end_label}{{name: $end_node_name}}) RETURN start.name AS related_node_name"""
    result = tx.run(query, end_node_name=end_node_name)
    related_node_names = [record["related_node_name"] for record in result]
    return related_node_names if related_node_names else None

def get_related_nodes(driver, end_label, relationship, start_label, end_node_name):
    with driver.session() as session:
        return session.execute_read(fetch_related_nodes, end_label, relationship, start_label, end_node_name)

def fetch_property_value(tx, label, name_property, name_value, property_key):
    query = f"""MATCH (n:{label}) WHERE n.{name_property}='{name_value}' RETURN n.{property_key} AS property_values"""
    result = tx.run(query, name_value=name_value)
    property_values =[record["property_values"] for record in result]
    return property_values if property_values else None

def get_property_value(driver, label, name_property, name_value, property_key):
    with driver.session() as session:
        return session.execute_read(fetch_property_value, label, name_property, name_value, property_key)

# nodes name/attributes+their labels to find corresponding disease name
# attributes=['age','infection','insurance','checklist','treatment','period','rate','money']
def disease_search(key_labels):
    nodes=['Disease','alias','part','department','symptom','drug']
    relationships=['疾病并发症','病症别名','病痛的部位','疾病所属部门','疾病症状','疾病所需药物']
    keys_list = list(key_labels.keys())
    disease=[]
    for key in keys_list:
        if key == 'Disease':
            for value in key_labels[key]:
                if value not in disease:
                    disease.append(value)
        else:
            #有可能一个药物对应多个疾病等等
            for value in key_labels[key]:
                index_of_key = nodes.index(key)
                corr_rel = relationships[index_of_key]
                re=get_related_nodes(graph, key, corr_rel, 'Disease', value)
                for d in re:
                    if d not in disease:
                        disease.append(d)
    return disease

def find_intention(sentence):
    model_dir = "facebook/bart-large-mnli"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

    candidate_labels = ["部门", "身体部位","传染性","人群","保险","药物","别名","花费","治愈率","症状","治疗方案","检查项目","并发症","时间","病症名称"]
    intention=['department','part','infection','age','insurance','drug','alias','money','rate','symptom','treatment','checklist','complication','period','Disease']

    result = classifier(sentence, candidate_labels)
    i=candidate_labels.index(result['labels'][0])
    print(result['labels'][0])
    return intention[i]

def fetch_end_nodes(tx, start_label, relationship, end_label, start_node_name):
    query = f"""MATCH (start:{start_label} {{name: $start_node_name}})-[r:{relationship}]->(end:{end_label}) RETURN end.name AS related_node_name"""
    result = tx.run(query, start_node_name=start_node_name)
    related_node_names = [record["related_node_name"] for record in result]
    return related_node_names if related_node_names else None

def get_end_nodes(driver, start_label, relationship, end_label, start_node_name):
    with driver.session() as session:
        return session.execute_read(fetch_end_nodes, start_label, relationship, end_label, start_node_name)


def find_final_answer(given_entity, want_to_know):
    #nodes
    if want_to_know=='Disease':
        return "监测到您提问的疾病是"+given_entity
    elif want_to_know=='alias':
        answer=get_end_nodes(graph, 'Disease', '病症别名', want_to_know, given_entity)
        if not answer:
            return '抱歉，答案缺失'
        reply = "、".join(answer)+"。"
        return "监测到您提问的疾病是"+given_entity+'。'+given_entity+'的别名是'+reply
    elif want_to_know=='part':
        answer=get_end_nodes(graph, 'Disease', '病痛的部位', want_to_know, given_entity)
        if not answer:
            return '抱歉，答案缺失'
        reply = "、".join(answer)+"。"
        return "监测到您提问的疾病是"+given_entity+'。'+given_entity+'病痛的部位是'+reply
    elif want_to_know=='department':
        answer=get_end_nodes(graph, 'Disease', '疾病所属部门', want_to_know, given_entity)
        if not answer:
            return '抱歉，答案缺失'
        reply = "、".join(answer)
        return "监测到您提问的疾病是"+given_entity+'。'+ given_entity+'应该去'+reply+"挂号。"
    elif want_to_know=='symptom':
        answer=get_end_nodes(graph, 'Disease', '疾病症状', want_to_know, given_entity)
        if not answer:
            return '抱歉，答案缺失'
        reply = "、".join(answer)+"。"
        return "监测到您提问的疾病是"+given_entity+'。'+given_entity+'的症状有'+reply
    elif want_to_know=='drug':
        answer=get_end_nodes(graph, 'Disease', '疾病所需药物', want_to_know, given_entity)
        if not answer:
            return '抱歉，答案缺失'
        reply = "、".join(answer)+"。"
        return "监测到您提问的疾病是"+given_entity+'。'+given_entity+'的治疗药物是'+reply
    elif want_to_know=='complication':
        answer=get_end_nodes(graph, 'Disease', '疾病并发症', 'Disease', given_entity)
        if not answer:
            return '抱歉，答案缺失'
        reply = "、".join(answer)+"。"
        return "监测到您提问的疾病是"+given_entity+'。'+given_entity+'的并发症是'+reply
    #attributes
    elif want_to_know=='age':
        answer=get_property_value(graph, 'Disease', 'name', given_entity, want_to_know)
        if not answer:
            return '抱歉，答案缺失'
        reply = "、".join(answer)
        return "监测到您提问的疾病是"+given_entity +'。'+reply+"是得"+given_entity+"的高风险人群"
    elif want_to_know=='infection':
        answer=get_property_value(graph, 'Disease', 'name', given_entity, want_to_know)
        if not answer:
            return '抱歉，答案缺失'
        reply = "、".join(answer)+'。'
        return "监测到您提问的疾病是"+given_entity + given_entity+reply
    elif want_to_know=='insurance':
        answer=get_property_value(graph, 'Disease', 'name', given_entity, want_to_know)
        if not answer:
            return '抱歉，答案缺失'
        reply = "、".join(answer)+'。'
        return "监测到您提问的疾病是"+given_entity + given_entity+reply
    elif want_to_know=='checklist':
        answer=get_property_value(graph, 'Disease', 'name', given_entity, want_to_know)
        if not answer:
            return '抱歉，答案缺失'
        words = answer[0].split()
        reply = "、".join(words)+'。'
        return "监测到您提问的疾病是"+given_entity + given_entity+"需要检查"+reply
    elif want_to_know=='treatment':
        answer=get_property_value(graph, 'Disease', 'name', given_entity, want_to_know)
        if not answer:
            return '抱歉，答案缺失'
        reply = "、".join(answer)+'。'
        return "监测到您提问的疾病是"+given_entity+given_entity+'的治疗方案是'+reply
    elif want_to_know=='period':
        answer=get_property_value(graph, 'Disease', 'name', given_entity, want_to_know)
        if not answer:
            return '抱歉，答案缺失'
        reply = "、".join(answer)+'。'
        return "监测到您提问的疾病是"+given_entity+given_entity+'的治疗时长是'+reply
    elif want_to_know=='rate':
        answer=get_property_value(graph, 'Disease', 'name', given_entity, want_to_know)
        if not answer:
            return '抱歉，答案缺失'
        reply = "、".join(answer)+'。'
        return "监测到您提问的疾病是"+given_entity+given_entity+"的治愈率是"+reply
    else:
        answer=get_property_value(graph, 'Disease', 'name', given_entity, want_to_know)
        if not answer:
            return '抱歉，答案缺失'
        reply = "、".join(answer)+'。'
        return "监测到您提问的疾病是"+given_entity+'治疗'+given_entity+'的花费是'+reply


def get_answer(sentence):
    entities=key_word(sentence)
    key_labels=discover_labels(entities)
    if key_labels=='situation 1':
        return '检测到您的提问的关键词属于多个标签，请详细说明：'
    elif key_labels=='situation 2':
        return '无法检测到医疗关键词，请详细说明：'
    else:
        t_disease=disease_search(key_labels)
        if len(t_disease)>1:
            return '检测到您的问题涉及多个疾病，分别是'+t_disease.join("，")+'，请具体说明您想了解的疾病'
        else:
            intention=find_intention(sentence)
            reply=find_final_answer(t_disease[0], intention)
            return reply

def main():
    # Define the Neo4j connection details

    while True:
        question = input("请问您有什么问题？")
        if question.lower() in ['退出']:
            print("再见！！！")
            break
        answer = get_answer(question)
        print(answer)

if __name__ == "__main__":
    main()

