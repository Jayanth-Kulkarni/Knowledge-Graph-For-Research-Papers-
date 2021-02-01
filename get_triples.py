import json

import numpy as np

total = 0
import os
from collections import defaultdict
from itertools import combinations
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from nltk import stem
lemmatizer = stem.WordNetLemmatizer()

with open("./keywords_cleaned/keywords_cleaned/all_cleaned_keywords_mythili", errors='ignore') as json_data:
    data = json.loads(json_data.read())

fullform_map = {
    "cs.AI": "Artificial Intelligence",
    "cs.CL": "Computation and Language",
    "cs.CC": "Computational Complexity",
    "cs.CE": "Computational Engineering, Finance, and Science",
    "cs.CG": "Computational Geometry",
    "cs.GT": "Computer Science and Game Theory",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.CY": "Computers and Society",
    "cs.CR": "Cryptography and Security",
    "cs.DS": "Data Structures and Algorithms",
    "cs.DB": "Databases",
    "cs.DL": "Digital Libraries",
    "cs.DM": "Discrete Mathematics",
    "cs.DC": "Distributed, Parallel, and Cluster Computing",
    "cs.ET": "Emerging Technologies",
    "cs.FL": "Formal Languages and Automata Theory",
    "cs.GL": "General Literature",
    "cs.GR": "Graphics",
    "cs.AR": "Hardware Architecture",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval",
    "cs.IT": "Information Theory",
    "cs.LO": "Logic in Computer Science",
    "cs.LG": "Machine Learning",
    "cs.MS": "Mathematical Software",
    "cs.MA": "Multiagent Systems",
    "cs.MM": "Multimedia",
    "cs.NI": "Networking and Internet Architecture",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.NA": "Numerical Analysis",
    "cs.OS": "Operating Systems",
    "cs.OH": "Other Computer Science",
    "cs.PF": "Performance",
    "cs.PL": "Programming Languages",
    "cs.RO": "Robotics",
    "cs.SI": "Social and Information Networks",
    "cs.SE": "Software Engineering",
    "cs.SD": "Sound",
    "cs.SC": "Symbolic Computation",
    "cs.SY": "Systems and Control"
}

data_file = []
path_server = '/home/kulka112/thesis_new/Bigram_MI/Arxiv_phrases/'
for r, d, f in os.walk(path_server):
    for file in f:
        if ".txt" in file:
            file1 =r+file
            # print(file1)
            # print(os.path.join(r,file))
            data_file.append(os.path.join(r,file))

def pre_process(l):
    # print(l)
    tokens = l.strip().lower().strip(',.:?!><').split()
    tokens = [lemmatizer.lemmatize(i).strip(',.:?!><()$%\\') for i in tokens]
    # print(tokens)
    return tokens

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def read_input(input_file):
    """This method reads the input file which is in gzip format"""
    for inp in input_file:
        logging.info("reading file {0}...this may take a while".format(inp))
        print(inp)
        with open(inp, 'r') as f:
            for i, line in enumerate(f):
                if (i % 1000 == 0):
                    logging.info("read {0} abstracts".format(i))
                # do some pre-processing and return a list of words for each review text
                # print("line = ",line)
                # print(gensim.utils.simple_preprocess(line))
                yield pre_process(line)


documents = list(read_input(data_file))

keyword_set = set()
all_labels = set()
all_papers = set()
with open("./Labels/CS_id_labels.txt", "r") as f:
    for line in f:
        line = line.split(" ")
        lab = line[1].replace("\t", "").replace("\n", "")
        keyword_set.add(lab)
        all_labels.add(lab)
    # all_papers.add(line[0].replace(":",".").replace(" ",""))
    # line[1] = line[1].replace("\t","").replace("\n","")
    # # print("line[0] = "+line[0]+"line[1] = "+line[1])
    # labels[line[0]] = line[1]

# with open("./keywords_cleaned/keywords_cleaned/keywords_mr_14666","r") as file:
#   for line in file:
#       total += int(line.split(":")[1])

# 1202.0331
keys_d = list(data.keys())
print(len(keys_d))
all_keywords = []


def print_it():
    count = 0
    for i in keys_d:
        # print(type(i))
        count += 1
        if data[i]["keywords"] == " " or data[i]["keywords"] == ":":
            continue
        l = data[i]["keywords"].split(",")
        l = [i for i in l if len(i) > 0]
        cou = []
        # print("l = ",l)
        for j in l:
            j = j.replace("keywords", "").replace("keyword", "").replace("-", "_")
            count = 0
            in_j = []
            flag = 0
            if j == " " or j == "":
                continue
            for inner in j:
                if inner == " " or inner == ":" or inner == "." or inner == "_" or inner == ";":
                    count += 1
                else:
                    in_j.append(count)
                    break
            cou.append([j, in_j])
        final = [i[0][i[1][0]:] for i in (cou)]
        final = [i.replace(" ", "_") for i in final]
        # removes s in the trailing character
        final = [i[:-1] if i[-1]=='s' else i for i in final]
        all_keywords.append([final, data[i]["Category"], i])


print_it()

keywords_cat = defaultdict(list)
for keywords in all_keywords:
    keywords_cat[keywords[1]].extend(keywords[0])

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

with open('data', 'wb') as fp:
    pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

x = []
y = []
keywords_dict = {}
for i in keywords_cat:
    x.append(i)
    y.append(len(set(keywords_cat[i])))
    cat = i
    print(i)
    cat = cat.replace("cs.","")
    keywords_dict[cat] = [lemmatizer.lemmatize(t).lower().strip(',.:?!><')  for t in keywords_cat[i]]


# series = pd.Series(keywords_dict)


# def unique(sequence):
#     seen = set()
#     unique =  [x for x in sequence if not (x in seen or seen.add(x))]
#     unique = [i for i in unique if i.count('_')==1]
#     unique_present = []
#     for i in unique:
#         count = sequence.count(i)
#         unique_present.append((i,count))
#     print("len(unique) = ",len(unique))
#     print("len(unique_present) = ",len(unique_present))
#     return unique_present
# # series.to_csv("all_keys") 
# import csv
# import collections
# all_list = []
# all_list_without_count = []
# length = []
# with open('all_keys_list_sorted.csv', 'w', newline="") as csv_file:  
#     writer = csv.writer(csv_file)
#     for key, value in keywords_dict.items():
#         counts = collections.Counter(value)
#         new_list = sorted(value, key=lambda x: -counts[x])
#         new_list = unique(new_list)
#         keyword_only = [i[0] for i in new_list]
#         all_list.append(new_list)
#         # all_list_without_count.append(keyword_only)
#         length.append(len(new_list))
#         writer.writerow([key, keyword_only])
# keywords_df = pd.DataFrame(all_list)
# keywords_df = keywords_df.transpose()
# keywords_df.columns = keywords_dict.keys()
# keywords_df.to_csv("arxiv_keywords_phrase.csv")


# df = pd.DataFrame(list(zip(keywords_dict.keys(), length)), 
#                columns =['Topic', 'Keywords_num']) 

# df.drop("cmp")
# df.sort_values('Keywords_num', inplace=True)
# df.reset_index(inplace=True)

# # Draw plot
# fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
# ax.vlines(x=df.index, ymin=0, ymax=df.Keywords_num, color='firebrick', alpha=0.7, linewidth=1)
# ax.scatter(x=df.index, y=df.Keywords_num, s=80, color='firebrick', alpha=0.7)

# # Title, Label, Ticks and Ylim
# ax.set_title('Number of Unique Keywords extracted per category', fontdict={'size':22})
# ax.set_ylabel('Number of Unique Keywords')
# ax.set_xticks(df.index)
# ax.set_xticklabels(df.Topic.str.upper(), rotation=60, fontdict={'horizontalalignment': 'right', 'size':12})
# # ax.set_ylim(0, 30)

# # Annotate
# for row in df.itertuples():
#     ax.text(row.Index, row.Keywords_num+.5, s=round(row.Keywords_num, 2), rotation=75, horizontalalignment= 'center', verticalalignment='bottom', fontsize=16)

# plt.savefig("Keywords_per_cat.jpg")
# plt.show()


entity_to_id = {}
relation_to_id = {}
paper_set = set()
for key, cat, num in all_keywords:
    for k in key:
        keyword_set.add(k)
    paper_set.add(num)

for i in fullform_map.keys():
    keyword_set.add(i)

keyword_set.add("computer_science")
keyword_set = keyword_set.union(paper_set)
for idx, x in enumerate(keyword_set):
    entity_to_id[x] = idx

reference_list = []
with open("./ArXiv_data/ArXivref.txt","r") as file:
    for idx,line in enumerate(file):
        print(idx)
        inner = []
        line = line.split()
        for i in line:
            j = i.replace(":",".")
            if j in paper_set:
                inner.append(j)
        reference_list.append(inner)

with open("./ArXiv_data/ArXivref_CS.txt","w") as file:
    for i in reference_list:
        if len(i)>1:
            for j in i:
                file.write(str(j)+"\t")
            file.write("\n")


reference = []
with open("./ArXiv_data/ArXivref_CS.txt", "r") as file:
    for i in file:
        reference.append(i)

relation_to_id["keyword_in"] = 0
relation_to_id["same_paper"] = 1
relation_to_id["papertopic"] = 2
relation_to_id["subtopic"] = 3
relation_to_id["reference"] = 4
relation_to_id["paper_keyword"] = 5

triple = []


def get_triple(all_key):
    for key, cat, num in all_key:
        all_comb = combinations(key, 2)
        for i, j in all_comb:
            triple.append([i, j, "same_paper"])
        for k in key:
            triple.append([k, cat, "keyword_in"])
            triple.append([num, k, "paper_keyword"])
        triple.append([num, cat, "papertopic"])
    for i in all_labels:
        triple.append(["computer_science", i, "subtopic"])
    for i in reference:
        i = i.split()
        for j in i[1:]:
            triple.append([i[0], j, "reference"])


triple_id = []


def get_triple_id(all_key):
    for key, cat, num in all_key:
        all_comb = combinations(key, 2)
        for i, j in all_comb:
            triple_id.append([entity_to_id[i], entity_to_id[j], relation_to_id["same_paper"]])
        for k in key:
            triple_id.append([entity_to_id[k], entity_to_id[cat], relation_to_id["keyword_in"]])
            triple_id.append([entity_to_id[num], entity_to_id[k], relation_to_id["paper_keyword"]])
        triple_id.append([entity_to_id[num], entity_to_id[cat], relation_to_id["papertopic"]])
    for i in all_labels:
        triple_id.append([entity_to_id["computer_science"], entity_to_id[i], relation_to_id["subtopic"]])
    for i in reference:
        i = i.split()
        for j in i[1:]:
            triple_id.append([entity_to_id[i[0]], entity_to_id[j], relation_to_id["reference"]])


get_triple(all_keywords)
get_triple_id(all_keywords)

print(type(triple_id))
print(triple_id[0])

triple_id_df = pd.DataFrame(triple_id)
triple_id_df.columns = ["Entity 1", "Entity 2", "Relation"]
print(triple_id_df.head())

validate, train, test = np.split(triple_id_df.sample(frac=1),
                                 [int(.1 * len(triple_id_df)), int(.9 * len(triple_id_df))])
print(train.shape[0])
print(test.shape[0])
print(validate.shape[0])
np.savetxt('./ArXiv_data/train2id.txt',train.values,header=str(train.shape[0]),fmt='%s')
np.savetxt('./ArXiv_data/test2id.txt',test.values,header= str(test.shape[0]),fmt='%s')
np.savetxt('./ArXiv_data/valid2id.txt',validate.values,header=str(validate.shape[0]),fmt='%s')

with open("./ArXiv_data/relation2id.txt","w") as file:
    file.write(str(len(relation_to_id.keys()))+"\n")
    for i in relation_to_id.keys():
        file.write(i+"\t"+str(relation_to_id[i])+"\n")

with open("./ArXiv_data/entity2id.txt","w") as file:
    file.write(str(len(entity_to_id.keys()))+"\n")
    for i in entity_to_id.keys():
        file.write(i+"\t"+str(entity_to_id[i])+"\n")

with open("./ArXiv_data/triple.txt","w") as file:
    file.write(str(len(triple))+"\n")
    for i in triple:
        for j in i:
            file.write(j+"\t")
        file.write("\n")

with open("./ArXiv_data/triple2id.txt","w") as file:
    file.write(str(len(triple_id))+"\n")
    for i in triple_id:
        for j in i:
            file.write(str(j)+"\t")
        file.write("\n")


'''
stats
'''


# Prepare Data
df = pd.DataFrame(list(zip(x, y)), 
               columns =['Topic', 'Keywords_num']) 

df.sort_values('Keywords_num', inplace=True)
df.reset_index(inplace=True)

# Draw plot
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.vlines(x=df.index, ymin=0, ymax=df.Keywords_num, color='firebrick', alpha=0.7, linewidth=1)
ax.scatter(x=df.index, y=df.Keywords_num, s=80, color='firebrick', alpha=0.7)

# Title, Label, Ticks and Ylim
ax.set_title('Number of Unique Keywords extracted per category', fontdict={'size':22})
ax.set_ylabel('Number of Unique Keywords')
ax.set_xticks(df.index)
ax.set_xticklabels(df.Topic.str.upper(), rotation=60, fontdict={'horizontalalignment': 'right', 'size':12})
# ax.set_ylim(0, 30)

# Annotate
for row in df.itertuples():
    ax.text(row.Index, row.Keywords_num+.5, s=round(row.Keywords_num, 2), rotation=75, horizontalalignment= 'center', verticalalignment='bottom', fontsize=16)

plt.savefig("Keywords_per_cat.jpg")
plt.show()

paper_cat = defaultdict(list)
for keywords in all_keywords:
    paper_cat[keywords[1]].extend(keywords[2])


x = []
y = []
for i in paper_cat:
    # print(fullform_map[i] + " :" + str(len(paper_cat[i])))
    x.append(i)
    y.append(len(paper_cat[i]))


df = pd.DataFrame(list(zip(x, y)), 
               columns =['Topic', 'Paper_num']) 

df.sort_values('Paper_num', inplace=True)
df.reset_index(inplace=True)
df = df[df.Topic!="cmp-lg"]
topics = df.Topic
topics = [fullform_map[i] for i in topics]
df["Names"] = topics
df.to_latex("num_papers")

# # Draw plot
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.vlines(x=df.index, ymin=0, ymax=df.Paper_num, color='firebrick', alpha=0.7, linewidth=1)
ax.scatter(x=df.index, y=df.Paper_num, s=80, color='firebrick', alpha=0.7)

# Title, Label, Ticks and Ylim
ax.set_title('Number of Papers per category', fontdict={'size':22})
ax.set_ylabel('Number of Papers')
ax.set_xticks(df.index)
ax.set_xticklabels(df.Topic.str.upper(), rotation=60, fontdict={'horizontalalignment': 'right', 'size':12})
# ax.set_ylim(0, 30)

# Annotate
for row in df.itertuples():
    ax.text(row.Index, row.Paper_num+.5, s=round(row.Paper_num, 2), rotation=75, horizontalalignment= 'center', verticalalignment='bottom', fontsize=16)

plt.savefig("Papers_per_cat.jpg")
plt.show()

# x_ax = range(0, len(x))
# plt.figure()
# plt.bar(x_ax, y, align="center")
# plt.xticks(x_ax, x, rotation='vertical')
# plt.title("Papers per Category")
# plt.savefig("Papers_per_category.png")
# plt.show()

# # networkx stuff

# G = nx.from_pandas_edgelist(triple_id_df, "Entity 1", "Entity 2", "Relation", create_using=nx.DiGraph())
# # nx.draw(G)
# # plt.show()

# N, K = G.order(), G.size()
# avg_deg = float(K) / N
# print("Nodes: ", N)
# print("Edges: ", K)
# print("mean degree: ", avg_deg)

# # print(page_rank)
# page_rank = nx.pagerank(G)
# deg_dist = nx.degree_histogram(G)
# plt.loglog(range(0, len(deg_dist)), deg_dist, 'o')
# plt.xlabel('degree')
# plt.ylabel('frequency')
# plt.title("Computer Science Knowledge graph Degree Distribution")
# plt.savefig("CS_KG_Degreedist.png")
# plt.show()
# plt.close()


# in_degrees = G.in_degree() # dictionary node:degree
# in_degrees = dict(in_degrees)
# in_values = sorted(set(in_degrees.values()))
# in_hist = [list(in_degrees.values()).count(x) for x in in_values]
# out_degrees = G.out_degree() # dictionary node:degree
# out_degrees = dict(out_degrees)
# out_values = sorted(set(out_degrees.values()))
# out_hist = [list(out_degrees.values()).count(x) for x in out_values]
# plt.figure()
# plt.loglog(in_values,in_hist,'ro-') # in-degree
# plt.loglog(out_values,out_hist,'bv-') # out-degree
# plt.legend(['In-degree','Out-degree'])
# plt.xlabel('Degree')
# plt.ylabel('Number of nodes')
# plt.title('Computer Science Knowledge Graph')
# plt.savefig('Computer_Science_Knowledge_Graph.png')
# plt.close()


# # Clustering coefficient of all nodes (in a dictionary)
# # G_ud = G.to_undirected()
# # # Clustering coefficient of node 0
# # print(nx.clustering(G_ud, 0))
# # clust_coefficients = nx.clustering(G_ud)
# # # Average clustering coefficient
# # ccs = nx.clustering(G_ud)
# # avg_clust = sum(ccs.values()) / len(ccs)
# # print(avg_clust)

# # triple_df = pd.DataFrame(triple)
# # triple_df.columns = ["Entity 1","Entity 2","Relation"]
# # triple_df["Relation_2"] = triple_id_df["Relation"]
# # print(triple_df.head())
# # G2=nx.from_pandas_edgelist(triple_df, "Entity 1", "Entity 2", "Relation_2",create_using=nx.DiGraph())
# # g2_deg = dict(G2.degree())
# # print(sorted(g2_deg.items(), key=lambda x:x[1], reverse=True)[:5])

# # Centrality measures
# # degree_centrality = nx.degree_centrality(G)
# # plt.plot(degree_centrality.values())
# # plt.show()