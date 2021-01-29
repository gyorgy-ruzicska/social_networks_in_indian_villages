#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:51:09 2019

@author: rgyuri
"""

#Social Networks 1 - Project - Individual level analysis - Tiedness

# Import the libraries
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

#Import the adjacency matrices from each village

folder = "/Users/rgyuri/Desktop/CEU/CEU_Fourth/Social_Networks/Final_Project"

list_of_names=["borrowmoney", "giveadvice", "helpdecision", "keroricecome", \
               "keroricego", "lendmoney", "medic", "nonrel", "rel", "templecompany", \
               "visitcome", "visitgo"]

for i in list_of_names:
    locals()["list_of_" + str(i)] = []
    locals()["nx_list_of_" + str(i)] = []
    for j in range(1,78):
        try:
            data=pd.read_csv(folder+"/Microfinance_data/Data/1. Network Data/Adjacency Matrices/adj_{}_vilno_{}.csv".format(i,j), header=None)
            index_list=[str(j).zfill(2)+str(k).zfill(4) for k in range(1, data.shape[0]+1)]
            data.insert(0, "New_index", index_list)
            data=data.set_index('New_index')
            data.columns=index_list
            data2=nx.from_pandas_adjacency(data)
            locals()["list_of_" + str(i)].append(data)
            locals()["nx_list_of_" + str(i)].append(data2)
        except:
            pass

#Sum up the different layers to obtain weighted matrices

weighted_matrices=[]
for j in range(0,75):
    for i in list_of_names:
        if i==list_of_names[0]:
            df_aux=locals()["list_of_" + str(i)][j]
        else:
            df_aux=df_aux.add(locals()["list_of_" + str(i)][j], fill_value=0)
    weighted_matrices.append(df_aux)

#Generate classification for 'tiedness' based on the number of weak/strong ties

for i in range(0,len(weighted_matrices)):
    if i==0:
        df1=pd.DataFrame((weighted_matrices[i]>4).sum(axis=1), columns=["Strong"])
        df2=pd.DataFrame(((weighted_matrices[i]<=4) & (weighted_matrices[i]>0)).sum(axis=1), columns=["Weak"])
        df3=pd.DataFrame(weighted_matrices[i].sum(axis=1), columns=["Neighbour_layers"])
        individual_ties=df1.merge(df2, left_on='New_index', right_on='New_index')
        individual_ties=individual_ties.merge(df3, left_on='New_index', right_on='New_index')
    else:
        df1=pd.DataFrame((weighted_matrices[i]>4).sum(axis=1), columns=["Strong"])
        df2=pd.DataFrame(((weighted_matrices[i]<=4) & (weighted_matrices[i]>0)).sum(axis=1), columns=["Weak"])
        df3=pd.DataFrame(weighted_matrices[i].sum(axis=1), columns=["Neighbour_layers"])
        df4=df1.merge(df2, left_on='New_index', right_on='New_index')
        df4=df4.merge(df3, left_on='New_index', right_on='New_index')
        individual_ties=pd.concat([individual_ties,df4])

individual_ties["Tiedness"]=individual_ties["Strong"]>individual_ties["Weak"]
individual_ties["Tiedness"]=individual_ties["Tiedness"].astype(int)

individual_ties["Max_neighbour_layers"]=12*(individual_ties["Strong"]+individual_ties["Weak"])
individual_ties["Neighbour_layers/Max_neighbour_layers"]=individual_ties["Neighbour_layers"]/individual_ties["Max_neighbour_layers"]

individual_ties["Strong/Weak"]=individual_ties['Strong']/individual_ties['Weak']


# Create a network object from the array of the adjacency matrices with weighted edges

weighted_networks=[]
for i in weighted_matrices:
    weighted_network=nx.from_pandas_adjacency(i, create_using=nx.MultiGraph())
    weighted_networks.append(weighted_network)

# Create an edgelist in pandas from the network object

weighted_edgelist=[]
for i in weighted_networks:
    weighted_elist=nx.convert_matrix.to_pandas_edgelist(i)
    weighted_edgelist.append(weighted_elist)

# Create an nd array (matrix) from pandas as it is easier to iterate in it

mat_weighted_edgelist=[]
for i in weighted_edgelist:
    mat_weighted_elist=i.values
    mat_weighted_edgelist.append(mat_weighted_elist)


# Create an empty list for the overlaps
overlaps=[]

for k in range(0,len(mat_weighted_edgelist)):
    overlap_aux=[]
    for e in range(mat_weighted_edgelist[k].shape[0]):

        # iterator in the source and the target
        i=mat_weighted_edgelist[k][e,0]
        j=mat_weighted_edgelist[k][e,1]

        # list of the neighbours for source and target: can be defined from the network object
        i_neig=list(weighted_networks[k].neighbors(i))
        j_neig=list(weighted_networks[k].neighbors(j))

        # calculating the union and the intersection of neighbours of source and target
        union_list = list(set(i_neig).union(set(j_neig)))
        intersect_list = list(set(i_neig).intersection(set(j_neig)))

        # calculate the overlap between the neighbours of source and target
        neigh_overlap=len(intersect_list)/len(union_list)

        # add the value of the overlap to the list
        overlap_aux.append(neigh_overlap)
    overlaps.append(overlap_aux)


# Add the list with all the overlaps to the pandas edgelist

for k in range(0,len(weighted_edgelist)):
    weighted_edgelist[k]['neigh_ovr']=overlaps[k]


# Generate correlation between number of edges and  overlap by person


for i in range(0,len(weighted_edgelist)):
    corr=weighted_edgelist[i].groupby('source')[['weight','neigh_ovr']].corr().iloc[0::2,-1]
    corr=corr.reset_index()
    if i==0:
        df_of_correlations=pd.DataFrame(corr)
    else:
        df_of_correlations=pd.concat([df_of_correlations, corr])

df_of_correlations=df_of_correlations.drop(['level_1'], axis=1)
df_of_correlations=df_of_correlations.rename(columns={"neigh_ovr": "Granovetter_corr"})

# Obtain centrality measures


def centralityCreator(n_weighted, borrowmoney, giveadvice, helpdecision, keroricecome, \
               keroricego, lendmoney, medic, nonrel, rel, templecompany, \
               visitcome, visitgo):
    degree_centrality = nx.degree_centrality(n_weighted)
    closeness_centrality = nx.closeness_centrality(n_weighted)
    #betweenness_centrality = nx.betweenness_centrality(n_weighted)
    degree_centrality_borrowmoney = nx.degree_centrality(borrowmoney)
    degree_centrality_giveadvice = nx.degree_centrality(giveadvice)
    degree_centrality_helpdecision = nx.degree_centrality(helpdecision)
    degree_centrality_keroricecome = nx.degree_centrality(keroricecome)
    degree_centrality_keroricego = nx.degree_centrality(keroricego)
    degree_centrality_lendmoney = nx.degree_centrality(lendmoney)
    degree_centrality_medic = nx.degree_centrality(medic)
    degree_centrality_nonrel = nx.degree_centrality(nonrel)
    degree_centrality_rel = nx.degree_centrality(rel)
    degree_centrality_templecompany = nx.degree_centrality(templecompany)
    degree_centrality_visitcome = nx.degree_centrality(visitcome)
    degree_centrality_visitgo = nx.degree_centrality(visitgo)
    list_of_dict=[degree_centrality, closeness_centrality,\
                  degree_centrality_borrowmoney, degree_centrality_giveadvice, degree_centrality_helpdecision, \
                  degree_centrality_keroricecome, degree_centrality_keroricego, degree_centrality_lendmoney, \
                  degree_centrality_medic, degree_centrality_nonrel, degree_centrality_rel, degree_centrality_templecompany, \
                  degree_centrality_visitcome, degree_centrality_visitgo]
    super_dict = {}
    for d in list_of_dict:
        for k, v in d.items():
            super_dict.setdefault(k, []).append(v)
    return super_dict

for i in range(0,len(weighted_networks)):
    dict2=centralityCreator(weighted_networks[i], nx_list_of_borrowmoney[i], nx_list_of_giveadvice[i], nx_list_of_helpdecision[i], nx_list_of_keroricecome[i], \
               nx_list_of_keroricego[i], nx_list_of_lendmoney[i], nx_list_of_medic[i], nx_list_of_nonrel[i], nx_list_of_rel[i], nx_list_of_templecompany[i], \
               nx_list_of_visitcome[i], nx_list_of_visitgo[i])
    if i==0:
        centrality_df=pd.DataFrame.from_dict(dict2, orient='index', columns=["Degree centrality", "Closeness centrality", "borrowmoney", "giveadvice", "helpdecision", "keroricecome", \
               "keroricego", "lendmoney", "medic", "nonrel", "rel", "templecompany", \
               "visitcome", "visitgo"])
    else:
        centrality_df=pd.concat([centrality_df, pd.DataFrame.from_dict(dict2, orient='index', columns=["Degree centrality", "Closeness centrality", "borrowmoney", "giveadvice", "helpdecision", "keroricecome", \
               "keroricego", "lendmoney", "medic", "nonrel", "rel", "templecompany", \
               "visitcome", "visitgo"])])

#Import individual level data and merge with household data

individual_characteristics=pd.read_csv(folder+"/Microfinance_data/Data/2. Demographics and Outcomes/individual_characteristics.csv", header=0 )
individual_characteristics=individual_characteristics.drop(columns=["pid"])
individual_characteristics["pid"]=individual_characteristics["village"].apply(lambda x: str(x).zfill(2))+individual_characteristics["adjmatrix_key"].apply(lambda x: str(x).zfill(4))

household_characteristics=pd.read_csv(folder+"/Microfinance_data/Data/2. Demographics and Outcomes/household_characteristics.csv", header=0 )
individual_characteristics=individual_characteristics.merge(household_characteristics, left_on='hhid', right_on='hhid')


#Merge four dataframes
individual_ties=individual_ties.reset_index()
centrality_df=centrality_df.reset_index()
individual_dataset=individual_characteristics.merge(individual_ties, left_on='pid', right_on='New_index')
individual_dataset=individual_dataset.merge(centrality_df, left_on='pid', right_on='index')
individual_dataset=individual_dataset.merge(df_of_correlations, left_on='pid', right_on='source')



"""
Regressors:
From individual leve dataset:
    resp_gend - gender
    resp_status - status
    age
    religion
    caste
    mothertongue
    speakother - speaks pther language
    educ - highest level of education
    villagenative - native to village
    workflag - did you work last week?
    shgparticipate - do you participate in an SHG or other savings group?
    savings - do you have savings account?
    electioncard - do you have an election card?
    rationcard - do you have a ration card?
From household level dataset:
    rooftype 1-5- dummies for different roof types
    room_no - number of rooms
    bed_no - number of beds
    electricity - is there electricity?
    latrine - type of latrine the hh has
    ownrent - owned/rented house
    leader - household contains a leader (see definition in paper)
From centrality measures:
    degree_centrality
    closeness_centrality

Base categories: hinduism, caste_dontknow, hindi language, latrine none, electricity no, house rented
"""

#First, convert yes/no questions to dummy and get dummies from categorical variables

dummy_list=["speakother", "villagenative", "workflag", "shgparticipate", "savings", \
            "electioncard", "rationcard"]

for name in dummy_list:
    individual_dataset[name] = individual_dataset[name].map({'Yes': 1, 'No': 0})

individual_dataset["resp_gend"] = individual_dataset["resp_gend"].map({1: 0, 2: 1})

individual_dataset["educ"] = individual_dataset["educ"].map({"1ST STANDARD": 1, "2ND STANDARD": 2, \
                   "3RD STANDARD": 3, "4TH STANDARD": 4, "5TH STANDARD": 5, "6TH STANDARD": 6, \
                   "7TH STANDARD": 7, "8TH STANDARD": 8, "9TH STANDARD": 9, "S.S.L.C.": 10, \
                   "1ST P.U.C.": 11, "2ND P.U.C.": 12, "UNCOMPLETED DEGREE": 13, "DEGREE OR ABOVE": 14, \
                   "OTHER DIPLOMA": 15, "NONE": 0})

categorical_list=["resp_status", "religion", "caste", "mothertongue", "electricity", "latrine", "ownrent"]

for name in categorical_list:
    locals()["dummies_" + str(name)]=pd.get_dummies(individual_dataset[name])
    locals()["dummies_" + str(name)].columns=[str(name)+"_"+str(col) for col in locals()["dummies_" + str(name)].columns]
    individual_dataset=individual_dataset.join(locals()["dummies_" + str(name)])

#Keep variables of interest
regression_dataset=individual_dataset[['village_x','resp_gend',
       'age', 'educ', 'villagenative', 'workflag', 'shgparticipate',  'savings',
       'electioncard', 'rationcard', 'rooftype1', 'rooftype2', 'rooftype3', 'rooftype4',
       'rooftype5', 'room_no', 'bed_no', 'leader', 'Strong', 'Weak',
       'Neighbour_layers/Max_neighbour_layers', 'Max_neighbour_layers', 'Neighbour_layers',
       'Strong/Weak', 'Tiedness' , 'Granovetter_corr',
       'Degree centrality', 'Closeness centrality',
       'borrowmoney', 'giveadvice', 'helpdecision',
       'keroricecome', 'keroricego', 'lendmoney', 'medic', 'nonrel', 'rel',
       'templecompany', 'visitcome', 'visitgo',
       'resp_status_Head of Household', 'resp_status_Other',
       'resp_status_Spouse of Head of Household', 'religion_CHRISTIANITY',
       'religion_HINDUISM', 'religion_ISLAM', 'caste_DO NOT KNOW',
       'caste_GENERAL', 'caste_OBC', 'caste_SCHEDULED CASTE',
       'caste_SCHEDULED TRIBE', 'mothertongue_HINDI', 'mothertongue_KANNADA',
       'mothertongue_MALAYALAM', 'mothertongue_MARATI', 'mothertongue_TAMIL',
       'mothertongue_TELUGU', 'mothertongue_URDU', 'latrine_Common',
       'electricity_No', 'electricity_Yes, Government', 'electricity_Yes, Private',
       'latrine_None', 'latrine_Owned', 'ownrent_0',
       'ownrent_6', 'ownrent_GIVEN BY GOVERNMENT', 'ownrent_LEASED',
       'ownrent_OWNED', 'ownrent_OWNED BUT SHARED', 'ownrent_RENTED']]

#Fill out not available data with zero
regression_dataset1=regression_dataset.fillna(0)


"""
No. 1 - independent variable is 'tiedness' measure
"""

regression_dataset1.to_csv(r'/Users/rgyuri/Desktop/CEU/CEU_Fourth/Social_Networks/Final_Project/reg1.csv')


"""
No. 2 - independent variable is ratio of strong and weak ties
"""

#Drop infinity

regression_dataset2=regression_dataset1.replace([np.inf, -np.inf], np.nan).dropna(subset=["Strong/Weak"], how="all")

regression_dataset2.to_csv(r'/Users/rgyuri/Desktop/CEU/CEU_Fourth/Social_Networks/Final_Project/reg2.csv')


plt.hist(regression_dataset2['Strong/Weak'], bins=20, range=[0,4])
# Add axis names
plt.ylabel('Number of individuals')
plt.xlabel('Ratio of strong and weak ties')
# Show graphic
plt.show()
"""
No. 3 - independent variable is ratio of actual number of ties and potential number of ties
"""


plt.hist(regression_dataset1['Neighbour_layers/Max_neighbour_layers'], bins=20, range=[0,1])
# Add axis names
plt.ylabel('Number of individuals')
plt.xlabel('Ratio of actual and maximum number of neighbour layers')
# Show graphic
plt.show()

"""
No. 4 - indepent variable is Granovetter's correlation
"""

regression_dataset3=regression_dataset.dropna(subset=["Granovetter_corr"], how="all")
regression_dataset3=regression_dataset3.fillna(0)

regression_dataset3.to_csv(r'/Users/rgyuri/Desktop/CEU/CEU_Fourth/Social_Networks/Final_Project/reg3.csv')

plt.hist(regression_dataset3['Granovetter_corr']
)
# Add axis names
plt.ylabel('Number of individuals')
plt.xlabel('Ganovetter correlation')
# Show graphic
plt.show()
