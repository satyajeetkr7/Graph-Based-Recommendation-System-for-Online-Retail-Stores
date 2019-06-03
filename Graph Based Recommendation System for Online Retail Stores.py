import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain

df = pd.read_excel('D:\Online Retail.xlsx', header = 0)
#print(df.columns)

print('dataset dimensions are:', df.shape)
df.describe(include = 'all')

df_sample = df.iloc[:200]
#print(df_sample.shape)
#print(df_sample.columns)
#print(df_sample.CustomerID) 

cleaned_retail = df_sample.loc[pd.isnull(df_sample.CustomerID) == False] #only keep rows that have customerID

#lookup table
item_lookup = cleaned_retail[['StockCode', 'Description']].drop_duplicates()
#print(item_lookup['StockCode'])
item_lookup['StockCode'] = item_lookup.StockCode.astype(str)

cleaned_retail['CustomerID'] = cleaned_retail.CustomerID.astype(int)#convert float customerID to int
#print(cleaned_retail)
print(cleaned_retail.columns)
cleaned_retail = cleaned_retail[['StockCode', 'Quantity', 'CustomerID']]
print(cleaned_retail.columns)
grouped_cleaned = cleaned_retail.groupby(['CustomerID', 'StockCode']).sum().reset_index()
#print(grouped_cleaned)

'''print(grouped_cleaned.Quantity.loc[grouped_cleaned.Quantity == 0])
grouped_cleaned.Quantity.loc[grouped_cleaned.Quantity == 0] = 1
print(grouped_cleaned.Quantity.loc[grouped_cleaned.Quantity == 1])'''

grouped_purchased = grouped_cleaned.query('Quantity > 0')
#print(grouped_purchased)

#number of products and number of customers in the reduced dataset 
no_products = len(grouped_purchased.StockCode.unique())
print('Number of products in dataset:', no_products)
no_customers = len(grouped_purchased.CustomerID.unique())
print('Number of customers in dataset:', no_customers)

#Turn raw data to pivot ('ratings' matrix)
ratings = grouped_purchased.pivot(index = 'CustomerID', columns='StockCode', values='Quantity').fillna(0).astype('int')
#print(ratings)

ratings_binary = ratings.copy()
ratings_binary[ratings_binary != 0] = 1
#print(ratings_binary)

products_integer = np.zeros((no_products,no_products))
#print(products_integer)

print('Counting how many times each pair of products has been purchased...')
for i in range(no_products):
    for j in range(no_products):
        if i != j:
            df_ij = ratings_binary.iloc[:,[i,j]] 
            sum_ij = df_ij.sum(axis=1)
            pairings_ij = len(sum_ij[sum_ij == 2]) #length of this array will tell how many times i and j have been bought together
            products_integer[i,j] = pairings_ij
            products_integer[j,i] = pairings_ij
#print(products_integer[10])            

print('Counting how many times each individual product has been purchased...')
times_purchased = products_integer.sum(axis = 1).astype(int)
#print(times_purchased)

print('Building weighted product matrix...')
products_weighted = np.zeros((no_products,no_products))
for i in range(no_products):
    for j in range(no_products):
        if (times_purchased[i]+times_purchased[j]) !=0:
            products_weighted[i,j] = (products_integer[i,j])/(times_purchased[i]+times_purchased[j])
#print(products_weighted[10])         

#print(ratings_binary.columns)
nodes_codes = np.array(ratings_binary.columns).astype('str')
#print(nodes_codes)

item_lookup_dict = pd.Series(item_lookup.Description.values,index=item_lookup.StockCode).to_dict()
#print(item_lookup_dict) 

nodes_labels = [item_lookup_dict[code] for code in nodes_codes] 
#print(nodes_labels)

#Create Graph object using the weighted product matrix as adjacency matrix
G = nx.from_numpy_matrix(products_weighted)
pos=nx.random_layout(G)
labels = {}
for idx, node in enumerate(G.nodes()):
    labels[node] = nodes_labels[idx]

nx.draw_networkx_nodes(G, pos , node_color="skyblue", node_size=30)
nx.draw_networkx_edges(G, pos,  edge_color='k', width= 0.3, alpha= 0.5)
nx.draw_networkx_labels(G, pos, labels, font_size=4)
plt.axis('off')
plt.show() # display

#Export graph to Gephi
H=nx.relabel_nodes(G,labels) #create a new graph with Description labels and save to Gephi for visualizations
nx.write_gexf(H, "products.gexf")

#Find communities of nodes (products)
partition = community_louvain.best_partition(G, resolution = 1)
values = list(partition.values())
#print(values)

#Check how many communities were created
print('Number of communities:', len(np.unique(values)))

#Create dataframe with product description and community id
products_communities = pd.DataFrame(nodes_labels, columns = ['product_description'])
products_communities['community_id'] = values

#Lets take a peek at community 1
products_communities[products_communities['community_id']==1].head(15)

#Lets now divide each element in products_weighted dataframe with the maximum of each row.
#This will normalize values in the row and we can perceive it as the possibility af a customer also buying
#product in column j after showing interest for the product in row i

#Turn into dataframe
products_weighted_pd = pd.DataFrame(products_weighted, columns = nodes_labels)
#print(products_weighted_pd)
products_weighted_pd.set_index(products_weighted_pd.columns, 'product', inplace=True)

products_prob = products_weighted_pd.divide(products_weighted_pd.max(axis = 1), axis = 0)
#print(products_prob)

#Now lets select a hypothetical basket of goods (one or more products) that a customer has already purchased or
#shown an interest for by clicking on an add or something, and then suggest him relative ones
basket = ['RECIPE BOX WITH METAL HEART']
#Also select the number of relevant items to suggest
no_of_suggestions = 3

all_of_basket = products_prob[basket]
all_of_basket = all_of_basket.sort_values(by = basket, ascending=False)
suggestions_to_customer = list(all_of_basket.index[:no_of_suggestions])

print("===========================================================================")
print('You may also consider buying:', suggestions_to_customer)
