import torch
from tqdm import tqdm
from transformers import pipeline
import os
import pandas as pd
import json

user=pd.read_json(('Dataset/user.json'))

user_text=list(user['description'])

# Prepare the user index
user_idx=user['id']
uid_index={uid:index for index,uid in enumerate(user_idx.values)}

# Prepare the user index to user id
print("extracting each_user's tweets")
id_tweet={i:[] for i in range(len(user_idx))}
for i in range(9):
    name='tweet_'+str(i)+'.json'
    user_tweets=json.load(open("Dataset/"+name,'r'))
    for each in user_tweets:
        uid='u'+str(each['author_id'])
        text=each['text']
        try:
            index=uid_index[uid]
            id_tweet[index].append(text)
        except KeyError:
            continue
json.dump(id_tweet,open('./processed_data/id_tweet.json','w'))

folder_path = 'processed_data'
processed_data_path=os.path.join(folder_path)

each_user_tweets=json.load(open(processed_data_path+"id_tweet.json",'r'))

# Load the model RoBERTa
feature_extract=pipeline('feature-extraction',model='roberta-base',tokenizer='roberta-base',device=3,padding=True, truncation=True,max_length=50, add_special_tokens = True)

def Des_embbeding():
    print('Running description embedding')
    path=processed_data_path+"des_tensor.pt"
    des_vec=[]
    for k,each in enumerate(tqdm(user_text)):
        if each is None:
            des_vec.append(torch.zeros(768))
        else:
            feature=torch.Tensor(feature_extract(each))
            for (i,tensor) in enumerate(feature[0]):
                if i==0:
                    feature_tensor=tensor
                else:
                    feature_tensor+=tensor
            feature_tensor/=feature.shape[1]
            des_vec.append(feature_tensor)
            
    des_tensor=torch.stack(des_vec,0)
    torch.save(des_tensor,path)
    print('Finished')
    return des_tensor

def tweets_embedding():
    print('Running tweets embedding')
    path=processed_data_path+"tweets_tensor.pt"
    tweets_list=[]
    for i in tqdm(range(len(each_user_tweets))):
        if len(each_user_tweets[str(i)])==0:
            total_each_person_tweets=torch.zeros(768)
        else:
            for j in range(len(each_user_tweets[str(i)])):
                each_tweet=each_user_tweets[str(i)][j]
                if each_tweet is None:
                    total_word_tensor=torch.zeros(768)
                else:
                    each_tweet_tensor=torch.tensor(feature_extract(each_tweet))
                    for k,each_word_tensor in enumerate(each_tweet_tensor[0]):
                        if k==0:
                            total_word_tensor=each_word_tensor
                        else:
                            total_word_tensor+=each_word_tensor
                    total_word_tensor/=each_tweet_tensor.shape[1]
                if j==0:
                    total_each_person_tweets=total_word_tensor
                elif j==20:
                    break
                else:
                    total_each_person_tweets+=total_word_tensor
            if (j==20):
                total_each_person_tweets/=20
            else:
                total_each_person_tweets/=len(each_user_tweets[str(i)])
                
        tweets_list.append(total_each_person_tweets)
            
    tweet_tensor=torch.stack(tweets_list)
    torch.save(tweet_tensor,path)

    print('Finished')

Des_embbeding()
tweets_embedding()
