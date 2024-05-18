import pandas as pd
import numpy as np
from scipy.io import savemat
from tqdm import tqdm
from datetime import datetime as dt
from math import log
import json
import re
import torch
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the user and label data
print('loading data')
user = pd.read_json("Dataset/user.json")
label = pd.read_csv("Dataset/label.csv")

# Prepare the user index
user_idx=user['id']
uid_index={uid:index for index,uid in enumerate(user_idx.values)}
user_index_to_uid = list(user.id)
uid_to_user_index = {x : i for i, x in enumerate(user_index_to_uid)}

# Create the folder processed_data
folder_path = 'processed_data'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Save the user index to a JSON file
file_path = os.path.join(folder_path, 'uid_index.json')

with open(file_path, 'w') as file:
    json.dump(uid_index, file)

# Prepare the labels
print('extracting labels')
uid_label={uid:label for uid, label in zip(label['id'].values,label['label'].values)}
label_new=[]

for i,uid in enumerate(tqdm(user_idx.values)):
    single_label=uid_label[uid]
    if single_label =='human':
        label_new.append(0)
    else:
        label_new.append(1)


# Prepare the features
print('extracting num_properties')
following_count_list=[] ##0
for i,each in enumerate(user['public_metrics']):
    if i==len(user):
        break
    if each is not None and isinstance(each,dict):
        if each['following_count'] is not None:
            following_count_list.append(each['following_count'])
        else:
            following_count_list.append(0)
    else:
        following_count_list.append(0)

following_count=pd.DataFrame(following_count_list)
following_count=(following_count-following_count.mean())/following_count.std()
    

followers_count_list=[] ##1
for each in user['public_metrics']:
    if each is not None and each['followers_count'] is not None:
        followers_count_list.append(int(each['followers_count']))
    else:
        followers_count_list.append(0)
    
followers_count=pd.DataFrame(followers_count_list)
followers_count=(followers_count-followers_count.mean())/followers_count.std()

tweet_count=[] ##2
for i,each in enumerate(user['public_metrics']):
    if i==len(user):
        break
    if each is not None and isinstance(each,dict):
        if each['tweet_count'] is not None:
            tweet_count.append(each['tweet_count'])
        else:
            tweet_count.append(0)
    else:
        tweet_count.append(0)

tweet_count=pd.DataFrame(tweet_count)
tweet_count=(tweet_count-tweet_count.mean())/tweet_count.std()
        
num_username=[] ##3
has_bot_word_in_username=[] ##15
number_count_in_username=[] ##28
rel_upper_lower_username=[] ##30 
digits_username=[] ##32
for each in user['username']:
    if each is not None:
        num_username.append(len(each))
        hashtags = re.findall(r'#\w', each)
        matchObj = re.search('bot', each, flags=re.IGNORECASE)
        if matchObj:
            has_bot_word_in_username.append(1)
        else:
            has_bot_word_in_username.append(0)
        numbers = re.findall(r'\d+', each)
        number_count_in_username.append(len(numbers))
        big_num = 0  
        small_num = 0 
        cnt = 0
        for c in each:
            if c.isupper():
                big_num += 1
            elif c.islower():
                small_num += 1
            if '0'<=c and c<='9':
                cnt += 1
        digits_username.append(cnt)
        if small_num == 0:
            rel_upper_lower_username.append(0)
        else:
            rel_upper_lower_username.append(big_num/small_num)
        
    else:
        num_username.append(int(0))
        has_bot_word_in_username.append(0)
        number_count_in_username.append(0)
        rel_upper_lower_username.append(0)
        digits_username.append(0)
        
num_username=pd.DataFrame(num_username)
num_username=(num_username-num_username.mean())/num_username.std()

has_bot_word_in_username = pd.DataFrame(has_bot_word_in_username)

number_count_in_username=pd.DataFrame(number_count_in_username)
number_count_in_username=(number_count_in_username-number_count_in_username.mean())/number_count_in_username.std()

rel_upper_lower_username=pd.DataFrame(rel_upper_lower_username)
rel_upper_lower_username=(rel_upper_lower_username-rel_upper_lower_username.mean())/rel_upper_lower_username.std()

digits_username=pd.DataFrame(digits_username)
digits_username=(digits_username-digits_username.mean())/digits_username.std()

num_name=[] ##4
has_bot_word_in_name=[] ##16
number_count_in_name=[] ##29
rel_upper_lower_name=[] #31
digits_name=[] ##33
for each in user['name']:
    if each is not None:
        num_name.append(len(each))
        matchObj = re.search('bot', each, flags=re.IGNORECASE)
        if matchObj:
            has_bot_word_in_name.append(1)
        else:
            has_bot_word_in_name.append(0)
        numbers = re.findall(r'\d+', each)
        number_count_in_name.append(len(numbers))
        big_num = 0  
        small_num = 0 
        cnt = 0
        for c in each:
            if c.isupper():
                big_num += 1
            elif c.islower():
                small_num += 1
            if '0'<=c and c<='9':
                cnt += 1
        digits_name.append(cnt)
        if small_num == 0:
            rel_upper_lower_name.append(0)
        else:
            rel_upper_lower_name.append(big_num/small_num)
    else:
        num_name.append(int(0))
        has_bot_word_in_name.append(0)
        number_count_in_name.append(0)
        rel_upper_lower_name.append(0)
        digits_name.append(0)

num_name=pd.DataFrame(num_name)
num_name=(num_name-num_name.mean())/num_name.std()

has_bot_word_in_name = pd.DataFrame(has_bot_word_in_name)

number_count_in_name=pd.DataFrame(number_count_in_name)
number_count_in_name=(number_count_in_name-number_count_in_name.mean())/number_count_in_name.std()

rel_upper_lower_name=pd.DataFrame(rel_upper_lower_name)
rel_upper_lower_name=(rel_upper_lower_name-rel_upper_lower_name.mean())/rel_upper_lower_name.std()

digits_name=pd.DataFrame(digits_name)
digits_name=(digits_name-digits_name.mean())/digits_name.std()

active_days_list=[] ##5
created_at=user['created_at']
created_at=pd.to_datetime(created_at,unit='s')
date0=dt.strptime('Wed Feb 14 00:00:00 +0000 2024 ','%a %b %d %X %z %Y ')

for each in created_at:
    active_days_list.append((date0-each).days)

    
active_days=pd.DataFrame(active_days_list)
active_days=active_days.fillna(int(1)).astype(np.float32)

active_days=pd.DataFrame(active_days)
active_days.fillna(int(0))
active_days=active_days.fillna(int(0)).astype(np.float32)
active_days=(active_days-active_days.mean())/active_days.std()



listed_count=[] ##9
for i,each in enumerate(user['public_metrics']):
    if i==len(user):
        break
    if each is not None and isinstance(each,dict):
        if each['listed_count'] is not None:
            listed_count.append(each['listed_count'])
        else:
            listed_count.append(0)
    else:
        listed_count.append(0)

listed_count=pd.DataFrame(listed_count)
listed_count=(listed_count-listed_count.mean())/listed_count.std()


num_of_hashtags=[] ##12
for each in user['entities']:
    try:
        num_of_hashtags.append(len(each['description']['hashtags']))
    except:
        num_of_hashtags.append(0)
num_of_hashtags=pd.DataFrame(num_of_hashtags)
num_of_hashtags=(num_of_hashtags-num_of_hashtags.mean())/num_of_hashtags.std()

has_description_list=[] ##10
has_bot_word_in_description=[] ##14
num_description=[] ##17
hashtags_count_in_description=[] ##19
urls_count_in_description=[] ##20
for each in user['description']:
    if each == "":
        has_description_list.append(0)
        has_bot_word_in_description.append(0)
        num_description.append(0)
        hashtags_count_in_description.append(0)
        urls_count_in_description.append(0)
    else:
        has_description_list.append(1)
        matchObj = re.search('bot', each, flags=re.IGNORECASE)
        if matchObj:
            has_bot_word_in_description.append(1)
        else:
            has_bot_word_in_description.append(0)
        num_description.append(len(each))
        hashtags = re.findall(r'#\w', each)
        hashtags_count_in_description.append(len(hashtags))
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', each)
        urls_count_in_description.append(len(urls))
        

has_description = pd.DataFrame(has_description_list)

has_bot_word_in_description = pd.DataFrame(has_bot_word_in_description)

num_description=pd.DataFrame(num_description)
num_description=(num_description-num_description.mean())/num_description.std()

hashtags_count_in_description=pd.DataFrame(hashtags_count_in_description)
hashtags_count_in_description=(hashtags_count_in_description-hashtags_count_in_description.mean())/hashtags_count_in_description.std()

urls_count_in_description=pd.DataFrame(urls_count_in_description)
urls_count_in_description=(urls_count_in_description-urls_count_in_description.mean())/urls_count_in_description.std()



get_followers_followees=[] ##18
get_following_followers=[] ##34
following_followers_square=[] ##35
following_total=[] ##36
followers2_following=[] ##37
followers2_greater100=[] ##38
for each in user['public_metrics']:
    follower=each['followers_count']
    following=each['following_count']

    if follower !=0 and following !=0:
        fracc=follower/following
        fracc3=following/((follower)**2)
        fracc4=following/(following+follower)
        fracc5=2*follower-following
        fracc6=2*follower
    else:
        fracc=0
        fracc3=0
        fracc4=0
        fracc5=0
        fracc6=0
    get_followers_followees.append(fracc)
    following_followers_square.append(fracc3)
    following_total.append(fracc4)
    followers2_following.append(fracc5)


    if fracc6>=100:
        followers2_greater100.append(1)
    else:
        followers2_greater100.append(0)
        
    
get_followers_followees=pd.DataFrame(get_followers_followees)
get_followers_followees=(get_followers_followees-get_followers_followees.mean())/get_followers_followees.std()

get_following_followers=pd.DataFrame(get_following_followers)
get_following_followers=(get_following_followers-get_following_followers.mean())/get_following_followers.std()


following_followers_square=pd.DataFrame(following_followers_square)
following_followers_square=(following_followers_square-following_followers_square.mean())/following_followers_square.std()

following_total=pd.DataFrame(following_total)
following_total=(following_total-following_total.mean())/following_total.std()

followers2_following=pd.DataFrame(followers2_following)
followers2_following=(followers2_following-followers2_following.mean())/followers2_following.std()

followers2_greater100=pd.DataFrame(followers2_greater100)

tweet_freq=[] ##22
for i,each in enumerate(user['public_metrics']):  
    try:   
        freq = float(each['tweet_count'])/active_days_list[i]
        tweet_freq.append(freq)   
    except:
        tweet_freq.append(0)
tweet_freq=pd.DataFrame(tweet_freq)
tweet_freq=(tweet_freq-tweet_freq.mean())/tweet_freq.std()


followers_growth_rate=[] ##23
for i,each in enumerate(user['public_metrics']):  
    try:   
        rate = float(each['followers_count'])/active_days_list[i]
        followers_growth_rate.append(rate)   
    except:
        followers_growth_rate.append(0)
followers_growth_rate=pd.DataFrame(followers_growth_rate)
followers_growth_rate=(followers_growth_rate-followers_growth_rate.mean())/followers_growth_rate.std()


friends_growth_rate=[] ##24
for i,each in enumerate(user['public_metrics']):  
    try:   
        rate = float(each['following_count'])/active_days_list[i]
        friends_growth_rate.append(rate)   
    except:
        friends_growth_rate.append(0)
friends_growth_rate=pd.DataFrame(friends_growth_rate)
friends_growth_rate=(friends_growth_rate-friends_growth_rate.mean())/friends_growth_rate.std()

listed_growth_rate=[] ##39
for i,each in enumerate(user['public_metrics']):  
    try:   
        rate = float(each['listed_count'])/active_days_list[i]
        listed_growth_rate.append(rate)   
    except:
        listed_growth_rate.append(0)
listed_growth_rate=pd.DataFrame(listed_growth_rate)
listed_growth_rate=(listed_growth_rate-listed_growth_rate.mean())/listed_growth_rate.std()


def Lev_distance(A,B):
    dp = np.array(np.arange(len(B)+1))
    for i in range(1, len(A)+1):
        temp1 = dp[0]
        dp[0] += 1
        for j in range(1, len(B)+1):
            temp2 = dp[j]
            if A[i-1] == B[j-1]:
                dp[j] = temp1
            else:
                dp[j] = min(temp1, min(dp[j-1], dp[j]))+1
            temp1 = temp2

    return dp[len(B)]


lev_distance_username_name=[] ##25
relation_length_username_name=[] ##26
for index, row in user.iterrows():
    name = row['name']
    username = row['username']
    distance = Lev_distance(name, username)
    lev_distance_username_name.append(distance)

    if len(name)==0:
        relation_length_username_name.append(0)
    else:
        fracc=len(username)/len(name)
        relation_length_username_name.append(fracc)
    

lev_distance_username_name=pd.DataFrame(lev_distance_username_name)
lev_distance_username_name=(lev_distance_username_name-lev_distance_username_name.mean())/lev_distance_username_name.std()

relation_length_username_name=pd.DataFrame(relation_length_username_name)
relation_length_username_name=(relation_length_username_name-relation_length_username_name.mean())/relation_length_username_name.std()



listed_followers=[] ##40
tweets_followers=[] ##41
listed_tweets=[] ##42
for each in user['public_metrics']:
    if each['followers_count'] !=0:
        fracc=(each['listed_count'])/(each['followers_count'])
        fracc2=(each['tweet_count'])/(each['followers_count'])
    else:
        fracc=0
        fracc2=0
    listed_followers.append(fracc)
    tweets_followers.append(fracc2)
    if each['tweet_count'] !=0:
        fracc3=(each['listed_count'])/(each['tweet_count'])
    else:
        fracc3=0
    listed_tweets.append(fracc3)
        
listed_followers=pd.DataFrame(listed_followers)
listed_followers=(listed_followers-listed_followers.mean())/listed_followers.std()

tweets_followers=pd.DataFrame(tweets_followers)
tweets_followers=(tweets_followers-tweets_followers.mean())/tweets_followers.std()

listed_tweets=pd.DataFrame(listed_tweets)
listed_tweets=(listed_tweets-listed_tweets.mean())/listed_tweets.std()


des_sentiment_score=[] ##43
for each in user['description']:
    score = SentimentIntensityAnalyzer().polarity_scores(each)['compound']
    des_sentiment_score.append(score)

des_sentiment_score=pd.DataFrame(des_sentiment_score)
des_sentiment_score=(des_sentiment_score-des_sentiment_score.mean())/des_sentiment_score.std()

entropy_name=[] ##44
for each in user['name']:
    # Inicialización del diccionario para contar la frecuencia de cada carácter
    word = {}
    for c in each:
        if c not in word:
            word[c] = 0
        word[c] += 1
        
    # Cálculo de la entropía de Shannon para el nombre completo
    ShannonEnt = 0.0
    for char, freq in word.items():
        prob = float(freq) / len(each)
        ShannonEnt -= prob * log(prob, 2)

    entropy_name.append(ShannonEnt)

entropy_name=pd.DataFrame(entropy_name)
entropy_name=(entropy_name-entropy_name.mean())/entropy_name.std()


entropy_username=[] ##45
for each in user['username']:
    # Inicialización del diccionario para contar la frecuencia de cada carácter
    word = {}
    for c in each:
        if c not in word:
            word[c] = 0
        word[c] += 1
        
    # Cálculo de la entropía de Shannon para el nombre completo
    ShannonEnt = 0.0
    for char, freq in word.items():
        prob = float(freq) / len(each)
        ShannonEnt -= prob * log(prob, 2)

    entropy_username.append(ShannonEnt)

entropy_username=pd.DataFrame(entropy_username)
entropy_username=(entropy_username-entropy_username.mean())/entropy_username.std()



################################# CAT PROPERTIES #################################
print('extracting cat_properties')

protected_list=[] ##6
protected=user['protected']
for each in protected:
    if each == True:
        protected_list.append(1)
    else:
        protected_list.append(0)

protected = pd.DataFrame(protected_list)


verified_list=[] ##7
verified=user['verified']
for each in verified:
    if each == True:
        verified_list.append(1)
    else:
        verified_list.append(0)
        
verified = pd.DataFrame(verified_list)


default_profile_image=[] ##8
for each in user['profile_image_url']:
    if each is not None:
        if each=='https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png':
            default_profile_image.append(int(1))
        elif each=='':
            default_profile_image.append(int(1))
        else:
            default_profile_image.append(int(0))
    else:
        default_profile_image.append(int(1))

default_profile_image = pd.DataFrame(default_profile_image)


has_location_list=[] ##11
for each in user['location']:
    if each is None:
        has_location_list.append(0)
    else:
        has_location_list.append(1)
has_location = pd.DataFrame(has_location_list)

has_url=[] ##13
for each in user['url']:
    if each == "":
        has_url.append(0)
    else:
        has_url.append(1)
has_url = pd.DataFrame(has_url)



def_profile=[] ##21
for index, row in user.iterrows():
    des = row['description']
    loca = row['location']
    url = row['url']
        
    if des=="" and loca is None and url=="":
        def_profile.append(1)
    else:
        def_profile.append(0)
def_profile = pd.DataFrame(def_profile)

has_pinned_tweet=[] ##27
for each in user['pinned_tweet_id']:
    if np.isnan(each):
        has_pinned_tweet.append(0)
    else:
        has_pinned_tweet.append(1)
has_pinned_tweet = pd.DataFrame(has_pinned_tweet)


# Concatenate all the features
X_user = pd.concat([following_count, tweet_count, followers_count, num_username, num_name, active_days, protected, verified, default_profile_image, listed_count, has_description, has_location, num_of_hashtags, has_url, has_bot_word_in_description, has_bot_word_in_username, has_bot_word_in_name, num_description, get_followers_followees, hashtags_count_in_description, urls_count_in_description, def_profile, tweet_freq, followers_growth_rate, friends_growth_rate, lev_distance_username_name, relation_length_username_name, has_pinned_tweet, number_count_in_username, number_count_in_name, rel_upper_lower_username, rel_upper_lower_name, digits_username, digits_name, get_following_followers, following_followers_square, following_total, followers2_following, followers2_greater100, listed_growth_rate, listed_followers, tweets_followers, listed_tweets, des_sentiment_score, entropy_name, entropy_username], axis=1)
        
# Prepare the labels
label_new = np.array(label_new)  # Asumiendo que label_new ya contiene los labels 0 (human) o 1 (bot)
y = pd.DataFrame(label_new)

# Convert the DataFrames to numpy arrays
X_user = X_user.to_numpy()
y = y.to_numpy()

# Save the data to a .mat file
savemat("user_data.mat", {"X": X_user, "y": y})

# Convert the data to PyTorch tensors
features_tensor = torch.tensor(X_user.values, dtype=torch.float32)
labels_tensor = torch.tensor(label_new, dtype=torch.int64)

# Save the tensors
torch.save(features_tensor, os.path.join(folder_path, 'features.pt'))
torch.save(labels_tensor, os.path.join(folder_path, 'labels.pt'))