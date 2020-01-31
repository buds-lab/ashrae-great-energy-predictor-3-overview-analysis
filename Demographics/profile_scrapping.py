import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import re
import time
from collections import Counter
import seaborn as sns
import pickle
from scipy import stats
from itertools import groupby

# kaggle_leaderboard.txt is the html code downloaded directly from the web url leaderboard
with open('kaggle_leaderboard.txt', 'r') as file:
    table_string = file.read()

# Create a dictionary with team name : team members
team_locations = [m.start() for m in re.finditer('data-th="Team Name" title=', table_string)]
team_names = [table_string[team+27 : (team+29+table_string[team+29:].find('"'))]
              for team in team_locations]

teams_members_dict = {}
for ind, x in enumerate(team_locations):
    end_of_string = team_locations[ind+1] if ind+1 != len(team_locations) else len(table_string)
    temp_string = table_string[x:end_of_string]

    users_on_team = [m.start() for m in re.finditer('class="avatar" href="', temp_string)]
    team_participants = [temp_string[participant + 21: (participant + 23 + temp_string[participant + 23:].find('"'))]
                         for participant in users_on_team]

    teams_members_dict[team_names[ind]] = team_participants

locations = [m.start() for m in re.finditer('class="avatar" href="', table_string)]
participant_urls = [table_string[participant+21 : (participant+23+table_string[participant+23:].find('"'))]
                    for participant in locations]

# Now get info from each participant page
# def get_participant_data(participant_urls):
# DEFINE VECTORS
country_vec = []
city_vec = []
occupation_vec = []
joindate_vec = []
performancetier_vec = []
performancetiercategory_vec = []
rankpercentage_vec = []
followers_vec = []
totalcompetitions_vec = []
user_vec = []
participant_vec = []

for ind, participant in enumerate(participant_urls[157:]):
    # ACCESS WEBPAGE
    print('https://www.kaggle.com' + participant)
    page = requests.get('https://www.kaggle.com' + participant)
    soup = BeautifulSoup(page.content, 'html.parser')
    scripts = soup.find_all('script')
    variable_string = scripts[16].contents[0]

    # IMPORTANT VARIABLES!
    user_ind = variable_string.find('"userId"')
    user = variable_string[user_ind+9 : user_ind+10+variable_string[user_ind+10:].find('"')]
    user_vec.append(user)

    country_ind = variable_string.find('"country"')
    country = variable_string[country_ind+11 : country_ind+12+variable_string[country_ind+12:].find('"')]
    country_vec.append(country)

    city_ind = variable_string.find('"city"')
    city = variable_string[city_ind+8 : city_ind+9+variable_string[city_ind+9:].find('"')]
    city_vec.append(city)

    occupation_ind = variable_string.find('"occupation"')
    occupation = variable_string[occupation_ind+14 : occupation_ind+15+variable_string[occupation_ind+15:].find('"')]
    occupation_vec.append(occupation)

    joindate_ind = variable_string.find('"userJoinDate"')
    joindate = variable_string[joindate_ind+16 : joindate_ind+17+variable_string[joindate_ind+17:].find('"')]
    joindate_vec.append(joindate)

    performancetier_ind = variable_string.find('"performanceTier"')
    performancetier = variable_string[performancetier_ind+19 : performancetier_ind+20+variable_string[performancetier_ind+20:].find('"')]
    performancetier_vec.append(performancetier)

    performancetiercategory_ind = variable_string.find('"performanceTierCategory"')
    performancetiercategory = variable_string[performancetiercategory_ind+27 : performancetiercategory_ind+28+variable_string[performancetiercategory_ind+28:].find('"')]
    performancetiercategory_vec.append(performancetiercategory)

    rankpercentage_ind = variable_string.find('"rankPercentage"')
    rankpercentage = variable_string[rankpercentage_ind+17 : rankpercentage_ind+17+variable_string[rankpercentage_ind+18:].find('"')]
    rankpercentage_vec.append(rankpercentage)

    followers_ind = variable_string.find('"count"')
    followers = variable_string[followers_ind+8 : followers_ind+8+variable_string[followers_ind+9:].find('"')]
    followers_vec.append(followers)

    totalcompetitions_ind = variable_string.find('"totalResults"')
    totalcompetitions = variable_string[totalcompetitions_ind+15 : totalcompetitions_ind+15+variable_string[totalcompetitions_ind+16:].find('"')]
    totalcompetitions_vec.append(totalcompetitions)
    time.sleep(10)

participant_dict = {'participant_vec': participant_vec,
                    'user_vec': user_vec,
                    'country': country_vec,
                    'city': city_vec,
                    'occupation': occupation_vec,
                    'joindate': joindate_vec,
                    'performancetier': performancetier_vec,
                    'performancetiercategory': performancetiercategory_vec,
                    'rankpercentage': rankpercentage_vec,
                    'followers': followers_vec,
                    'totalcompetitions': totalcompetitions_vec}
pickle.dump(participant_dict, open('/Users/jonathanroth/PycharmProjects/Probabilistic Forecasting/Kaggle/kaggle_profiles2700.obj', 'wb'))

# REMOVE BAD DATA
# country_vec = [x for x in country_vec if "Kaggle" not in x]
# city_vec = [x for x in city_vec if "Kaggle" not in x]
# occupation_vec = [x for x in occupation_vec if "Kaggle" not in x]
# joindate_vec = [x for x in joindate_vec if "Kaggle" not in x]
# performancetier_vec = [x for x in performancetier_vec if "Kaggle" not in x]
# performancetiercategory_vec = [x for x in performancetiercategory_vec if "Kaggle" not in x]
# rankpercentage_vec = [x for x in rankpercentage_vec if "Kaggle" not in x]
# followers_vec = [x for x in followers_vec if "Kaggle" not in x]
# totalcompetitions_vec = [x for x in totalcompetitions_vec if "Kaggle" not in x]
# user_vec = [x for x in user_vec if "Kaggle" not in x]
# participant_vec = [x for x in participant_vec if "Kaggle" not in x]

# COUNTRIES
CUTOFF_NUMBER = 0.01
country_vec_clean = ['Italy' if x=='IT' else x for x in country_vec]
country_vec_clean = ['USA' if x=='United States' else x for x in country_vec_clean]
country_vec_clean = ['China' if x=='中国' else x for x in country_vec_clean]
country_vec_clean = ['Japan' if x=='日本' else x for x in country_vec_clean]
country_vec_clean = ['Spain' if x=='España' else x for x in country_vec_clean]
country_vec_clean = ['Russia' if x=='Россия' else x for x in country_vec_clean]
country_vec_clean = ['Hungary' if x=='Magyarország' else x for x in country_vec_clean]
country_vec_clean = ['Indonesia' if x=='West Java' else x for x in country_vec_clean]
country_vec_clean = ['Vietnam' if x=='Hanoi' else x for x in country_vec_clean]
country_vec_clean = ['UK' if x=='England' else x for x in country_vec_clean]
country_vec_clean = ['UK' if x=='United Kingdom' else x for x in country_vec_clean]
country_vec_clean = ['Latvia' if x=='LV' else x for x in country_vec_clean]
country_vec_clean = ['Czechia' if x=='Česká republika' else x for x in country_vec_clean]
country_vec_clean = ['India' if x=='IN' else x for x in country_vec_clean]
country_vec_clean = ['Russia' if x=='Russian Federation' else x for x in country_vec_clean]
country_vec_clean = ['South Korea' if x=='Republic of Korea' else x for x in country_vec_clean]
country_vec_clean = ['Denmark' if x=='DE' else x for x in country_vec_clean]
country_vec_clean = ['India' if x=='INDIA' else x for x in country_vec_clean]
country_vec_clean = ['Israel' if x=='israel' else x for x in country_vec_clean]
country_vec_clean = ['Poland' if x=='PL' else x for x in country_vec_clean]
country_vec_clean = ['Brazil' if x=='State of São Paulo' else x for x in country_vec_clean]
country_vec_clean = ['Kazakhstan' if x=='Казахстан' else x for x in country_vec_clean]
country_vec_clean = ['Indonesia' if x=='Jakarta' else x for x in country_vec_clean]
country_vec_clean = ['Indonesia' if x=='South Kalimantan' else x for x in country_vec_clean]
country_vec_clean = ['Taiwan' if x=='台灣' else x for x in country_vec_clean]
country_vec_clean = ['Greece' if x=='Ελλάδα' else x for x in country_vec_clean]
country_vec_clean = ['Turkey' if x=='Zonguldak' else x for x in country_vec_clean]
country_vec_clean = ['Brazil' if x=='Recife' else x for x in country_vec_clean]
country_vec_clean = ['Bosnia' if x=='Bosnia and Herzegovina' else x for x in country_vec_clean]
country_vec_clean = ['Tunisia' if x=='Tunisie' else x for x in country_vec_clean]
country_vec_clean = ['USA' if x=='United States of America' else x for x in country_vec_clean]
country_vec_clean = ['Spain' if x=='Espainia' else x for x in country_vec_clean]
country_vec_clean = ['Algeria' if x=='Algérie' else x for x in country_vec_clean]
country_vec_clean = ['Switzerland' if x=='Schweiz' else x for x in country_vec_clean]
country_vec_clean = ['Norway' if x=='NO' else x for x in country_vec_clean]
country_vec_clean = ['Brazil' if x=='Presidente Prudente' else x for x in country_vec_clean]
country_vec_clean = ['Korea' if x=='South Korea' else x for x in country_vec_clean]

country_vec_clean = ['No Data' if x=='ull,' else x for x in country_vec_clean]

country_df = pd.DataFrame({'Country': list(Counter(country_vec_clean).keys()), 'Percentage': list(Counter(country_vec_clean).values())})
country_df['Percentage'] = pd.to_numeric(country_df['Percentage'])/len(country_vec_clean)
country_df['Country'] = country_df['Country'].astype(str)
country_df = country_df.loc[country_df['Country'] != 'No Data', ] # TODO
number_other_countries = country_df.loc[country_df['Percentage'] <= CUTOFF_NUMBER, ].shape[0]/len(country_vec_clean)
country_df = country_df.loc[country_df['Percentage'] > CUTOFF_NUMBER, ]
country_df = country_df.append({'Country': 'Others', 'Percentage': number_other_countries}, ignore_index=True)
country_df = country_df.sort_values(['Percentage'], ascending=False).reset_index(drop=True)

ax = sns.barplot(x="Country", y='Percentage', data=country_df, color=(0.2, 0.4, 0.6, 0.6))
plt.xticks(rotation=35)
plt.savefig('/Users/jonathanroth/PycharmProjects/Probabilistic Forecasting/Kaggle/participant_figures/countries_hist.pdf')
plt.show()

# RANK
rank_df = pd.DataFrame({'Rank': list(Counter(performancetier_vec).keys()), 'Percentage': list(Counter(performancetier_vec).values())})
rank_df['Percentage'] = pd.to_numeric(rank_df['Percentage'])/len(performancetier_vec)
rank_df['Rank'] = rank_df['Rank'].astype(str)
rank_df = rank_df.sort_values(['Percentage'], ascending=False).reset_index(drop=True)
ax = sns.barplot(x="Rank", y='Percentage', data=rank_df, color=(0.2, 0.4, 0.6, 0.6))
plt.xticks(rotation=30)
plt.savefig('/Users/jonathanroth/PycharmProjects/Probabilistic Forecasting/Kaggle/participant_figures/rank_hist.pdf', bbox_inches='tight', pad_inches=2)
plt.show()

# RANK VS. #COMPETITIONS
rank_competitions_df = pd.DataFrame({'Rank': performancetier_vec,
                                     '# Competitions': pd.to_numeric(totalcompetitions_vec)})

ax = sns.boxplot(x="Rank", y="# Competitions", data=rank_competitions_df,
                 order=["novice", "contributor", "expert", "master", "grandmaster"])
plt.savefig('/Users/jonathanroth/PycharmProjects/Probabilistic Forecasting/Kaggle/participant_figures/rankvcompetitions.pdf')
plt.show()

# JOIN-KAGGLE DATE
joindate_array = pd.to_datetime(joindate_vec)
joindate_df = pd.DataFrame({'Year': joindate_array.year})
joindate_df['Month'] = joindate_array.month
joindate_df['Date'] = joindate_array
joindate_df['YearMonth'] = joindate_df['Date'].map(lambda x: 100*x.year + x.month)
sns.distplot(joindate_df['YearMonth'],bins=50)
plt.xlim(right=201912)
plt.show()

joindate_df2 = joindate_df.loc[:,['Date']]
joindate_df2 = joindate_df2.groupby([joindate_df2["Date"].dt.year, joindate_df2["Date"].dt.month]).count()
joindate_df3 = joindate_df2.tail(36)
ax = joindate_df3.plot(kind="bar")
plt.xlabel('Date (Year, Month)')
plt.ylabel('Count')
plt.savefig('/Users/jonathanroth/PycharmProjects/Probabilistic Forecasting/Kaggle/participant_figures/join_date.pdf',bbox_inches='tight', pad_inches=2)
plt.show()

months3 = joindate_df.loc[(joindate_df['Month'] > 9) & (joindate_df['Year'] == 2019), ].shape[0]
months3_6 = joindate_df.loc[(joindate_df['Month'] <= 9) & (joindate_df['Month'] > 6) & (joindate_df['Year'] == 2019), ].shape[0]
months6_9 = joindate_df.loc[(joindate_df['Month'] <= 6) & (joindate_df['Month'] > 3) & (joindate_df['Year'] == 2019), ].shape[0]
months9_12 = joindate_df.loc[(joindate_df['Month'] <= 3) & (joindate_df['Year'] == 2019), ].shape[0]
months12_18 = joindate_df.loc[(joindate_df['Month'] > 6) & (joindate_df['Year'] == 2018), ].shape[0]
months18_24 = joindate_df.loc[(joindate_df['Month'] <= 6) & (joindate_df['Year'] == 2018), ].shape[0]
months24_36 = joindate_df.loc[(joindate_df['Year'] == 2017), ].shape[0]
months36_48 = joindate_df.loc[(joindate_df['Year'] == 2016), ].shape[0]
months48_60 = joindate_df.loc[(joindate_df['Year'] == 2015), ].shape[0]
months60 = joindate_df.loc[(joindate_df['Year'] < 2015), ].shape[0]

joindate_binned = pd.DataFrame({'Count': [months3, months3_6, months6_9, months9_12, months12_18,
                                          months18_24, months24_36, months36_48, months48_60, months60],
                                'Time': ['months3', 'months3_6', 'months6_9', 'months9_12', 'months12_18',
                                          'months18_24', 'months24_36', 'months36_48', 'months48_60', 'months60']})

ax = sns.barplot(x="Time", y='Count', data=joindate_binned)
plt.xticks(rotation=45)
plt.show()

# STATS!
percentage_ranked = len(list(filter(lambda a: a != '1.0060828', rankpercentage_vec)))/len(user_vec)
number_countries = len(list(Counter(country_vec_clean).keys())) - 1
joined_for_competition = months3


# print(soup.prettify())

