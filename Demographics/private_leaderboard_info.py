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
import squarify
from bokeh.palettes import viridis, inferno, plasma
from scipy import stats
from itertools import groupby

def create_unique_countries(country_vec):
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
    country_vec_clean = ['Brazil' if x=='Brasil' else x for x in country_vec_clean]
    country_vec_clean = ['Bosnia' if x=='Bosnia and Herzegovina' else x for x in country_vec_clean]
    country_vec_clean = ['Tunisia' if x=='Tunisie' else x for x in country_vec_clean]
    country_vec_clean = ['USA' if x=='United States of America' else x for x in country_vec_clean]
    country_vec_clean = ['Spain' if x=='Espainia' else x for x in country_vec_clean]
    country_vec_clean = ['Algeria' if x=='Algérie' else x for x in country_vec_clean]
    country_vec_clean = ['Switzerland' if x=='Schweiz' else x for x in country_vec_clean]
    country_vec_clean = ['Norway' if x=='NO' else x for x in country_vec_clean]
    country_vec_clean = ['Brazil' if x=='Presidente Prudente' else x for x in country_vec_clean]
    country_vec_clean = ['Korea' if x=='South Korea' else x for x in country_vec_clean]
    country_vec_clean = ['No Data' if x=='=window.Kaggle||{};Kaggle.State=Kaggle.State||[];Kaggle.State.push({' else x for x in country_vec_clean]
    country_vec_clean = ['Colombia' if x=='Antioquia' else x for x in country_vec_clean]
    country_vec_clean = ['Germany' if x=='Deutschland' else x for x in country_vec_clean]
    country_vec_clean = ['France' if x=='FR' else x for x in country_vec_clean]
    country_vec_clean = ['UK' if x=='GB' else x for x in country_vec_clean]
    country_vec_clean = ['Greece' if x=='GR' else x for x in country_vec_clean]
    country_vec_clean = ['Taiwan' if x=='Hsinchu City' else x for x in country_vec_clean]
    country_vec_clean = ['India' if x=='Maharashtra' else x for x in country_vec_clean]
    country_vec_clean = ['Brazil' if x=='Maringá' else x for x in country_vec_clean]
    country_vec_clean = ['Brazil' if x=='Mogi das Cruzes' else x for x in country_vec_clean]
    country_vec_clean = ['Netherlands' if x=='Nederland' else x for x in country_vec_clean]
    country_vec_clean = ['Poland' if x=='Polska' else x for x in country_vec_clean]
    country_vec_clean = ['Indonesia' if x=='Special Region of Yogyakarta' else x for x in country_vec_clean]
    country_vec_clean = ['Sweden' if x=='Sverige' else x for x in country_vec_clean]
    country_vec_clean = ['Netherlands' if x=='The Netherlands' else x for x in country_vec_clean]
    country_vec_clean = ['Turkey' if x=='Türkiye' else x for x in country_vec_clean]
    country_vec_clean = ['USA' if x=='US' else x for x in country_vec_clean]
    country_vec_clean = ['Russia' if x=='RU' else x for x in country_vec_clean]
    country_vec_clean = ['Ukraine' if x=='Украина' else x for x in country_vec_clean]
    country_vec_clean = ['No Data' if x=='ull,' else x for x in country_vec_clean]
    return country_vec_clean


def get_participant_info(participant_urls):
    # Now get info from each participant page
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

    for ind, participant in enumerate(participant_urls[4250:]):
        # ACCESS WEBPAGE
        print('https://www.kaggle.com' + participant)
        page = requests.get('https://www.kaggle.com' + participant)
        soup = BeautifulSoup(page.content, 'html.parser')
        scripts = soup.find_all('script')
        variable_string = scripts[16].contents[0]

        # IMPORTANT VARIABLES!
        user_ind = variable_string.find('"userId"')
        user = variable_string[user_ind + 9: user_ind + 10 + variable_string[user_ind + 10:].find('"')]
        user_vec.append(user)

        country_ind = variable_string.find('"country"')
        country = variable_string[country_ind + 11: country_ind + 12 + variable_string[country_ind + 12:].find('"')]
        country_vec.append(country)

        city_ind = variable_string.find('"city"')
        city = variable_string[city_ind + 8: city_ind + 9 + variable_string[city_ind + 9:].find('"')]
        city_vec.append(city)

        occupation_ind = variable_string.find('"occupation"')
        occupation = variable_string[occupation_ind + 14: occupation_ind + 15 + variable_string[occupation_ind + 15:].find('"')]
        occupation_vec.append(occupation)

        joindate_ind = variable_string.find('"userJoinDate"')
        joindate = variable_string[joindate_ind + 16: joindate_ind + 17 + variable_string[joindate_ind + 17:].find('"')]
        joindate_vec.append(joindate)

        performancetier_ind = variable_string.find('"performanceTier"')
        performancetier = variable_string[performancetier_ind + 19: performancetier_ind + 20 + variable_string[performancetier_ind + 20:].find('"')]
        performancetier_vec.append(performancetier)

        performancetiercategory_ind = variable_string.find('"performanceTierCategory"')
        performancetiercategory = variable_string[performancetiercategory_ind + 27: performancetiercategory_ind + 28 + variable_string[performancetiercategory_ind + 28:].find('"')]
        performancetiercategory_vec.append(performancetiercategory)

        rankpercentage_ind = variable_string.find('"rankPercentage"')
        rankpercentage = variable_string[rankpercentage_ind + 17: rankpercentage_ind + 17 + variable_string[rankpercentage_ind + 18:].find('"')]
        rankpercentage_vec.append(rankpercentage)

        followers_ind = variable_string.find('"count"')
        followers = variable_string[followers_ind + 8: followers_ind + 8 + variable_string[followers_ind + 9:].find('"')]
        followers_vec.append(followers)

        totalcompetitions_ind = variable_string.find('"totalResults"')
        totalcompetitions = variable_string[totalcompetitions_ind + 15: totalcompetitions_ind + 15 + variable_string[totalcompetitions_ind + 16:].find('"')]
        totalcompetitions_vec.append(totalcompetitions)
        time.sleep(2)

    participant_dict = {'participant_vec': participant_urls,
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
    pickle.dump(participant_dict,open('/Users/jonathanroth/PycharmProjects/Kaggle_Demographics/kaggle_profiles_all.obj', 'wb'))

    # REMOVE DUPLICATE/BAD DATA
    participant_dict2 = participant_dict
    participant_dict2['participant_vec'] = participant_dict['user_vec']  # TEMPORARY SO THE NEXT LINE WORKS
    participant_dict2 = {k: np.delete(np.array(v), 720) for k, v in participant_dict.items()}
    participant_dict2['participant_vec'] = np.array(participant_urls)
    unique_users = np.zeros_like(participant_dict2['user_vec'], dtype=bool)
    unique_users[np.unique(participant_dict2['user_vec'], return_index=True)[1]] = True
    participant_dict2 = {k: np.array(v)[unique_users] for k, v in participant_dict2.items()}
    [len(v) for k, v in participant_dict2.items()]
    pickle.dump(participant_dict2,open('/Users/jonathanroth/PycharmProjects/Kaggle_Demographics/kaggle_profiles_all2.obj', 'wb'))
    return participant_dict2
    # LOAD DATA


def create_country_plots(participant_dict2, CUTOFF_NUMBER=1):
    country_vec_clean = create_unique_countries(participant_dict2['country'])

    country_df = pd.DataFrame(
        {'Country': list(Counter(country_vec_clean).keys()), 'Count': list(Counter(country_vec_clean).values())})
    country_df['Percentage'] = pd.to_numeric(country_df['Count']) / len(country_vec_clean) * 100
    country_df['Country'] = country_df['Country'].astype(str)
    country_df = country_df.loc[country_df['Country'] != 'No Data',]  # TODO
    number_other_countries = country_df.loc[country_df['Percentage'] <= CUTOFF_NUMBER,]
    country_df = country_df.loc[country_df['Percentage'] > CUTOFF_NUMBER,]
    country_df = country_df.append({'Country': 'Other Countries',
                                    'Count': number_other_countries['Count'].sum(),
                                    'Percentage': number_other_countries['Percentage'].sum()}, ignore_index=True)
    country_df = country_df.sort_values(['Percentage'], ascending=False).reset_index(drop=True)

    # TREEMAP PLOT
    squarify.plot(sizes=country_df['Count'],
                  label=country_df['Country'] + '\nCount=' + country_df['Count'].astype(str),
                  alpha=.7,
                  color=inferno(14))
    plt.axis('off')
    plt.title('Participants Countries of Origin')
    plt.savefig('/Users/jonathanroth/PycharmProjects/Kaggle_Demographics/participant_figures/countries_treemap.png')
    plt.show()

    # BARPLOT
    ax = sns.barplot(x="Country", y='Percentage', data=country_df, color=(0.2, 0.4, 0.6, 0.6))
    plt.xticks(rotation=35)
    plt.savefig(
        '/Users/jonathanroth/PycharmProjects/Probabilistic Forecasting/Kaggle/participant_figures/countries_hist.pdf')
    plt.show()
    return country_df


def ranking_plots(participant_dict2):
    performancetier_vec = participant_dict2['performancetier']
    performancetier_vec = np.delete(performancetier_vec, 147)

    rank_df = pd.DataFrame(
        {'Rank': list(Counter(performancetier_vec).keys()), 'Count': list(Counter(performancetier_vec).values())})
    rank_df['Percentage'] = pd.to_numeric(rank_df['Count']) / len(performancetier_vec)
    rank_df['Rank'] = rank_df['Rank'].astype(str)
    rank_df = rank_df.sort_values(['Percentage'], ascending=False).reset_index(drop=True)
    rank_df['circle_size'] = rank_df['Count'] ** 0.5 / np.pi
    ax = sns.barplot(x="Rank", y='Count', data=rank_df, color=(0.2, 0.4, 0.6, 0.6))
    plt.xticks(rotation=30)
    plt.savefig(os.getcwd() + '/participant_figures/rank_hist.pdf', bbox_inches='tight', pad_inches=2)
    plt.show()

    plt.rcParams['axes.xmargin'] = 0.1
    ax = sns.scatterplot(x="Rank", y=np.repeat(1, 5), s=rank_df['Count'] * 2, data=rank_df,
                         color=(0.2, 0.4, 0.6, 0.6))
    plt.savefig(os.getcwd() + '/participant_figures/ranks_numbers.png', bbox_inches='tight')
    plt.show()
    return rank_df


def rankVcompetitions(participant_dict2):
    performancetier_vec = participant_dict2['performancetier']
    performancetier_vec = np.delete(performancetier_vec, 147)

    plt.rcParams['axes.ymargin'] = 0.05
    plt.rcParams['axes.xmargin'] = 0.9
    totalcompetitions_vec = participant_dict2['totalcompetitions']
    totalcompetitions_vec = np.delete(totalcompetitions_vec, 147)
    rank_competitions_df = pd.DataFrame({'Rank': performancetier_vec,
                                         '# Competitions': pd.to_numeric(totalcompetitions_vec)})

    ax = sns.boxplot(x="Rank", y="# Competitions", data=rank_competitions_df,
                     order=["novice", "contributor", "expert", "master", "grandmaster"])
    plt.savefig(os.getcwd() + '/participant_figures/rankvcompetitions.png')
    plt.show()
    return rank_competitions_df


def join_dates(participant_dict2):
    joindate_vec = participant_dict2['joindate']
    joindate_vec = np.delete(joindate_vec, 147)
    joindate_array = pd.to_datetime(joindate_vec)
    joindate_df = pd.DataFrame({'Year': joindate_array.year})
    joindate_df['Month'] = joindate_array.month
    joindate_df['Date'] = joindate_array
    joindate_df['YearMonth'] = joindate_df['Date'].map(lambda x: 100 * x.year + x.month)

    joindate_df2 = joindate_df.loc[:, ['Date']]
    joindate_df2 = joindate_df2.groupby([joindate_df2["Date"].dt.year, joindate_df2["Date"].dt.month]).count()
    joindate_df3 = joindate_df2.tail(36)
    ax = joindate_df3.plot(kind="bar")
    plt.xlabel('Date (Year, Month)')
    plt.ylabel('Count')
    plt.savefig(os.getcwd() + '/participant_figures/join_date.png', bbox_inches='tight', pad_inches=2)
    plt.show()
    # months3 = joindate_df.loc[(joindate_df['Month'] > 9) & (joindate_df['Year'] == 2019), ].shape[0]
    # months3_6 = joindate_df.loc[(joindate_df['Month'] <= 9) & (joindate_df['Month'] > 6) & (joindate_df['Year'] == 2019), ].shape[0]
    # months6_9 = joindate_df.loc[(joindate_df['Month'] <= 6) & (joindate_df['Month'] > 3) & (joindate_df['Year'] == 2019), ].shape[0]
    # months9_12 = joindate_df.loc[(joindate_df['Month'] <= 3) & (joindate_df['Year'] == 2019), ].shape[0]
    # months12_18 = joindate_df.loc[(joindate_df['Month'] > 6) & (joindate_df['Year'] == 2018), ].shape[0]
    # months18_24 = joindate_df.loc[(joindate_df['Month'] <= 6) & (joindate_df['Year'] == 2018), ].shape[0]
    # months24_36 = joindate_df.loc[(joindate_df['Year'] == 2017), ].shape[0]
    # months36_48 = joindate_df.loc[(joindate_df['Year'] == 2016), ].shape[0]
    # months48_60 = joindate_df.loc[(joindate_df['Year'] == 2015), ].shape[0]
    # months60 = joindate_df.loc[(joindate_df['Year'] < 2015), ].shape[0]
    # joindate_binned = pd.DataFrame({'Count': [months3, months3_6, months6_9, months9_12, months12_18,
    #                                           months18_24, months24_36, months36_48, months48_60, months60],
    #                                 'Time': ['months3', 'months3_6', 'months6_9', 'months9_12', 'months12_18',
    #                                           'months18_24', 'months24_36', 'months36_48', 'months48_60', 'months60']})
    # ax = sns.barplot(x="Time", y='Count', data=joindate_binned)
    # plt.xticks(rotation=45)
    # plt.show()
    return joindate_df3


def scoresVentries(teams_scores_df):
    # PLOTS -- SCORES AND ENTRIES
    plt.rcParams['axes.ymargin'] = 0
    plt.rcParams['axes.xmargin'] = 0
    plt.scatter(teams_scores_df['Entries'], teams_scores_df['Score'])
    plt.ylim([1.2, 2])
    plt.show()


def score_boxplot(final_df):
    ax = sns.boxplot(x="performancetier", y="Score", data=final_df[final_df['Score'] <=2],
                     order=["novice", "contributor", "expert", "master", "grandmaster"])
    # plt.ylim([1.2, 2])
    plt.savefig(os.getcwd() + '/participant_figures/rankvscores.png')
    plt.show()


def medals_plots(medal_groups):
    plt.rcParams['axes.xmargin'] = 0.1
    plt.rcParams['axes.ymargin'] = 0.2
    ax = sns.scatterplot(x="performancetier", y="Medals", s=medal_groups['country']*20, data=medal_groups,
                         color=(0.2, 0.4, 0.6, 0.6))
    # plt.savefig(os.getcwd() + '/participant_figures/medals_rank.png', bbox_inches='tight')
    plt.show()
    return 0


if __name__ == '__main__':

    # private_leaderboard.txt is the html code (in string format) downloaded directly from the web url leaderboard
    with open('private_leaderboard.txt', 'r') as file:
        table_string = file.read()

    # Create a dictionary with team name : team members
    team_locations = [m.start() for m in re.finditer('data-th="Team Name" title=', table_string)]
    team_names = [table_string[team+27 : (team+29+table_string[team+29:].find('"'))] for team in team_locations]

    teams_members_dict = {}
    teams_scores = {}
    for ind, x in enumerate(team_locations):
        end_of_string = team_locations[ind+1] if ind+1 != len(team_locations) else len(table_string)
        temp_string = table_string[x:end_of_string]

        users_on_team = [m.start() for m in re.finditer('class="avatar" href="', temp_string)]
        team_participants = [temp_string[participant + 21: (participant + 23 + temp_string[participant + 23:].find('"'))]
                             for participant in users_on_team]

        teams_members_dict[team_names[ind]] = team_participants

        score_loc_start = temp_string.find('class="competition-leaderboard__td-score">')
        score_loc_end = temp_string.find('</td><td data-th="Number of Entries" class')
        score = float(temp_string[score_loc_start+42:score_loc_end])

        if ind != 3571:
            entires_loc_start = temp_string.find('class="competition-leaderboard__td-entries">')
            entires_loc_end = temp_string.find('</td><td data-th="Last Entry" class')
            entries = float(temp_string[entires_loc_start+44:entires_loc_end])
            teams_scores[team_names[ind]] = [ind+1, score, entries]

    participant_urls = [item for sublist in list(teams_members_dict.values()) for item in sublist]
    # get_participant_info(participant_urls)  # NO NEED TO RUN THIS ANYMORE -- ALL INFO SCRAPPED AND PICKLED
    participant_dict2 = pickle.load(open('/Users/jonathanroth/PycharmProjects/Kaggle_Demographics/kaggle_profiles_all2.obj', 'rb'))

    # COUNTRY PLOTS
    country_df = create_country_plots(participant_dict2)

    # PLOTS FOR KAGGLE RANKING // RANK VS. #COMPETITIONS // JOIN DATES
    rank_df = ranking_plots(participant_dict2)
    rank_competitions_df = rankVcompetitions(participant_dict2)
    joindate_df = join_dates(participant_dict2)

    # CREATE DATAFRAMES
    teams_scores_df = pd.DataFrame(teams_scores).T
    teams_scores_df.columns = ['Team #', 'Score', 'Entries']
    teams_scores_df['Team Name'] = teams_scores_df.index
    team_members_df = pd.DataFrame.from_dict(teams_members_dict, orient='index')
    team_members_df = pd.DataFrame([(k, member) for k, v in teams_members_dict.items() for member in v])
    team_members_df.columns = ['Team Name', 'Member']
    members_scores = pd.merge(team_members_df, teams_scores_df, on=['Team Name'])
    participants_df = pd.DataFrame(participant_dict2).drop(147)
    col_names = list(participants_df.columns)
    col_names[0] = 'Member'
    participants_df.columns = col_names
    final_df = pd.merge(participants_df, members_scores, on=['Member'])
    medals_df = final_df.iloc[0:578,]
    medals = np.concatenate([np.repeat('Winners', 11),
                             np.repeat('Gold', 24),
                             np.repeat('Silver', 285),
                             np.repeat('Bronze', 258)])

    medals_df['Medals'] = medals
    medal_groups = medals_df.groupby(['performancetier', 'Medals']).count()
    medal_groups.reset_index(inplace=True)
    medal_groups['Member'] = medal_groups['Member'].astype(str)
    medal_groups['user_vec'] = medal_groups['user_vec'].astype(str)
    medal_groups['order'] = [1,1,1,1,2,2,2,2,4,4,4,4,3,3,3,3,0,0,0]
    medal_groups = medal_groups.sort_values('order')
    medal_groups['order']
    medals_plots(medal_groups)

    # STATS!
    rankpercentage_vec = participant_dict2['rankpercentage']
    rankpercentage_vec = np.delete(rankpercentage_vec, 147)
    rankpercentage_vec = rankpercentage_vec.astype(float)
    percentage_ranked = len(list(filter(lambda a: a > 1, rankpercentage_vec))) / len(user_vec)
    print(final_df['Score'][final_df['performancetier'] == 'novice'].median())
    print(final_df['Score'][final_df['performancetier'] == 'contributor'].median())
    print(final_df['Score'][final_df['performancetier'] == 'expert'].median())
    print(final_df['Score'][final_df['performancetier'] == 'master'].median())
    print(final_df['Score'][final_df['performancetier'] == 'grandmaster'].median())
