import pandas as pd
'''
To get the historical record of how far each team advanced in the playoffs.
'''
url = "https://www.basketball-reference.com/playoffs/series.html"
raw_playoffs_data = pd.read_html(url)
df = pd.DataFrame(raw_playoffs_data[0])
'''
Explore and Handle Raw Data.
'''
print(df.head())
df.info()
df = df.drop(df.columns[[1, 3, 4, 6, 7, 9, 10, 11, 12]], axis=1) #drop the complete null(4, 7, 10) and the value are the totally same(1)columns. irrealted(3, 11, 12) (6, 9)
df = df.set_axis(['year', 'series', 'winner', 'loser'], axis=1) #unnamed columns cause so many problem so name them
df = df.dropna(how='all') #drop the null row
df = df[df['year'] != 'Yr'] #drop the non data row
df['year'] = df['year'].astype('int64') 
df = df[df['year'] >= 1984] #drop the row with the year before merge
print(df.head())
df.info()

df_final = df[df['series'] == 'Finals']
df_champ = df_final.drop(columns=['series', 'loser'])
df_champ.info()
#print(df_champ)

df_finals = df_final.drop(columns=['series', 'winner'])
df_finals.info()
#print(df_finals)

df_east_conf = df[df['series'] == 'Eastern Conf Finals']
df_west_conf = df[df['series'] == 'Western Conf Finals']
df_conf_finals = pd.concat([df_east_conf, df_west_conf])
df_conf_finals = df_conf_finals.drop(columns=['series', 'winner'])
df_conf_finals.info()
#print(df_conf_finals)

df_east_semi = df[df['series'] == 'Eastern Conf Semifinals']
df_west_semi = df[df['series'] == 'Western Conf Semifinals']
df_semi = pd.concat([df_east_semi, df_west_semi])
df_semi = df_semi.drop(columns=['series', 'winner'])
df_semi.info()
#print(df_semi)

df_east_first_round = df[df['series'] == 'Eastern Conf First Round']
df_west_first_round = df[df['series'] == 'Western Conf First Round']
df_first_round = pd.concat([df_east_first_round, df_west_first_round])
df_first_round = df_first_round.drop(columns=['series', 'winner'])
df_first_round.info()
#print(df_first_round)

df_champ = df_champ.assign(rounds=4)
df_finals = df_finals.assign(rounds=3)
df_conf_finals = df_conf_finals.assign(rounds=2)
df_semi = df_semi.assign(rounds=1)
df_first_round = df_first_round.assign(rounds=0)
# print(df_champ)
# print(df_finals)
# print(df_conf_finals)
# print(df_semi)
# print(df_first_round)

df_champ = df_champ.rename(columns={'winner': 'loser'})
df_record = pd.concat([df_champ, df_finals, df_conf_finals, df_semi, df_first_round])
df_record.info()
df_record['loser'] = df_record['loser'].str[:-3]
df_record['team'] = df_record['year'].astype(str)+' '+df_record['loser']
df_record = df_record.drop(columns={'year', 'loser'})
df_record['team'] = df_record['team'].str.strip()
new_order = ['team', 'rounds']
df_record = df_record[new_order]
print(df_record)

df_record.to_csv('../data/Playoffs_Records.csv', index=False)