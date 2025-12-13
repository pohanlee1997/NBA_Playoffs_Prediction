import pandas as pd
#for the data from 1984~2015 seasons
def get_data(yr):
	url = 'https://www.basketball-reference.com/leagues/NBA_'+str(yr)+'.html'
	teams_data = pd.read_html(url)
	per_game_stats = pd.DataFrame(teams_data[2])
	advanced_stats = pd.DataFrame(teams_data[8])
	advanced_stats.columns = advanced_stats.columns.get_level_values(1)
	league_average = advanced_stats[advanced_stats['Team'] == 'League Average']
	advanced_stats = advanced_stats[advanced_stats['Team'] != 'League Average']
	new_columns = list(advanced_stats)
	new_columns[18] = 'Offense_eFG%'
	new_columns[19] = 'Offense_TOV%'
	new_columns[20] = 'Offense_ORB%'
	new_columns[21] = 'Offense_FT/FGA'
	new_columns[23] = 'Defense_eFG%'
	new_columns[24] = 'Defense_TOV%'
	new_columns[25] = 'Defense_ORB%'
	new_columns[26] = 'Defense_FT/FGA'
	advanced_stats.columns = new_columns
	teams_stats = pd.merge(per_game_stats, advanced_stats, on='Team')
	teams_stats = teams_stats.drop(teams_stats.columns[[0, 2, 25, 41, 46, 51, 52, 53, 54]], axis=1)
	teams_stats = teams_stats.assign(year=yr)
	return teams_stats
#for the data from 2016~2025 seasons
def get_data_2(yr):
	url = 'https://www.basketball-reference.com/leagues/NBA_'+str(yr)+'.html'
	teams_data = pd.read_html(url)
	per_game_stats = pd.DataFrame(teams_data[4])
	advanced_stats = pd.DataFrame(teams_data[10])
	advanced_stats.columns = advanced_stats.columns.get_level_values(1)
	league_average = advanced_stats[advanced_stats['Team'] == 'League Average']
	advanced_stats = advanced_stats[advanced_stats['Team'] != 'League Average']
	new_columns = list(advanced_stats)
	new_columns[18] = 'Offense_eFG%'
	new_columns[19] = 'Offense_TOV%'
	new_columns[20] = 'Offense_ORB%'
	new_columns[21] = 'Offense_FT/FGA'
	new_columns[23] = 'Defense_eFG%'
	new_columns[24] = 'Defense_TOV%'
	new_columns[25] = 'Defense_ORB%'
	new_columns[26] = 'Defense_FT/FGA'
	advanced_stats.columns = new_columns
	teams_stats = pd.merge(per_game_stats, advanced_stats, on='Team')
	teams_stats = teams_stats.drop(teams_stats.columns[[0, 2, 25, 41, 46, 51, 52, 53, 54]], axis=1)
	teams_stats = teams_stats.assign(year=yr)
	return teams_stats

teams_stats = get_data(1984)
for i in range(1985, 2016):
	teams_stats_get = get_data(i)
	teams_stats = pd.concat([teams_stats, teams_stats_get])
for i in range(2016, 2026):
	teams_stats_get = get_data_2(i)
	teams_stats = pd.concat([teams_stats, teams_stats_get])

teams_stats.info()
print(teams_stats)
teams_stats.to_csv('../data/Regular_season_stats.csv', index=False)