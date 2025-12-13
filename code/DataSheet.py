import pandas as pd

df = pd.read_csv('../data/Regular_season_stats.csv')
df.info()
print(df.columns)
print(df.head())
df['Team'] = df['Team'].str.replace('*', '')
df['Team'] = df['year'].astype(str)+' '+df['Team']
df = df.drop(columns=['year', 'PW', 'PL', 'FG', '3P', '2P','FT', "TRB"])
df = df.rename(columns={'Team': 'team'})

MVP = pd.read_csv('../data/MVP.csv')
df = df.merge(MVP, on='team', how='left')

DPOY = pd.read_csv('../data/DPOY.csv')
df = df.merge(DPOY, on='team', how='left')

sixth = pd.read_csv('../data/sixth.csv')
df =df.merge(sixth, on='team', how='left')

COY = pd.read_csv('../data/COY.csv')
df = df.merge(COY, on='team', how='left')

label = pd.read_csv('../data/Playoffs_Records.csv')
df =df.merge(label, on='team', how='left')

df['MVP'] = df['MVP'].fillna(0).astype(int)
df['DPOY'] = df['DPOY'].fillna(0).astype(int)
df['sixth'] = df['sixth'].fillna(0).astype(int)
df['COY'] = df['COY'].fillna(0).astype(int)
df['rounds'] = df['rounds'].fillna(-1).astype(int)

df.info()
print(df.columns)
print(df.head())

df.to_csv('../data/DataSheet.csv', index=False)