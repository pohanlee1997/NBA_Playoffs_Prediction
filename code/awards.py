import pandas as pd

#MVP:
url = 'https://www.espn.com/nba/history/awards/_/id/33'
df = pd.read_html(url)
MVP = pd.DataFrame(df[0])
'''
MVP.info()
print(MVP.columns)
print(MVP)
'''
MVP = MVP.drop(index=[MVP.index[0], MVP.index[1]])
MVP = MVP.drop(MVP.columns[[1, 2, 4, 5, 6, 7, 8, 9]], axis=1)
MVP = MVP.set_axis(['year', 'team'], axis=1)
MVP['year'] = MVP['year'].astype(int)
MVP = MVP[MVP['year'] >= 1984]
MVP['team'] = MVP['year'].astype(str)+' '+MVP['team']
MVP=MVP.drop(columns=['year'])
MVP['MVP']=1
# MVP.info()
# print(MVP.columns)
# print(MVP)
MVP.to_csv('../data/MVP.csv', index=False)

#DPOY:
url = 'https://www.espn.com/nba/history/awards/_/id/39'
df = pd.read_html(url)
DPOY = pd.DataFrame(df[0])
'''
DPOY.info()
print(DPOY.columns)
print(DPOY)
'''
DPOY = DPOY.drop(index=[DPOY.index[0], DPOY.index[1]])
DPOY = DPOY.drop(DPOY.columns[[1, 2, 4, 5, 6, 7, 8, 9]], axis=1)
DPOY = DPOY.set_axis(['year', 'team'], axis=1)
DPOY['year'] = DPOY['year'].astype(int)
DPOY = DPOY[DPOY['year'] >= 1984]
DPOY['team'] = DPOY['year'].astype(str)+' '+DPOY['team']
DPOY=DPOY.drop(columns=['year'])
DPOY['DPOY']=1
# DPOY.info()
# print(DPOY.columns)
# print(DPOY)
DPOY.to_csv('../data/DPOY.csv', index=False)

#sixth 
url = 'https://www.espn.com/nba/history/awards/_/id/40'
df = pd.read_html(url)
sixth = pd.DataFrame(df[0])
'''
sixth.info()
print(sixth.columns)
print(sixth)
'''
sixth = sixth.drop(index=[sixth.index[0], sixth.index[1]])
sixth = sixth.drop(sixth.columns[[1, 2, 4, 5, 6, 7, 8, 9]], axis=1)
sixth = sixth.set_axis(['year', 'team'], axis=1)
sixth['year'] = sixth['year'].astype(int)
sixth = sixth[sixth['year'] >= 1984]
sixth['team'] = sixth['year'].astype(str)+' '+sixth['team']
sixth=sixth.drop(columns=['year'])
sixth['sixth']=1
# sixth.info()
# print(sixth.columns)
# print(sixth)
sixth.to_csv('../data/sixth.csv', index=False)

#COY
url = 'https://www.espn.com/nba/history/awards/_/id/34'
df = pd.read_html(url)
COY = pd.DataFrame(df[0])
'''
COY.info()
print(COY.columns)
print(COY)
'''
COY = COY.drop(index=[COY.index[0], COY.index[1]])
COY = COY.drop(COY.columns[[1, 3, 4, 5, 6, 7, 8, 9]], axis=1)
COY = COY.set_axis(['year', 'team'], axis=1)
COY['year'] = COY['year'].astype(int)
COY = COY[COY['year'] >= 1984]
COY['team'] = COY['year'].astype(str)+' '+COY['team']
COY=COY.drop(columns=['year'])
COY['COY']=1
# COY.info()
# print(COY.columns)
# print(COY)
COY.to_csv('../data/COY.csv', index=False)