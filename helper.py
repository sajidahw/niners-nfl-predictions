import pandas as pd
import requests
from bs4 import BeautifulSoup
# bs4 is py library for pulling data out of HTML, XML files
# request allows sending of HTTP requests using python

symbols = ['CRD', 'ATL', 'RAV', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GNB', 'HTX', 'CLT', 'JAX', 'KAN', 'RAI', 'SDG', 'RAM', 'MIA', 'MIN', 'NWE', 'NOR', 'NYG', 'NYJ', 'PHI', 'PIT', 'SFO', 'SEA', 'TAM', 'OTI', 'WAS']
team_names = ['Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills', 'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns', 'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers', 'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Kansas City Chiefs', 'Las Vegas Raiders', 'Los Angeles Chargers', 'Los Angeles Rams', 'Miami Dolphins', 'Minnesota Vikings', 'New England Patriots', 'New Orleans Saints', 'New York Giants', 'New York Jets', 'Philadelphia Eagles', 'Pittsburgh Steelers', 'San Francisco 49ers', 'Seattle Seahawks', 'Tampa Bay Buccaneers', 'Tennessee Titans', 'Washington Football Team']



def get_new_data(team, year):
    '''
    Function to pull NFL stats from Pro Football Reference (https://www.pro-football-reference.com/).
    
    - team : team name (str)
    - year : year (int)
    '''
    # pull data
    sym = symbols[team_names.index(team)].lower()
    url = f'https://www.pro-football-reference.com/teams/{sym}/{year}.htm'
    html = requests.get(url).text

    # parse the data
    soup = BeautifulSoup(html,'html.parser')
    table = soup.find('table', id='games')
    tablerows = table.find_all('tr')[2:]
    data = []

    for tablerow in tablerows:
        data.append([tabledata.get_text(strip=True) for tabledata in tablerow.find_all('td')])

    df = pd.DataFrame(data)


    # subset
    index = [0,1,4,8,9,10] + list(range(11,21))
    new_data = df.iloc[:,index].copy()

    # rename columns
    col_names = ['day', 'date', 'result', 'opponent', 'tm_score', 'opp_score', '1stD_offense', 'TotYd_offense', 'PassY_offense', 'RushY_offense', 'TO_offense', '1stD_defense', 'TotYd_defense', 'PassY_defense', 'RushY_defense', 'TO_defense']
    new_data.columns = col_names

    # encode results
    result_encoder = {'result': {'L': 0, 'T': 0,'W': 1,'' : pd.NA},
                     'TO_offense' : {'' : 0},
                     'TO_defense' : {'' : 0}}
    new_data.replace(result_encoder, inplace=True)

    # remove future dates
    new_data = new_data[new_data.result.notnull()]

    # add week variable back
    week = list(range(1,len(new_data)+1))
    new_data.insert(0, 'week', week)
    
    # add team name
    tn_col = pd.Series([f'{team}']).repeat(len(new_data)).reset_index(drop=True)
    new_data.insert(0, 'team_name', tn_col)

    # return a dataframe object
    if type(new_data) == pd.Series:
        new_data = new_data.to_frame().T
        return new_data.reset_index(drop=True)
    else:
        return new_data.reset_index(drop=True)
