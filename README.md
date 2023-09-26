# ProjetoSQL-ebac

import math
from typing import Iterator
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

cases = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/01-12-2021.csv', sep=',') # extraçao de dados de casos da universidade John Hopkins.
cases.head()

def date_range(start_date: datetime, end_date: datetime) -> Iterator[datetime]: # iterar dentro de um intervalo de tempo definido para extrair.
  date_range_days: int = (end_date - start_date).days
  for lag in range(date_range_days):
    yield start_date + timedelta(lag)

start_date = datetime(2021,  1,  1)
end_date   = datetime(2021, 12, 31)

cases = None # selecionar as colunas de interesse e as linhas referentes ao Brasil.
cases_is_empty = True

for date in date_range(start_date=start_date, end_date=end_date):

  date_str = date.strftime('%m-%d-%Y')
  data_source_url = f'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/{date_str}.csv'

  case = pd.read_csv(data_source_url, sep=',')

  case = case.drop(['FIPS', 'Admin2', 'Last_Update', 'Lat', 'Long_', 'Recovered', 'Active', 'Combined_Key', 'Case_Fatality_Ratio'], axis=1)
  case = case.query('Country_Region == "Brazil"').reset_index(drop=True)
  case['Date'] = pd.to_datetime(date.strftime('%Y-%m-%d'))

  if cases_is_empty:
    cases = case
    cases_is_empty = False
  else:
    cases = cases.append(case, ignore_index=True)

cases.query('Province_State == "Sao Paulo"').head()

# wrangling casos
cases.head()
cases.shape
cases.info()

vaccines = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv', sep=',', parse_dates=[3], infer_datetime_format=True) # dados de vacinação da universidade de Oxford.
vaccines = vaccines.query('location == "Brazil"').reset_index(drop=True)
vaccines = vaccines[['location', 'population', 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'total_boosters', 'date']]
vaccines.head()
# wrangling vacinas
vaccines.head()

vaccines.shape
vaccines.info()
vaccines = vaccines.fillna(method='ffill') # tratar os dados faltantes, a estratégia será a de preencher os buracos com o valor anterior válido mais próximo.

#Transformação
cases = cases.rename( # editar nome das colunas
  columns={
    'Province_State': 'state',
    'Country_Region': 'country'
  }
)

for col in cases.columns:
  cases = cases.rename(columns={col: col.lower()})

states_map = { # ajustar nome dos estados
    'Amapa': 'Amapá',
    'Ceara': 'Ceará',
    'Espirito Santo': 'Espírito Santo',
    'Goias': 'Goiás',
    'Para': 'Pará',
    'Paraiba': 'Paraíba',
    'Parana': 'Paraná',
    'Piaui': 'Piauí',
    'Rondonia': 'Rondônia',
    'Sao Paulo': 'São Paulo'
}

cases['state'] = cases['state'].apply(lambda state: states_map.get(state) if state in states_map.keys() else state)

cases['month'] = cases['date'].apply(lambda date: date.strftime('%Y-%m')) # computar chaves temporais
cases['year']  = cases['date'].apply(lambda date: date.strftime('%Y'))

cases['population'] = round(100000 * (cases['confirmed'] / cases['incident_rate'])) # população estimada do estado
cases = cases.drop('incident_rate', axis=1)

cases_ = None # Número, média móvel (7 dias) e estabilidade (14 dias) de casos e mortes por estado
cases_is_empty = True

def get_trend(rate: float) -> str:

  if np.isnan(rate):
    return np.NaN

  if rate < 0.85:
    status = 'downward'
  elif rate > 1.15:
    status = 'upward'
  else:
    status = 'stable'

  return status


for state in cases['state'].drop_duplicates():

  cases_per_state = cases.query(f'state == "{state}"').reset_index(drop=True)
  cases_per_state = cases_per_state.sort_values(by=['date'])

  cases_per_state['confirmed_1d'] = cases_per_state['confirmed'].diff(periods=1)
  cases_per_state['confirmed_moving_avg_7d'] = np.ceil(cases_per_state['confirmed_1d'].rolling(window=7).mean())
  cases_per_state['confirmed_moving_avg_7d_rate_14d'] = cases_per_state['confirmed_moving_avg_7d']/cases_per_state['confirmed_moving_avg_7d'].shift(periods=14)
  cases_per_state['confirmed_trend'] = cases_per_state['confirmed_moving_avg_7d_rate_14d'].apply(get_trend)

  cases_per_state['deaths_1d'] = cases_per_state['deaths'].diff(periods=1)
  cases_per_state['deaths_moving_avg_7d'] = np.ceil(cases_per_state['deaths_1d'].rolling(window=7).mean())
  cases_per_state['deaths_moving_avg_7d_rate_14d'] = cases_per_state['deaths_moving_avg_7d']/cases_per_state['deaths_moving_avg_7d'].shift(periods=14)
  cases_per_state['deaths_trend'] = cases_per_state['deaths_moving_avg_7d_rate_14d'].apply(get_trend)

  if cases_is_empty:
    cases_ = cases_per_state
    cases_is_empty = False
  else:
    cases_ = cases_.append(cases_per_state, ignore_index=True)

cases = cases_
cases_ = None

# Vacinas: filtrar a base de dados de acordo com a coluna date para garantir que ambas as bases de dados tratam do mesmo período de tempo.
vaccines = vaccines[(vaccines['date'] >= '2021-01-01') & (vaccines['date'] <= '2021-12-31')].reset_index(drop=True)

vaccines = vaccines.rename( # alterar o nome das colunas.
  columns={
    'location': 'country',
    'total_vaccinations': 'total',
    'people_vaccinated': 'one_shot',
    'people_fully_vaccinated': 'two_shots',
    'total_boosters': 'three_shots',
  }
)

vaccines['month'] = vaccines['date'].apply(lambda date: date.strftime('%Y-%m')) # chaves temporais
vaccines['year']  = vaccines['date'].apply(lambda date: date.strftime('%Y'))

vaccines['one_shot_perc'] = round(vaccines['one_shot'] / vaccines['population'], 4) # dados relativos
vaccines['two_shots_perc'] = round(vaccines['two_shots'] / vaccines['population'], 4)
vaccines['three_shots_perc'] = round(vaccines['three_shots'] / vaccines['population'], 4)

vaccines['population'] = vaccines['population'].astype('Int64') # type casting das colunas.
vaccines['total'] = vaccines['total'].astype('Int64')
vaccines['one_shot'] = vaccines['one_shot'].astype('Int64')
vaccines['two_shots'] = vaccines['two_shots'].astype('Int64')
vaccines['three_shots'] = vaccines['three_shots'].astype('Int64')

vaccines = vaccines[['date', 'country', 'population', 'total', 'one_shot', 'one_shot_perc', 'two_shots', 'two_shots_perc', 'three_shots', 'three_shots_perc', 'month', 'year']] # reorganizar as colunas e conferir o resultado final.

vaccines.tail()

#Carregamento
cases.to_csv('./covid-cases.csv', sep=',', index=False)
vaccines.to_csv('./covid-vaccines.csv', sep=',', index=False)
