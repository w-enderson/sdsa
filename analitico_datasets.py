from os import read
from numpy.core.fromnumeric import repeat
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats.morestats import levene
import statsmodels.stats.multicomp as multicomp


data_climates = pd.read_csv('./datasets/{}.csv'.format('climates')) 
data_climates['banco de dados'] = repeat('climates', repeats=len(data_climates))
data_dry_climates = pd.read_csv('./datasets/{}.csv'.format('dry-climates')) 
data_dry_climates['banco de dados'] = repeat('dry-climates', repeats=len(data_dry_climates))
data_european_climates = pd.read_csv('./datasets/{}.csv'.format('european-climates')) 
data_european_climates['banco de dados'] = repeat('european-climates', repeats=len(data_european_climates))
data_mushroom = pd.read_csv('./datasets/{}.csv'.format('mushroom')) 
data_mushroom['banco de dados'] = repeat('mushroom', repeats=len(data_mushroom))
data_climas = pd.concat([data_climates, data_dry_climates, data_european_climates])

#médias das temperaturas minimas do ano
print(data_climas.groupby('banco de dados').mean().iloc[:,0:24:2].mean(1)) 

#médias das temparaturas maximas do ano
print(data_climas.groupby('banco de dados').mean().iloc[:,1:24:2].mean(1)) #médias das temparaturas maximas do ano

#médias por estação do ano e por clima
print(data_climas.groupby('banco de dados').mean().iloc[:,24:32])  

#médias das variáveis do banco de dados de cogumelos
print(data_mushroom.mean().to_latex())


#teste ANOVA

results_ivabc = pd.read_csv('./results_test/{}/climates,dry-climates,european-climates,mushroom.csv'.format('ivabc'))
results_sdsa = pd.read_csv('./results_test/{}/climates,dry-climates,european-climates,mushroom.csv'.format('sdsa'))
results_sdsa_lr = pd.read_csv('./results_test/{}/climates,dry-climates,european-climates,mushroom.csv'.format('sdsa_lr'))
results_sdsa_lr_not_update = pd.read_csv('./results_test/{}/climates,dry-climates,european-climates,mushroom.csv'.format('sdsa_lr_not_update'))
results_sdsa_not_update = pd.read_csv('./results_test/{}/climates,dry-climates,european-climates,mushroom.csv'.format('sdsa_not_update'))
results_sdsa_rf = pd.read_csv('./results_test/{}/climates,dry-climates,european-climates,mushroom.csv'.format('sdsa_rf'))
results_sdsa_rf_not_update = pd.read_csv('./results_test/{}/climates,dry-climates,european-climates,mushroom.csv'.format('sdsa_rf_not_update'))
results_sdsa_svc = pd.read_csv('./results_test/{}/climates,dry-climates,european-climates,mushroom.csv'.format('sdsa_svc'))
results_sdsa_svc_not_update = pd.read_csv('./results_test/{}/climates,dry-climates,european-climates,mushroom.csv'.format('sdsa_svc_not_update'))

results_data = pd.concat([results_ivabc, results_sdsa,results_sdsa_lr, results_sdsa_lr_not_update, results_sdsa_not_update,
results_sdsa_rf,results_sdsa_rf_not_update,results_sdsa_svc,results_sdsa_svc_not_update])

medias = [np.mean(results_ivabc.acc), np.mean(results_sdsa.acc),np.mean(results_sdsa_lr.acc), np.mean(results_sdsa_lr_not_update.acc), np.mean(results_sdsa_not_update.acc),
np.mean(results_sdsa_rf.acc),np.mean(results_sdsa_rf_not_update.acc),np.mean(results_sdsa_svc.acc),np.mean(results_sdsa_svc_not_update.acc)]

#ANOVA

teste = stats.f_oneway([np.mean(results_ivabc.acc), np.mean(results_sdsa.acc),np.mean(results_sdsa_lr.acc), np.mean(results_sdsa_lr_not_update.acc), np.mean(results_sdsa_not_update.acc),
np.mean(results_sdsa_rf.acc),np.mean(results_sdsa_rf_not_update.acc),np.mean(results_sdsa_svc.acc),np.mean(results_sdsa_svc_not_update.acc)])
print(teste)


