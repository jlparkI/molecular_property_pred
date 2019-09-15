import numpy as np, rdkit, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

constdict = dict()
soy = open('train.csv')
soy.readline()
for line in soy:
	couple = line.split(',')[4]
	const = float(line.strip().split(',')[5])
	if couple not in constdict:
		constdict[couple] = [const]
	else:
		constdict[couple].append(const)

soy.close()

constlist = []
typelist = []
soy = open('train.csv')
soy.readline()
for line in soy:
	typelist.append(line.split(',')[4])
	constlist.append(float(line.strip().split(',')[5]))

soy.close()
d = {'Coupling constant type':typelist, 'Coupling constant (hz)':constlist}

df = pd.DataFrame.from_dict(d)
sns.boxenplot(x='Coupling constant type', y='Coupling constant (hz)',
	data=df)
plt.title('Coupling constant distribution for each category in the dataset')
plt.show()
