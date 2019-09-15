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

x = np.load('train_3JHNx_0.npy')[:,:,0:28]
y = np.load('train_3JHNy_0.npy')

x2 = np.load('train_1JHNx_0.npy')[:,:,0:28]
y2 = np.load('train_1JHNy_0.npy')

x3 = np.load('train_2JHNx_0.npy')[:,:,0:28]
y3 = np.load('train_2JHNy_0.npy')

jhn1charges = np.zeros((x2.shape[0],2))
jhn2charges = np.zeros((x3.shape[0],2))
jhn3charges = np.zeros((x.shape[0],2))


for i in range(0, x.shape[0]):
	for j in range(0, x.shape[1]):
		if x[i,j,24]==1:
			if x[i,j,5] == 1:
				jhn3charges[i,0] = x[i,j,14]
			else:
				jhn3charges[i,1] = x[i,j,14]


for i in range(0, x2.shape[0]):
	for j in range(0, x2.shape[1]):
		if x2[i,j,24]==1:
			if x2[i,j,5] == 1:
				jhn1charges[i,0] = x2[i,j,14]
			else:
				jhn1charges[i,1] = x2[i,j,14]



for i in range(0, x3.shape[0]):
	for j in range(0, x3.shape[1]):
		if x3[i,j,24]==1:
			if x3[i,j,5] == 1:
				jhn2charges[i,0] = x3[i,j,14]
			else:
				jhn2charges[i,1] = x3[i,j,14]

import seaborn as sns
fig, axes = plt.subplots(nrows=2,ncols=3)
axes[0][0].hexbin(jhn1charges[:,0], y2, cmap='Blues')
axes[1][0].hexbin(jhn1charges[:,1], y2, cmap='Blues')
axes[0][1].hexbin(jhn2charges[:,0], y3, cmap='Reds')
axes[1][1].hexbin(jhn2charges[:,1], y3, cmap='Reds')
axes[0][2].hexbin(jhn3charges[:,0], y, cmap='Greens')
axes[1][2].hexbin(jhn3charges[:,1], y, cmap='Greens')
axes[0][0].set_title('1JHN, charge on nitrogen vs.\ncoupling constant')
axes[1][0].set_title('1JHN, charge on hydrogen vs.\ncoupling constant')
axes[0][1].set_title('2JHN, charge on nitrogen vs.\ncoupling constant')
axes[1][1].set_title('2JHN, charge on hydrogen vs.\ncoupling constant')
axes[0][2].set_title('3JHN, charge on nitrogen vs.\ncoupling constant')
axes[1][2].set_title('3JHN, charge on hydrogen vs.\ncoupling constant')
plt.tight_layout()


for i in range(0, 2):
	for j in range(0,3):
		axes[i][j].set_xlabel('Partial charge')
		axes[i][j].set_ylabel('Coupling constant (hz)')

plt.show()


