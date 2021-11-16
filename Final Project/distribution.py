import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 14)
y = [2.9454460397664364,
         11.319226471174266,
         11.016965997147508,
         22.050459496080535,
         9.794929667066727,
         15.633631363522158,
         7.806456777862537,
         10.666786928273375,
         19.1436048538587,
         10.10604858671961,
         16.489758197540592,
         20.56515485018235,
         7.057356440635417]

print(np.mean(y))
fig, ax = plt.subplots()
ax.bar(x, y)

fig.set_figwidth(12)    #  ширина Figure
fig.set_figheight(6)    #  высота Figure
plt.xlabel('Number of peak')
plt.ylabel('D')
plt.title('D distribution')
plt.savefig('picd1.png', dpi=900)
plt.show()
