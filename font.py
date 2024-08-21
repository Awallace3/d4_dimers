import matplotlib as mpl
from matplotlib.font_manager import findfont, FontProperties
import matplotlib.pyplot as plt

# rebuild font cache
font = findfont(FontProperties(family=['sans-serif']))
font = findfont(FontProperties(family=['helvetica']))
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        "mathtext.fontset": "custom",
    }
)
x, y = range(5), range(5)
plt.plot(x, y)
plt.title(f'Font: {font}')
plt.show()
