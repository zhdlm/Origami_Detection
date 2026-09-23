import igor2
from pprint import pprint
import matplotlib.pyplot as plt

file = r"E:\KUN\1hour\DMEM_purify_ps_rect_to_tri_37C_1hour0000.ibw"

f = igor2.binarywave.load(file)

pprint(f["wave"]["wData"])
fig, ax = plt.subplots(ncols=8)
for i in range(8):
    ax[i].imshow(f["wave"]["wData"][:,:,i], cmap="grey")
plt.show()