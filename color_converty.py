import pandas as pd
from PIL import Image, ImageCms
import math
from sklearn.cluster import KMeans

picture = Image.open("C:/Users/ethan/OneDrive/Documents/Python Scripts/Image Changer/lighting.jpg").convert("RGB")

print(picture.size[0], picture.size[1])

simple_color_list = []
for x in range(picture.size[0]):
    for y in range(picture.size[1]):
        r, g, b = picture.getpixel((x, y))[0:3]
        color = [r, g, b]
        simple_color_list.append(color)

df_colors =pd.DataFrame(simple_color_list,columns=['R','G', 'B'])
print("Starting Training")
model = KMeans(n_clusters=8)
model.fit(df_colors)
print("Finishing Training")

colors = model.cluster_centers_.round(0).astype(int)

#Puts Pixels into picture
print(len(model.labels_))
for index in range(len(model.labels_)):
    x = (index // picture.size[1])
    y = (index - (x * picture.size[1])) % picture.size[1]

    picture.putpixel( (x, y), tuple(colors[model.labels_[index]]))
    if(index % 100000 == 0):
        print(index)
picture.show()
picture.save("C:/Users/ethan/OneDrive/Documents/Python Scripts/Image Changer/Final.png")
