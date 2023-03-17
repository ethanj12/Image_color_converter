import pandas as pd
from PIL import Image, ImageCms
from sklearn.cluster import KMeans


name_of_file = "woman_with_lights.jpg"
NUM_OF_COLORS = 9
picture = Image.open(
    f"C:/Users/ethan/Documents/Python/Color Converter/Images/{name_of_file}").convert("RGB")
print(f"Size of Image: {picture.size[0]}x{picture.size[1]}")

# Goes through the original image and gets the R, G, and B Values for each of the pixels in the image
simple_color_list = []
for x in range(picture.size[0]):
    for y in range(picture.size[1]):
        r, g, b = picture.getpixel((x, y))[0:3]
        color = [r, g, b]
        simple_color_list.append(color)

# Converts the list into a pandas dataframe for training in sklearn
df_colors = pd.DataFrame(simple_color_list, columns=['R', 'G', 'B'])
print("Starting Training")
model = KMeans(n_clusters=NUM_OF_COLORS, n_init=10)
model.fit(df_colors)
print("Finishing Training")

# Gets all of the predicted values for the image and converts them to int to replace old pixels in picture variable
colors = model.cluster_centers_.round(0).astype(int)

# Puts Pixels into picture
for index in range(len(model.labels_)):
    x = (index // picture.size[1])
    y = (index - (x * picture.size[1])) % picture.size[1]

    picture.putpixel((x, y), tuple(colors[model.labels_[index]]))
    if (index % 100000 == 0):
        print(f"Number of Pixels Replaced: {index}")
        
picture.save(f"C:/Users/ethan/Documents/Python/Color Converter/Finished Images/Final_{name_of_file}")
picture.show()
