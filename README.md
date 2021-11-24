# DeepFashion6

Originally, this was a challenge about fashion image classification. We have thousands of fashion images, and want to classify them into different classes. Of the classification, it specified 6 categories (or dimensions), such as sleeve, neckline, texture, decoration pattern and so on. And for each category, it has several classes.

|Category 1 |Category 2   |Category 3 |Category 4       |Category 5 |Category 6   |
|-----------|-------------|-----------|-----------------|-----------|-------------|
|floral     |long_sleeve  |maxi_length|crew_neckline    |denim      |tight        |
|graphic    |short_sleeve |mini_length|v_neckline       |chiffon    |loose        |
|striped    |sleeveless   |no_dress   |squared_neckline |cotton     |conventional |
|embroidered|             |           |no_neckline      |leather    |             |
|pleated    |             |           |                 |faux       |             |
|solid      |             |           |                 |knit       |             |
|lattice    |             |           |                 |           |             |

This means, for an individual image, we need to predict 6 class labels, and each class label corresponds to one category.

The dataset has bounding box and landmarks.

| ![image](https://user-images.githubusercontent.com/41459515/143170241-bcd4012b-ab9d-429e-89c7-98b7fd89ef04.png) | ![image](https://user-images.githubusercontent.com/41459515/143169385-3274c8dc-d9ca-45a9-b504-2b80f7b291c8.png) |![image](https://user-images.githubusercontent.com/41459515/143169404-da158a5f-94f9-4eb9-b3c2-b9ea74769ca2.png) | ![image](https://user-images.githubusercontent.com/41459515/143169482-9acf1433-3347-42d7-a2ab-fca9c562e712.png) |
| ------------- | ------------- | ------------- | ------------- |







