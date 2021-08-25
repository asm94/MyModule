# Plotter.py
## plot_confusion_matrix(matrix, lebales, title=None)
Used to visualize the confusion matrix. The "matrix" is must be square, and the number of "labels" must be equal to the length of one side of the square.
### Usage
```python
import numpy as np
cm = np.random.randint(0,100,(3,3))
plot_confusion_matrix(cm, ['label1','label2','label3'], title='Matrix')
```
![example_confusion_matrix](https://github.com/asm94/MyModule/blob/image/example__plot_confusion_matrix.png)


## plot_beeswarm(data, x_label=None, y_label=None, x_ticklabels=[])
Used to create a beeswarm. For numerical data columns only. The "x_ticklabels" assumes a list, the number of which must be the same as the number of columns of numerical data.
### Usage
```python
import seaborn as sns
data = sns.load_dataset("iris")
plot_beeswarm(data, x_label='Item name', y_label='Size', x_ticklabels=data.drop('species',axis=1).columns)
```
![example_beeswarm](https://github.com/asm94/MyModule/blob/image/example__plot_beeswarm.png)


## plot_2D_scatters(data, ticklabels=None, group=None, display_correlation=False)
Used to create a sccaters. Create a scatter plot with all column combinations. You can stratify the data by specifying a "group" of the same length as the "data".
### Usage
```python
import seaborn as sns
data = sns.load_dataset("iris")
plot_2D_scatters(data.drop('species',axis=1), ticklabels=None, group=data['species'], display_correlation=True)
```
![example_2Dscatters](https://github.com/asm94/MyModule/blob/image/example__plot_2D_scatters.png)
