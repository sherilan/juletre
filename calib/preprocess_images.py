import pathlib 
import os 
import utils 

folder = pathlib.Path(__file__).parent / 'data'
tree = 'tree1'
view = '4'
view_folder = folder / 'calibration' / tree / view
data_folder = folder / 'preprocessing' / tree
data_path = data_folder / f'{view}.csv' 

os.makedirs(data_folder, exist_ok=True)

data = utils.preprocess_data(
    view_folder=view_folder,
    j1=400, j2=900,
)
data.to_csv(data_path)