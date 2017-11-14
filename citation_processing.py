import snap
import pandas as pd
import numpy as np
import csv

file_path = '~/uspatentcitation.tsv'
df = pd.read_table(file_path)
df.head(4)

# with open(file_path) as tsvfile:
# 	reader = csv.reader(file_path)

# 	for row in reader:
# 		print row
