# Map from string to number

import pandas as pd

# load from csv
df = pd.read_csv('conservation_actions.csv')
print(df)

# count effectiveness categories
categories = df['effectiveness'].unique()
print(categories)
print(len(categories))

# convert string to number
mapping = dict()

# negative categories (0)
mapping['Likely to be ineffective or harmful'] = 0
mapping['Unlikely to be beneficial'] = 0

# neutral categories (1)
mapping['Unknown effectiveness (limited evidence)'] = 1
mapping['No evidence found (no assessment)'] = 1
mapping['Evidence not assessed'] = 1
mapping['Awaiting assessment'] = 1
mapping['Trade-off between benefit and harms'] = 1

# positive categories (2)
mapping['Likely to be beneficial'] = 2
mapping['Beneficial'] = 2

df['effectiveness_number'] = df['effectiveness'].map(mapping)

print(df['effectiveness_number'].value_counts())
print(df)

df.to_csv('conservation_actions_bigger.csv')