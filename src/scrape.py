# Scrape conservation actions data
# Website: https://www.conservationevidence.com/data/index?pp=100&page=1#interventions

import requests
from bs4 import BeautifulSoup
import pandas as pd

i = 0

actions_text = []
effectiveness_text = []

for i in range(1, 37): # 36 pages
    print('Retrieving page {}...'.format(i))
    page = requests.get("https://www.conservationevidence.com/data/index?pp=100&page={}#interventions".format(i))

    if page.status_code != 200:
        print('Could not retrieve page.')
        break
    
    soup = BeautifulSoup(page.content, 'html.parser')
    
    actions = soup.find_all('p', class_='title')
    effectiveness = soup.find_all('span', class_='effectiveness')

    if(len(actions) != len(effectiveness)):
        print('Error: Number of actions and effectiveness labels are not equal.')
    
    print(len(actions))

    actions_text.extend([a.get_text().strip() for a in actions])
    effectiveness_text.extend([e.get_text().strip() for e in effectiveness])

conservation_df = pd.DataFrame({
    "action": actions_text,
    "effectiveness": effectiveness_text
})
print(conservation_df)
conservation_df.to_csv('conservation_actions.csv')