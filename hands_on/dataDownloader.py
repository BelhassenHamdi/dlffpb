import requests #pip3 install requests
import pandas as pd #pip install pandas
import json
from pandas.io.json import json_normalize
from bs4 import BeautifulSoup

url = 'https://stationdata.wunderground.com/cgi-bin/stationlookup?station=KMAHADLE7&units=both&v=2.0&format=json&callback=jQuery1720724027235122559_1542743885014&_=15'
res = requests.get(url)
soup = BeautifulSoup(res.content, "lxml")
s = soup.select('html')[0].text.strip('jQuery1720724027235122559_1542743885014(').strip(')')
s = s.replace('null','"placeholder"')
data= json.loads(s)
data = json_normalize(data)
df = pd.DataFrame(data)
print(df)