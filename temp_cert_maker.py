from urllib.request import urlopen
from bs4 import BeautifulSoup as soup

url = 'https://bio-atlas.psu.edu'
response = urlopen(url).read()
print(response)
