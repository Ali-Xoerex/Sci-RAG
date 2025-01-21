import requests
from bs4 import BeautifulSoup
from scholarly import scholarly
query = 'ENTER YOUR QUERY HERE'
search_query = scholarly.search_pubs(query)
document = next(search_query) # get the first search result
r = requests.get("https://sci-hub.se/"+document["pub_url"]) # using only one url for now
soup = BeautifulSoup(r.text, 'html.parser')
url = "http://"+soup.find("div", {"id": "article"}).find("embed").attrs["src"][2:] # find the url of the pdf, in case it exists
response = requests.get(url, stream=True)

# save it to a pdf file
with open(document["bib"]["title"]+".pdf", "wb") as handle:
    for data in response.iter_content():
        handle.write(data)