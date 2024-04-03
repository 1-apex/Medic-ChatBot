from bs4 import BeautifulSoup
import requests


def scrape_wiki(keyword):
    html_text = requests.get(f"https://en.wikipedia.org/wiki/{keyword}").text
    soup = BeautifulSoup(html_text, 'lxml')

    blocks = soup.find('div', class_="mw-body-content")

    if blocks:
        para_tags = blocks.find_all('p')

        with open('data.txt', 'w', encoding='utf-8') as f:
            for para in para_tags:
                f.write(para.text + '\n')

# scrape_wiki('crocin')