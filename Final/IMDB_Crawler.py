import pandas as pd
import requests
from bs4 import BeautifulSoup


def crawler(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    movie_data = soup.findAll('div', class_='lister-item mode-advanced')

    movie_list = []

    for store in movie_data:
        try:
            name = store.h3.a.text.strip()
            year = store.h3.find('span',
                                 class_='lister-item-year').text.strip("()")
            runtime = store.p.find('span',
                                   class_='runtime').text.replace(" min", "")
            genre = store.p.find('span',
                                 class_='genre').text.replace(", ",
                                                              ",").strip()
            rating = store.find(
                'div',
                class_='inline-block ratings-imdb-rating').strong.text.strip()
        except AttributeError:
            continue  # Skip this item if any required field is missing

        value = store.find_all('span', attrs={'name': 'nv'})
        votes = value[0].text if len(value) > 0 else "0"
        gross = value[1].text if len(value) > 1 else "*****"

        try:
            cast_info = store.find("p", class_="").text.replace('\n',
                                                                '').split('|')
            cast_info = [s.strip() for s in cast_info]
            director = cast_info[0].replace("Director:", "").strip()
            stars = cast_info[1].replace("Stars:", "").strip().split(",")
            stars = [s.strip() for s in stars]
        except:
            director = ""
            stars = []

        movie_list.append({
            "Name of movie": name,
            "Year of relase": year,
            "Watchtime": runtime,
            "theme": genre,
            "Votes": votes,
            "Gross collection": gross,
            "Director": director,
            "Star": stars,
            "Movie Rating": rating
        })

    return pd.DataFrame(movie_list)


def turnpages():
    full_df = pd.DataFrame()
    for i in range(1, 601, 100):
        url = f'https://www.imdb.com/search/title/?groups=top_1000&sort=user_rating,desc&count=100'
        if i != 1:
            url += f'&start={i}&ref_=adv_nxt'
        print(f'Crawling: {url}')
        df = crawler(url)
        full_df = pd.concat([full_df, df], ignore_index=True)
    return full_df


if __name__ == "__main__":
    movie_df = turnpages()
    movie_df.index += 1
    movie_df.to_csv("Top_600_IMDB_Movies.csv",
                    index=False,
                    encoding="utf-8-sig")
