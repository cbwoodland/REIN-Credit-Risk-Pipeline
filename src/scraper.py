from bs4 import BeautifulSoup
import pandas as pd
import requests

url = "https://www.bisnow.com/news/atlanta"

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

print(f"Fetching data from {url}...")
response = requests.get(url, headers=headers)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")
    scraped_articles = []

    articles = soup.find_all("article")
    if not articles:
        articles = soup.find_all("a", href=True)

    for item in articles:
        if item.name == "article":
            title_element = item.find("h2") or item.find("h3") or item.find("a")
            link_element = item.find("a", href=True)
            title = title_element.text.strip() if title_element else None
            link = link_element["href"] if link_element else None
        else:
            title = item.text.strip()
            link = item["href"]

        if title and link and len(title) > 20 and "/news/" in link:
            full_link = link if link.startswith("http") else f"https://www.bisnow.com{link}"
            scraped_articles.append({
                "title": title,
                "link": full_link
            })

    df = pd.DataFrame(scraped_articles).drop_duplicates(subset=["link"]).reset_index(drop=True)

    print(f"\nSuccessfully scraped {len(df)} unique article entries!")
    print("\n--- FIRST 5 SCRAPED HEADLINES ---")
    print(df.head())

    df.to_csv("atlanta_cre_news.csv", index=False)
    print("\nSaved raw headlines to 'atlanta_cre_news.csv'")

else:
    print(f"Request failed with status code: {response.status_code}")