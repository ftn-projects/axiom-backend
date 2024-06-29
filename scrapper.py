import random
from urllib.request import urlopen
import re


BASE_URL = 'https://regularshow.fandom.com'
TEST_DIR = 'test/'
TRAIN_DIR = 'train/'
TARGET_SUCCESSES=54
PROBABILITY=0.2


def get_page_html(page_url: str):
    return urlopen(page_url).read().decode("utf-8") 


def get_episode_links(contents_url: str) -> list:
    html = get_page_html(contents_url)
    pattern = re.compile(r'<a[^>]*?href="([^"]*?/Transcript)"[^>]*?>')
    return [BASE_URL + link for link in pattern.findall(html)]


def get_episode_title(html) -> str:
    span_title_pattern = re.compile(r'<span[^>]*class="mw-page-title-main"[^>]*>(.*?)</span>', re.DOTALL)
    h1_title_pattern = re.compile(r'<h1[^>]*id="firstHeading"[^>]*>(.*?)</h1>', re.DOTALL) 

    match = span_title_pattern.search(html)
    if match:
        title = match.group(1)[:-11]
    else:
        match = h1_title_pattern.search(html).group(1)
        title = re.compile(r'"(.*?)"').findall(match)[0]

    return re.sub(r'[<>:"/\\|?*]', '', title)


def paragraphs_outside_tables(html: str) -> list:
    paragraph_pattern = re.compile(r'<p>(.*?)</p>', re.DOTALL)
    table_pattern = re.compile(r'<table.*?</table>', re.DOTALL)

    table_ranges = {(m.start(), m.end()) for m in table_pattern.finditer(html)}
    return [p for p in paragraph_pattern.findall(html) if all(not (s <= html.find(p) < e) for s, e in table_ranges)]


def cleanup_html(html) -> str:
    return re.sub('<.*?>', '', html).replace('&#160;', ' ')


def load_episode(episode_url: str) -> tuple:
    html = get_page_html(episode_url)

    paragraphs = paragraphs_outside_tables(html)
    raw_transcript = ''.join(paragraphs)

    return get_episode_title(html), cleanup_html(raw_transcript)


def main():
    contents_page1 = BASE_URL + '/wiki/Category:Transcripts'
    contents_page2 = contents_page1 + '?from=Space+Escape%2FTranscript'
    links = get_episode_links(contents_page1) + get_episode_links(contents_page2)

    max_train = 0
    for link in links:
        title, transcript = load_episode(link)
        if random.random() <= PROBABILITY and max_train<=TARGET_SUCCESSES:
            max_train+=1
            with open(TEST_DIR + title + ".txt", "w", encoding='utf-8') as f:
                f.write(transcript)
        else:
            with open(TRAIN_DIR + title + ".txt", "w", encoding='utf-8') as f:
                f.write(transcript) 


if __name__ == "__main__":
    main()
