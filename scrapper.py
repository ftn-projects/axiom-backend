from urllib.request import urlopen
import re

def find_paragraphs(table_matches, html, paragraph_matches, paragraphs_in_tables_indices):
    for table_match in table_matches:
        table_start_index = html.find(table_match)
        table_end_index = table_start_index + len(table_match)
        for i, paragraph_match in enumerate(paragraph_matches):
            paragraph_start_index = html.find(paragraph_match)
            if paragraph_start_index >= table_start_index and paragraph_start_index < table_end_index:
                paragraphs_in_tables_indices.add(i)

def remove_html_tags_and_entities(text):
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    text = text.replace('&#160;', ' ') 
    return text

def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '', name)

def print_transcript(title, paragraph_matches, paragraphs_in_tables_indices):
    title = sanitize_filename(title)
    f = open(title + ".txt", "w", encoding='utf-8')

    for i, match in enumerate(paragraph_matches):
        if i not in paragraphs_in_tables_indices:
            clean_text = remove_html_tags_and_entities(match)
            f.write(clean_text)

    f.close()

def extract_all_pages():
    url = "https://regularshow.fandom.com/wiki/Category:Transcripts"

    page = urlopen(url)
    html_bytes = page.read()
    html = html_bytes.decode("utf-8")

    a_tag_pattern = re.compile(r'<a[^>]*?href="([^"]*?/Transcript)"[^>]*?>')

    a_tag_matches = a_tag_pattern.findall(html)

    links = []

    for link in a_tag_matches:
        links.append("https://regularshow.fandom.com" + link)

    return links


def main():
    links = extract_all_pages()

    for url in links:
        page = urlopen(url)
        html_bytes = page.read()
        html = html_bytes.decode("utf-8")

        paragraph_pattern = re.compile(r'<p>(.*?)</p>', re.DOTALL)
        paragraph_matches = paragraph_pattern.findall(html)

        table_pattern = re.compile(r'<table.*?</table>', re.DOTALL)
        table_matches = table_pattern.findall(html)

        paragraphs_in_tables_indices = set()

        title_pattern = re.compile(r'<span[^>]*class="mw-page-title-main"[^>]*>(.*?)</span>', re.DOTALL)
        title_match = title_pattern.search(html)

        if title_match:
            page_title = title_match.group(1)
            title = page_title[:-11]
        else:
            h1_pattern = re.compile(r'<h1[^>]*id="firstHeading"[^>]*>(.*?)</h1>', re.DOTALL)
            h1_match = h1_pattern.search(html)

            page_title = h1_match.group(1)

            pattern = re.compile(r'"(.*?)"')
    
            title = pattern.findall(page_title)[0]


        find_paragraphs(table_matches, html, paragraph_matches, paragraphs_in_tables_indices)
        print_transcript(title, paragraph_matches, paragraphs_in_tables_indices)

if __name__ == "__main__":
    main()