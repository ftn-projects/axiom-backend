# regular-show-generator
Konzolna aplikacija razvijena u Python-u koja generiše transkripte epizoda Regular Show. 

## Definicija problema

Rešava se problem generisanja Regular Show transkripata epizoda putem treniranog transformer modela.

## Motivacija

Glavna motivacija je upoznavanje sa radom sa radom transformer LLM modela, njihovog treniranja i evaluacije. Izabrali smo Regular Show za generisanje jer su epizode obično kratke, radnja je epizodna i zato što bismo hteli još epizoda iako je serija završena.

## Skup podataka

Neobrađen skup podataka se nalazi na web sajtu [Regular Show Wiki](https://regularshow.fandom.com/wiki/Category:Transcripts). Obim skupa je 269 epizoda.

## Pretprocesiranje podataka

Budući da su instance podataka u vidu HTML-a, potrebno ih je web scrape-ovati i sačuvati za tokenizaciju.

## Metodologija

Koristiće se pre-trained transformer model (GPT2 ili neki drugi) koji će potom biti fine-tunovan sa epizodama (80%) iz skupa podataka koje su prošle proces tokenizacije. Istreniran model će kao izlaz imati transkript generisane epizode upisan u tekstualnu datoteku.

## Evaluacija

Evaluacija modela će biti sprovedena kroz **Rouge** (*Recall Oriented Understudy for Gisting Evaluation*) metriku koja će da proverava relevantnost generisanog teksta i predikcija. Iz skupa podataka će biti izdvojeno 53 epizode (20%) za svrhe evaluacije modela.

## Tehnologije

Implementacija projekta bi bila u Python-u. Web scraping će biti sprovedeno koriščenjem ugrađene Python biblioteke **urllib**. Za kreiranje, tokenizaciju, evaluciju modela i tokenizaciju ulaznog skupa podataka biće korišćena biblioteka **Hugging face**.

## Fine-tuned modeli

Na [Google Drive Folderu]() su dostupni istrenirani modeli sa sledećim kombinacijama parametara:
- 4 epohe, 4 batch size
- 3 epohe, 4 batch size
- 3 epohe, 2 batch size (sa 8 koraka akumulacije gradijenta)
Tokom evaluacije najbolje rezultate je postigao model `E4_B4_finetuned_gpt2` (treniran u 4 epohe sa 4 batch size), ali je empirijski utvrđeno da ne generiše dovoljno kreativne traskripte. Sledeći model sa najboljim Rouge metrikama je `E3_B4_finetuned_gpt2` koji je ubedljivo bolji od poslednjeg.

## Literatura

[Python Docs urllib](https://docs.python.org/3/library/urllib.html)
[Hugging Face Docs](https://huggingface.co/docs/transformers/)
