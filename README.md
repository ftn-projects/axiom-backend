# axiom-backend

Konzolna aplikacija razvijena u Python-u koja generiše teze za učenje (skripte) iz tekstualnih materijala. 

## Definicija problema

Mnogi prilikom učenja (pogotovo studenti) koriste materijale u vidu skripata. Ideja je da se ovi materijali ne pišu ručno jer to obično zahteva dosta vremena, već da budu generisani.

## Skup podataka

Podaci su dobijeni sa [Studocu](https://www.studocu.com) web platforme za deljenje teza za učenje. Inicijalni skup podataka je u vidu PDF datoteka.

## Pretprocesiranje podataka

Pre tokenizacije teksta skripata PDF datoteke bi se parsirale u običan tekst.

## Metodologija

GPT-2 model se "fine-tune" pomoću podataka.

## Tehnologije

Python, PyTorch, Hugging Face

## Literatura

[Hugging Face Docs](https://huggingface.co/docs/transformers/)
[PyTorch Docs](https://pytorch.org/docs)
