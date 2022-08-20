# License: cc-by-4.0

# download the Wiktionary file
curl -O https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.json

# data example ... 
# {"pos": "noun", "heads": [{"template_name": "en-noun"}], "forms": [{"form": "zymurgies", "tags": ["plural"]}], "word": "zymurgy", "lang": "English", "lang_code": "en", "senses": [{"glosses": ["The chemistry of fermentation with yeasts, especially the science involved in beer and winemaking."], "derived": [{"word": "zymurgic"}, {"word": "zymurgical"}, {"word": "zymurgist"}], "related": [{"word": "zythepsary"}], "categories": ["Beer", "Zymurgy"], "id": "zymurgy-noun"}]}

# clean the downloaded Wiktionary file
python construct_wiktionary.py

# remove the original file
rm kaikki.org-dictionary-English.json