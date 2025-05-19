from pathlib import Path
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from gensim import models, corpora
import pickle

BASE = Path(__file__).resolve().parent.parent
model_path = BASE / "outputs" / "lda_model.gensim"
dict_path  = BASE / "outputs" / "lda_dictionary.dict"
corpus_path = BASE / "outputs" / "lda_corpus.mm"

lda = models.LdaModel.load(str(model_path))
dictionary = corpora.Dictionary.load(str(dict_path))
corpus = corpora.MmCorpus(str(corpus_path))

vis_data = gensimvis.prepare(lda, corpus, dictionary)
pyLDAvis.save_html(vis_data, str(BASE / "outputs" / "lda_vis.html"))

print("✅ lda_vis.html dosyası outputs klasörüne kaydedildi.")
