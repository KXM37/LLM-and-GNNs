# -*- coding: utf-8 -*-
"""3025Term TF-IDF.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oNnOotUlg-F5Bu2TGPvKD38f5WW2xe_W
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz

# Read the CSV file
olddf = pd.read_csv('/home/kevin/ACM-PrePro/LLM-and-GNNs/3025SumTerm.csv')
df = olddf.copy() 
print("DataFrame loaded successfully.")

df['Summary'] = df['Summary'].astype(str)
df['Authors'] = df['Authors'].astype(str)
df['KeyTerms'] = df['KeyTerms'].astype(str)
print("Columns converted to strings and combined.")

# Join 'Citation', 'Paper', and 'Summary' columns into a single column
df.loc[:, 'combined'] = df.apply(lambda row: ' '.join(row[col] for col in ['Summary','Authors', 'KeyTerms']), axis=1)

# Assuming 'df' is a slice from a larger DataFrame
df.loc[:, 'combined'] = df['combined'].apply(lambda x: str(x).lower())

# Ensure each column is a string and convert to lowercase
df = df.applymap(lambda x: str(x).lower())
Sumt = df.copy()

# Provided stop words array
stopwords_array = np.array([
        [np.array(['a'])],
        [np.array(['able'])],
        [np.array(['summary'])],
        [np.array(['terms'])],
        [np.array(['key'])],
        [np.array(['about'])],
        [np.array(['above'])],
        [np.array(['abst'])],
        [np.array(['accordance'])],
        [np.array(['according'])],
        [np.array(['accordingly'])],
        [np.array(['across'])],
        [np.array(['act'])],
        [np.array(['actually'])],
        [np.array(['added'])],
        [np.array(['adj'])],
        [np.array(['adopted'])],
        [np.array(['affected'])],
        [np.array(['affecting'])],
        [np.array(['affects'])],
        [np.array(['after'])],
        [np.array(['afterwards'])],
        [np.array(['again'])],
        [np.array(['against'])],
        [np.array(['ah'])],
        [np.array(['all'])],
        [np.array(['almost'])],
        [np.array(['alone'])],
        [np.array(['along'])],
        [np.array(['already'])],
        [np.array(['also'])],
        [np.array(['although'])],
        [np.array(['always'])],
        [np.array(['am'])],
        [np.array(['among'])],
        [np.array(['amongst'])],
        [np.array(['an'])],
        [np.array(['and'])],
        [np.array(['announce'])],
        [np.array(['another'])],
        [np.array(['any'])],
        [np.array(['anybody'])],
        [np.array(['anyhow'])],
        [np.array(['anymore'])],
        [np.array(['anyone'])],
        [np.array(['anything'])],
        [np.array(['anyway'])],
        [np.array(['anyways'])],
        [np.array(['anywhere'])],
        [np.array(['apparently'])],
        [np.array(['approximately'])],
        [np.array(['are'])],
        [np.array(['aren'])],
        [np.array(['arent'])],
        [np.array(['arise'])],
        [np.array(['around'])],
        [np.array(['as'])],
        [np.array(['aside'])],
        [np.array(['ask'])],
        [np.array(['asking'])],
        [np.array(['at'])],
        [np.array(['auth'])],
        [np.array(['available'])],
        [np.array(['away'])],
        [np.array(['awfully'])],
        [np.array(['b'])],
        [np.array(['back'])],
        [np.array(['be'])],
        [np.array(['became'])],
        [np.array(['because'])],
        [np.array(['become'])],
        [np.array(['becomes'])],
        [np.array(['becoming'])],
        [np.array(['been'])],
        [np.array(['before'])],
        [np.array(['beforehand'])],
        [np.array(['begin'])],
        [np.array(['beginning'])],
        [np.array(['beginnings'])],
        [np.array(['begins'])],
        [np.array(['behind'])],
        [np.array(['being'])],
        [np.array(['believe'])],
        [np.array(['below'])],
        [np.array(['beside'])],
        [np.array(['besides'])],
        [np.array(['between'])],
        [np.array(['beyond'])],
        [np.array(['biol'])],
        [np.array(['both'])],
        [np.array(['brief'])],
        [np.array(['briefly'])],
        [np.array(['but'])],
        [np.array(['by'])],
        [np.array(['c'])],
        [np.array(['ca'])],
        [np.array(['came'])],
        [np.array(['can'])],
        [np.array(['cannot'])],
        [np.array(["can't"])],
        [np.array(['cause'])],
        [np.array(['causes'])],
        [np.array(['certain'])],
        [np.array(['certainly'])],
        [np.array(['co'])],
        [np.array(['com'])],
        [np.array(['come'])],
        [np.array(['comes'])],
        [np.array(['contain'])],
        [np.array(['containing'])],
        [np.array(['contains'])],
        [np.array(['could'])],
        [np.array(['couldnt'])],
        [np.array(['d'])],
        [np.array(['date'])],
        [np.array(['did'])],
        [np.array(["didn't"])],
        [np.array(['different'])],
        [np.array(['do'])],
        [np.array(['does'])],
        [np.array(["doesn't"])],
        [np.array(['doing'])],
        [np.array(['done'])],
        [np.array(["don't"])],
        [np.array(['down'])],
        [np.array(['downwards'])],
        [np.array(['due'])],
        [np.array(['during'])],
        [np.array(['e'])],
        [np.array(['each'])],
        [np.array(['ed'])],
        [np.array(['edu'])],
        [np.array(['effect'])],
        [np.array(['eg'])],
        [np.array(['eight'])],
        [np.array(['eighty'])],
        [np.array(['either'])],
        [np.array(['else'])],
        [np.array(['elsewhere'])],
        [np.array(['end'])],
        [np.array(['ending'])],
        [np.array(['enough'])],
        [np.array(['especially'])],
        [np.array(['et'])],
        [np.array(['et-al'])],
        [np.array(['etc'])],
        [np.array(['even'])],
        [np.array(['ever'])],
        [np.array(['every'])],
        [np.array(['everybody'])],
        [np.array(['everyone'])],
        [np.array(['everything'])],
        [np.array(['everywhere'])],
        [np.array(['ex'])],
        [np.array(['except'])],
        [np.array(['f'])],
        [np.array(['far'])],
        [np.array(['few'])],
        [np.array(['ff'])],
        [np.array(['fifth'])],
        [np.array(['first'])],
        [np.array(['five'])],
        [np.array(['fix'])],
        [np.array(['followed'])],
        [np.array(['following'])],
        [np.array(['follows'])],
        [np.array(['for'])],
        [np.array(['former'])],
        [np.array(['formerly'])],
        [np.array(['forth'])],
        [np.array(['found'])],
        [np.array(['four'])],
        [np.array(['from'])],
        [np.array(['further'])],
        [np.array(['furthermore'])],
        [np.array(['g'])],
        [np.array(['gave'])],
        [np.array(['get'])],
        [np.array(['gets'])],
        [np.array(['getting'])],
        [np.array(['give'])],
        [np.array(['given'])],
        [np.array(['gives'])],
        [np.array(['giving'])],
        [np.array(['go'])],
        [np.array(['goes'])],
        [np.array(['gone'])],
        [np.array(['got'])],
        [np.array(['gotten'])],
        [np.array(['h'])],
        [np.array(['had'])],
        [np.array(['happens'])],
        [np.array(['hardly'])],
        [np.array(['has'])],
        [np.array(["hasn't"])],
        [np.array(['have'])],
        [np.array(["haven't"])],
        [np.array(['having'])],
        [np.array(['he'])],
        [np.array(['hed'])],
        [np.array(['hence'])],
        [np.array(['her'])],
        [np.array(['here'])],
        [np.array(['hereafter'])],
        [np.array(['hereby'])],
        [np.array(['herein'])],
        [np.array(['heres'])],
        [np.array(['hereupon'])],
        [np.array(['hers'])],
        [np.array(['herself'])],
        [np.array(['hes'])],
        [np.array(['hi'])],
        [np.array(['hid'])],
        [np.array(['him'])],
        [np.array(['himself'])],
        [np.array(['his'])],
        [np.array(['hither'])],
        [np.array(['home'])],
        [np.array(['how'])],
        [np.array(['howbeit'])],
        [np.array(['however'])],
        [np.array(['hundred'])],
        [np.array(['i'])],
        [np.array(['id'])],
        [np.array(['ie'])],
        [np.array(['if'])],
        [np.array(["i'll"])],
        [np.array(['im'])],
        [np.array(['immediate'])],
        [np.array(['immediately'])],
        [np.array(['importance'])],
        [np.array(['important'])],
        [np.array(['in'])],
        [np.array(['inc'])],
        [np.array(['indeed'])],
        [np.array(['index'])],
        [np.array(['information'])],
        [np.array(['instead'])],
        [np.array(['into'])],
        [np.array(['invention'])],
        [np.array(['inward'])],
        [np.array(['is'])],
        [np.array(["isn't"])],
        [np.array(['it'])],
        [np.array(['itd'])],
        [np.array(["it'll"])],
        [np.array(['its'])],
        [np.array(['itself'])],
        [np.array(["i've"])],
        [np.array(['j'])],
        [np.array(['just'])],
        [np.array(['k'])],
        [np.array(['keep'])],
        [np.array(['keeps'])],
        [np.array(['kept'])],
        [np.array(['keys'])],
        [np.array(['kg'])],
        [np.array(['km'])],
        [np.array(['know'])],
        [np.array(['known'])],
        [np.array(['knows'])],
        [np.array(['l'])],
        [np.array(['largely'])],
        [np.array(['last'])],
        [np.array(['lately'])],
        [np.array(['later'])],
        [np.array(['latter'])],
        [np.array(['latterly'])],
        [np.array(['least'])],
        [np.array(['less'])],
        [np.array(['lest'])],
        [np.array(['let'])],
        [np.array(['lets'])],
        [np.array(['like'])],
        [np.array(['liked'])],
        [np.array(['likely'])],
        [np.array(['line'])],
        [np.array(['little'])],
        [np.array(["'ll"])],
        [np.array(['look'])],
        [np.array(['looking'])],
        [np.array(['looks'])],
        [np.array(['ltd'])],
        [np.array(['m'])],
        [np.array(['made'])],
        [np.array(['mainly'])],
        [np.array(['make'])],
        [np.array(['makes'])],
        [np.array(['many'])],
        [np.array(['may'])],
        [np.array(['maybe'])],
        [np.array(['me'])],
        [np.array(['mean'])],
        [np.array(['means'])],
        [np.array(['meantime'])],
        [np.array(['meanwhile'])],
        [np.array(['merely'])],
        [np.array(['mg'])],
        [np.array(['might'])],
        [np.array(['million'])],
        [np.array(['miss'])],
        [np.array(['ml'])],
        [np.array(['more'])],
        [np.array(['moreover'])],
        [np.array(['most'])],
        [np.array(['mostly'])],
        [np.array(['mr'])],
        [np.array(['mrs'])],
        [np.array(['much'])],
        [np.array(['mug'])],
        [np.array(['must'])],
        [np.array(['my'])],
        [np.array(['myself'])],
        [np.array(['n'])],
        [np.array(['na'])],
        [np.array(['name'])],
        [np.array(['namely'])],
        [np.array(['nay'])],
        [np.array(['nd'])],
        [np.array(['near'])],
        [np.array(['nearly'])],
        [np.array(['necessarily'])],
        [np.array(['necessary'])],
        [np.array(['need'])],
        [np.array(['needs'])],
        [np.array(['neither'])],
        [np.array(['never'])],
        [np.array(['nevertheless'])],
        [np.array(['new'])],
        [np.array(['next'])],
        [np.array(['nine'])],
        [np.array(['ninety'])],
        [np.array(['no'])],
        [np.array(['nobody'])],
        [np.array(['non'])],
        [np.array(['none'])],
        [np.array(['nonetheless'])],
        [np.array(['noone'])],
        [np.array(['nor'])],
        [np.array(['normally'])],
        [np.array(['nos'])],
        [np.array(['not'])],
        [np.array(['noted'])],
        [np.array(['nothing'])],
        [np.array(['now'])],
        [np.array(['nowhere'])],
        [np.array(['o'])],
        [np.array(['obtain'])],
        [np.array(['obtained'])],
        [np.array(['obviously'])],
        [np.array(['of'])],
        [np.array(['off'])],
        [np.array(['often'])],
        [np.array(['oh'])],
        [np.array(['ok'])],
        [np.array(['okay'])],
        [np.array(['old'])],
        [np.array(['omitted'])],
        [np.array(['on'])],
        [np.array(['once'])],
        [np.array(['one'])],
        [np.array(['ones'])],
        [np.array(['only'])],
        [np.array(['onto'])],
        [np.array(['or'])],
        [np.array(['ord'])],
        [np.array(['other'])],
        [np.array(['others'])],
        [np.array(['otherwise'])],
        [np.array(['ought'])],
        [np.array(['our'])],
        [np.array(['ours'])],
        [np.array(['ourselves'])],
        [np.array(['out'])],
        [np.array(['outside'])],
        [np.array(['over'])],
        [np.array(['overall'])],
        [np.array(['owing'])],
        [np.array(['own'])],
        [np.array(['p'])],
        [np.array(['page'])],
        [np.array(['pages'])],
        [np.array(['part'])],
        [np.array(['particular'])],
        [np.array(['particularly'])],
        [np.array(['past'])],
        [np.array(['per'])],
        [np.array(['perhaps'])],
        [np.array(['placed'])],
        [np.array(['please'])],
        [np.array(['plus'])],
        [np.array(['poorly'])],
        [np.array(['possible'])],
        [np.array(['possibly'])],
        [np.array(['potentially'])],
        [np.array(['pp'])],
        [np.array(['predominantly'])],
        [np.array(['present'])],
        [np.array(['previously'])],
        [np.array(['primarily'])],
        [np.array(['probably'])],
        [np.array(['promptly'])],
        [np.array(['proud'])],
        [np.array(['provides'])],
        [np.array(['put'])],
        [np.array(['q'])],
        [np.array(['que'])],
        [np.array(['quickly'])],
        [np.array(['quite'])],
        [np.array(['qv'])],
        [np.array(['r'])],
        [np.array(['ran'])],
        [np.array(['rather'])],
        [np.array(['rd'])],
        [np.array(['re'])],
        [np.array(['readily'])],
        [np.array(['really'])],
        [np.array(['recent'])],
        [np.array(['recently'])],
        [np.array(['ref'])],
        [np.array(['refs'])],
        [np.array(['regarding'])],
        [np.array(['regardless'])],
        [np.array(['regards'])],
        [np.array(['related'])],
        [np.array(['relatively'])],
        [np.array(['research'])],
        [np.array(['respectively'])],
        [np.array(['resulted'])],
        [np.array(['resulting'])],
        [np.array(['results'])],
        [np.array(['right'])],
        [np.array(['run'])],
        [np.array(['s'])],
        [np.array(['said'])],
        [np.array(['same'])],
        [np.array(['saw'])],
        [np.array(['say'])],
        [np.array(['saying'])],
        [np.array(['says'])],
        [np.array(['sec'])],
        [np.array(['section'])],
        [np.array(['see'])],
        [np.array(['seeing'])],
        [np.array(['seem'])],
        [np.array(['seemed'])],
        [np.array(['seeming'])],
        [np.array(['seems'])],
        [np.array(['seen'])],
        [np.array(['self'])],
        [np.array(['selves'])],
        [np.array(['sent'])],
        [np.array(['seven'])],
        [np.array(['several'])],
        [np.array(['shall'])],
        [np.array(['she'])],
        [np.array(['shed'])],
        [np.array(["she'll"])],
        [np.array(['shes'])],
        [np.array(['should'])],
        [np.array(["shouldn't"])],
        [np.array(['show'])],
        [np.array(['showed'])],
        [np.array(['shown'])],
        [np.array(['showns'])],
        [np.array(['shows'])],
        [np.array(['significant'])],
        [np.array(['significantly'])],
        [np.array(['similar'])],
        [np.array(['similarly'])],
        [np.array(['since'])],
        [np.array(['six'])],
        [np.array(['slightly'])],
        [np.array(['so'])],
        [np.array(['some'])],
        [np.array(['somebody'])],
        [np.array(['somehow'])],
        [np.array(['someone'])],
        [np.array(['somethan'])],
        [np.array(['something'])],
        [np.array(['sometime'])],
        [np.array(['sometimes'])],
        [np.array(['somewhat'])],
        [np.array(['somewhere'])],
        [np.array(['soon'])],
        [np.array(['sorry'])],
        [np.array(['specifically'])],
        [np.array(['specified'])],
        [np.array(['specify'])],
        [np.array(['specifying'])],
        [np.array(['state'])],
        [np.array(['states'])],
        [np.array(['still'])],
        [np.array(['stop'])],
        [np.array(['strongly'])],
        [np.array(['sub'])],
        [np.array(['substantially'])],
        [np.array(['successfully'])],
        [np.array(['such'])],
        [np.array(['sufficiently'])],
        [np.array(['suggest'])],
        [np.array(['sup'])],
        [np.array(['sure'])],
        [np.array(['t'])],
        [np.array(['take'])],
        [np.array(['taken'])],
        [np.array(['taking'])],
        [np.array(['tell'])],
        [np.array(['tends'])],
        [np.array(['th'])],
        [np.array(['than'])],
        [np.array(['thank'])],
        [np.array(['thanks'])],
        [np.array(['thanx'])],
        [np.array(['that'])],
        [np.array(["that'll"])],
        [np.array(['thats'])],
        [np.array(["that've"])],
        [np.array(['the'])],
        [np.array(['their'])],
        [np.array(['theirs'])],
        [np.array(['them'])],
        [np.array(['themselves'])],
        [np.array(['then'])],
        [np.array(['thence'])],
        [np.array(['there'])],
        [np.array(['thereafter'])],
        [np.array(['thereby'])],
        [np.array(['thered'])],
        [np.array(['therefore'])],
        [np.array(['therein'])],
        [np.array(["there'll"])],
        [np.array(['thereof'])],
        [np.array(['therere'])],
        [np.array(['theres'])],
        [np.array(['thereto'])],
        [np.array(['thereupon'])],
        [np.array(["there've"])],
        [np.array(['these'])],
        [np.array(['they'])],
        [np.array(['theyd'])],
        [np.array(["they'll"])],
        [np.array(['theyre'])],
        [np.array(["they've"])],
        [np.array(['think'])],
        [np.array(['this'])],
        [np.array(['those'])],
        [np.array(['thou'])],
        [np.array(['though'])],
        [np.array(['thoughh'])],
        [np.array(['thousand'])],
        [np.array(['throug'])],
        [np.array(['through'])],
        [np.array(['throughout'])],
        [np.array(['thru'])],
        [np.array(['thus'])],
        [np.array(['til'])],
        [np.array(['tip'])],
        [np.array(['to'])],
        [np.array(['together'])],
        [np.array(['too'])],
        [np.array(['took'])],
        [np.array(['toward'])],
        [np.array(['towards'])],
        [np.array(['tried'])],
        [np.array(['tries'])],
        [np.array(['truly'])],
        [np.array(['try'])],
        [np.array(['trying'])],
        [np.array(['ts'])],
        [np.array(['twice'])],
        [np.array(['two'])],
        [np.array(['u'])],
        [np.array(['un'])],
        [np.array(['under'])],
        [np.array(['unfortunately'])],
        [np.array(['unless'])],
        [np.array(['unlike'])],
        [np.array(['unlikely'])],
        [np.array(['until'])],
        [np.array(['unto'])],
        [np.array(['up'])],
        [np.array(['upon'])],
        [np.array(['ups'])],
        [np.array(['us'])],
        [np.array(['use'])],
        [np.array(['used'])],
        [np.array(['useful'])],
        [np.array(['usefully'])],
        [np.array(['usefulness'])],
        [np.array(['uses'])],
        [np.array(['using'])],
        [np.array(['usually'])],
        [np.array(['v'])],
        [np.array(['value'])],
        [np.array(['various'])],
        [np.array(["'ve"])],
        [np.array(['very'])],
        [np.array(['via'])],
        [np.array(['viz'])],
        [np.array(['vol'])],
        [np.array(['vols'])],
        [np.array(['vs'])],
        [np.array(['w'])],
        [np.array(['want'])],
        [np.array(['wants'])],
        [np.array(['was'])],
        [np.array(["wasn't"])],
        [np.array(['way'])],
        [np.array(['we'])],
        [np.array(['wed'])],
        [np.array(['welcome'])],
        [np.array(["we'll"])],
        [np.array(['went'])],
        [np.array(['were'])],
        [np.array(["weren't"])],
        [np.array(["we've"])],
        [np.array(['what'])],
        [np.array(['whatever'])],
        [np.array(["what'll"])],
        [np.array(['whats'])],
        [np.array(['when'])],
        [np.array(['whence'])],
        [np.array(['whenever'])],
        [np.array(['where'])],
        [np.array(['whereafter'])],
        [np.array(['whereas'])],
        [np.array(['whereby'])],
        [np.array(['wherein'])],
        [np.array(['wheres'])],
        [np.array(['whereupon'])],
        [np.array(['wherever'])],
        [np.array(['whether'])],
        [np.array(['which'])],
        [np.array(['while'])],
        [np.array(['whim'])],
        [np.array(['whither'])],
        [np.array(['who'])],
        [np.array(['whod'])],
        [np.array(['whoever'])],
        [np.array(['whole'])],
        [np.array(["who'll"])],
        [np.array(['whom'])],
        [np.array(['whomever'])],
        [np.array(['whos'])],
        [np.array(['whose'])],
        [np.array(['why'])],
        [np.array(['widely'])],
        [np.array(['willing'])],
        [np.array(['wish'])],
        [np.array(['with'])],
        [np.array(['within'])],
        [np.array(['without'])],
        [np.array(["won't"])],
        [np.array(['words'])],
        [np.array(['world'])],
        [np.array(['would'])],
        [np.array(["wouldn't"])],
        [np.array(['www'])],
        [np.array(['x'])],
        [np.array(['y'])],
        [np.array(['yes'])],
        [np.array(['yet'])],
        [np.array(['you'])],
        [np.array(['youd'])],
        [np.array(["you'll"])],
        [np.array(['your'])],
        [np.array(['youre'])],
        [np.array(['yours'])],
        [np.array(['yourself'])],
        [np.array(['yourselves'])],
        [np.array(["you've"])],
        [np.array(['z'])],
        [np.array(['zero'])]
], dtype=object)

# Extract stop words from the nested structure
stopWords = [word[0][0] for word in stopwords_array]

# Initialize TF-IDF Vectorizer with custom stop words
vectorizer = TfidfVectorizer(stop_words=stopWords)

# Assuming 'df' is your DataFrame and 'combined' is the column with text data
tfidf_matrix = vectorizer.fit_transform(df['combined'])
print("TF-IDF vectorization completed.")

# Sum TF-IDF scores for each word across all documents
sum_scores = tfidf_matrix.sum(axis=0)

# Get feature names
words = vectorizer.get_feature_names_out()

# Create a dictionary of word and its corresponding sum of scores
word_scores = {word: sum_scores[0, idx] for idx, word in enumerate(words)}

# Sort the words based on scores
sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

# Select the top 1900 words
top_words = sorted_words[:1900]
print(top_words[:9])

# add the top words to new term array
top1900 = []
for word, score in top_words:
  top1900.append(word)

# Creating a Series with indices
series = pd.Series(top1900, name='terms')

# Save Series to CSV
series.to_csv('termsIndex.csv', header=True, index_label='Index')
print("Top words extracted and saved.")

# Load the terms DataFrame (if not already loaded)
termsdf = pd.read_csv('/home/kevin/ACM-PrePro/LLM-and-GNNs/termsIndex.csv')

# Create the terms dictionary for quick lookup
term_dict = {term: idx for idx, term in termsdf['terms'].items()}

# Number of papers and terms
num_papers = len(Sumt)

num_terms = len(termsdf)

# Create an empty LIL sparse matrix
TvsP_matrix = lil_matrix((num_terms, num_papers), dtype=float)

# Populate the matrix
for index, row in df.iterrows():
    if index % 100 == 0:  # Print progress for every 100th paper
        print(f"Processing paper {index}/{num_papers}")
    # Split the 'combined' column's string into individual terms
    paper_terms = row['combined'].split()

    for term in paper_terms:
        term_index = term_dict.get(term)
        if term_index is not None:
            TvsP_matrix[term_index, index] = 1.0

TvsP = TvsP_matrix.tocsr()
print("Matrix population completed.")
# Save the CSR matrix to a file
save_npz('/home/kevin/ACM-PrePro/LLM-and-GNNs/TvP.npz', TvsP)
print("Matrix saved to file.")
