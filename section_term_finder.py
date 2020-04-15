
# import packages
import pandas as pd
import numpy as np
import re
import os
os.getcwd()

# Read the data
df = pd.read_csv("C:/Users/kiruthika.parthiban/Desktop/EHR/section.csv",encoding = "ISO-8859-1")

notes = df['Notes']

# Extract the history of present illness section
hopi = []
for val in notes:

    # one or more spaces and newlines
    regex_multi_space   = re.compile(r' +')
    regex_multi_newline = re.compile(r'\n+')

    # collapse repeated newlines into a single newline
    single_newline_report = regex_multi_newline.sub('\n', val)

     # collapse repeated spaces into a single space
    clean_report = regex_multi_space.sub(' ', single_newline_report).lower()


    with open("file.txt", "w") as att_file:
            att_file.write(clean_report + "\n")

    length_of_note = len(val)
    clean_report = clean_report.replace("history of the present illness:", "history of present illness:")

    start_offset = clean_report.find("history of present illness:")
    print(start_offset)

    first_offset = clean_report.find(":",start_offset,length_of_note)
    print(first_offset)

    end_offset = clean_report.find(":",first_offset+1,length_of_note)
    print(end_offset)

    hopi.append(clean_report[start_offset:end_offset][0:clean_report[start_offset:end_offset].rfind("\n")].replace("history of present illness:",''))

list_of_hopi = list(filter(None, hopi))


# create indexing for each history sections
Paragraph_no = []
i=0
for val in list_of_hopi:
    Paragraph_no.append('para'+str(i))
    i+=1

# create a dataframe
data = { 'para_no': Paragraph_no, 'Notes': list_of_hopi}
notes_df = pd.DataFrame(data)

# tokenization - sentence and word
import nltk
import string
from nltk import word_tokenize, pos_tag, pos_tag_sents,sent_tokenize
paragraph = [nltk.sent_tokenize(val) for val in notes_df['Notes']]

# create senetence numbering
i=0
j=0
lines = []
para_no = []
sent_no = []
for para in paragraph:
    j+=1
    for sent in para:
        lines.append(sent)
        para_no.append(j)
        sent_no.append(i)
        i+=1

# create a dataframe
data = { 'para_no': para_no, 'sent_no': sent_no, 'lines': lines}
lines_df = pd.DataFrame(data)

# Remove punctuations found in the columns
lines_df['lines'] = lines_df["lines"].apply(lambda x:''.join([i for i in x
                                                  if i not in string.punctuation]))

# Remove stop words from the sentences
from nltk.corpus import stopwords
stop = stopwords.words('english')

lines_df['lines'] = lines_df['lines'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

l = lines_df.dropna()

l = lines_df[pd.notnull(lines_df['lines'])]

# importing all necessery modules
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd

text = " ".join(review for review in lines_df['lines'])
print ("There are {} words in the combination of all review.".format(len(text)))

stopwords = set(STOPWORDS)
stopwords.update(["patient", "now", "started", "transferred", "noted","presented","family","reports","approximately","noted","home","Hospital"])
# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
# plot the WordCloud image
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()


# convert dataframe into list
sentences = lines_df['lines'].tolist()
tagged_texts = pos_tag_sents(map(word_tokenize, sentences))

i=0
lines = []
sent_no = []
pos_tags = []
for val in tagged_texts:
    i+=1
    for word, tag in val:
        lines.append(word)
        pos_tags.append(tag)
        sent_no.append('sentence'+str(i))

# create a dataframe
data = { 'sent_no':sent_no,'words': lines, 'pos_tag':pos_tags}
ner_df = pd.DataFrame(data)

drug_names = pd.read_csv("C:/Users/kiruthika.parthiban/Desktop/EHR/Drug_names.csv",encoding = "ISO-8859-1")
regimen = pd.read_csv("C:/Users/kiruthika.parthiban/Desktop/EHR/Regimen.csv",encoding = "ISO-8859-1")
symptom_name = pd.read_csv("C:/Users/kiruthika.parthiban/Desktop/EHR/symptom_name.csv",encoding = "ISO-8859-1")

# For one word labelling , use inner join
result  = pd.merge(ner_df, symptom_name, left_on = "words", right_on = "Symptom_Name" , how = 'left')
result1  = pd.merge(ner_df, regimen, left_on = "words", right_on = "Regimens" , how = 'inner')

# for two word labelling, use looping technique based on the bi-gram model
f = [nltk.word_tokenize(val) for val in lines_df['lines']]
f = list(filter(None, f))

a = f[:250000]

symptom_name = pd.read_csv('C:/Users/kiruthika.parthiban/Desktop/EHR/two_word_symptom.csv', encoding = "ISO-8859-1")
lists = list(symptom_name['Symptom Name'])

s = a

s = list(filter(None, s))

for (sentence,t) in zip(a,s):
        i = 0
        while i < len(sentence):
            print(sentence)
            #print(i)
            for m in lists:
                #print(m)

                words = m.split(' ')
                if sentence[i] == words[0]:
                    #print(sentence[i], words[0])
                    print(words)
                    for j in range(1, len(words)):
                        print(i)
                        print(len(sentence))
                        if len(sentence) < i+1:
                            break
                        if len(sentence) > i+1:
                            if sentence[i + 1] == words[j]:
                               t[i] = 'B-S'
                               t[i+1] = 'E-S'
                        #sentence.pop(i + 1)

            i += 1
