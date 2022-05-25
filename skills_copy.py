import pandas as pd
from csv import writer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline


d = pd.read_csv(r'/home/vostroraf/Documents/HW_Saver_internship/Files/Example_Technical_Skills (copy).csv')
print(len(d.query('value == 1')))
# d_shuf = d.sample(frac=1)
d_shuf = d
d_content = d_shuf['Technology Skills']
d_label = d_shuf['value']
print(d_label)
vectorizer = CountVectorizer()

pipeline = Pipeline([
    ('bag-of-words', CountVectorizer()),
    ('random forest', RandomForestClassifier()),
    ])
make_pipeline(CountVectorizer(), RandomForestClassifier(n_estimators=1000))
pipeline.fit(d_content, d_label)
print(pipeline.score(d_content[3000:], d_label[3000:]))

f1 = pd.read_csv(r'/home/vostroraf/Documents/HW_Saver_internship/Files/Raw_Skills_Dataset (copy).csv')
f2 = open('/home/vostroraf/Documents/HW_Saver_internship/Technical-skills.csv', 'w')
f3 = open('/home/vostroraf/Documents/HW_Saver_internship/Soft-skills.csv', 'w')

writer2 = writer(f2)
writer3 = writer(f3)

for i in f1['RAW DATA']:
    # print(pipeline.predict([i])[0])
    if pipeline.predict([i]) == 1:
        print(f"{i.split('$$')} : {pipeline.predict([i])}")
        writer2.writerow(i.split('Made by Musharaf'))
    else:
        print(f"{i.split('$$')} : {pipeline.predict([i])}")
        writer3.writerow(i.split('Made by Musharaf'))

f2.close()
f3.close()