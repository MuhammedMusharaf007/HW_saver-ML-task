# Importing the necessary Libraries

# Pandas to read CSV files
import pandas as pd

# CSV to write csv files
from csv import writer

# Sci-kit learn to use CountVectorizer and Random Forest Classifier in the Pipeline we create
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline

# Importing and reading the Dataset file
d = pd.read_csv(r'/home/vostroraf/Documents/HW_Saver_internship/Files/Example_Technical_Skills (copy).csv')
print(len(d.query('value == 1')))
d_shuf = d

# Taking the column which has the important data required
d_content = d_shuf['Technology Skills']
d_label = d_shuf['value']
print(d_label)

# Creating the Vectorizer instance
vectorizer = CountVectorizer()

# Creating the pipeline instance
pipeline = Pipeline([
    ('bag-of-words', CountVectorizer()),
    ('random forest', RandomForestClassifier()),
    ])

# Creating the pipeline with countVectorizer and RandomForestClassifier
make_pipeline(CountVectorizer(), RandomForestClassifier(n_estimators=1000)) # n_estimators can be adjusted on a trial and error method
pipeline.fit(d_content, d_label)

# Checking whether the model is trained by an obvious scorecheck (will return 100 percent accuracy since we used all training data to creare ML model)
print(pipeline.score(d_content[3000:], d_label[3000:]))

# Importing the testing Dataset to split into technical and soft skills.
f1 = pd.read_csv(r'/home/vostroraf/Documents/HW_Saver_internship/Files/Raw_Skills_Dataset (copy).csv')

# Creating files to write the data samples after recognizing whether they are Technical skills or not.
f2 = open('/home/vostroraf/Documents/HW_Saver_internship/Technical-skills.csv', 'w')
f3 = open('/home/vostroraf/Documents/HW_Saver_internship/Soft-skills.csv', 'w')

writer2 = writer(f2)
writer3 = writer(f3)

# Iterating through the whole prediction set.
for i in f1['RAW DATA']:

    # Checking whether the data is technical skills >> Adding to technical skills file.
    if pipeline.predict([i]) == 1:
        print(f"{i.split('$$')} : {pipeline.predict([i])}")
        writer2.writerow(i.split('Made by Musharaf'))
    
    # Adding to soft skills file if the sample is not a technical skill.
    else:
        print(f"{i.split('$$')} : {pipeline.predict([i])}")
        writer3.writerow(i.split('Made by Musharaf'))

# Closing the opened files after entering.
f2.close()
f3.close()


#____________________________________#
### Created By @MuhammedMusharaf007 ##
#------------------------------------#