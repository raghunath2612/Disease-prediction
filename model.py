# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Importing the dataset
dataset = pd.read_csv('Training.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
Decisionclassifier = DecisionTreeClassifier(random_state=20)
Decisionclassifier.fit(X, y)

#Loading Decision tree classifier to decision.pkl
pickle.dump(Decisionclassifier, open('decision.pkl','wb'))
decision_pikle = pickle.load(open('decision.pkl','rb'))

# Fitting Random forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
RandomForestclassifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RandomForestclassifier.fit(X,np.ravel(y))

#Loading Random Forest classifier to decision.pkl
pickle.dump(RandomForestclassifier, open('random.pkl','wb'))
random_pikle = pickle.load(open('random.pkl','rb'))

# Fitting Naive Bayes Classification to the Training set
from sklearn.naive_bayes import GaussianNB
NaiveBayesclassifier = GaussianNB()
NaiveBayesclassifier.fit(X, y)

#Loading Naive Bayes classifier to naive.pkl
pickle.dump(NaiveBayesclassifier, open('naive.pkl','wb'))
naive_pikle = pickle.load(open('naive.pkl','rb'))


diseases='none,itching,skin_rash,nodal_skin_eruptions,continuous_sneezing,shivering,chills,joint_pain,stomach_pain,acidity,ulcers_on_tongue,muscle_wasting,vomiting,burning_micturition,spotting_ urination,fatigue,weight_gain,anxiety,cold_hands_and_feets,mood_swings,weight_loss,restlessness,lethargy,patches_in_throat,irregular_sugar_level,cough,high_fever,sunken_eyes,breathlessness,sweating,dehydration,indigestion,headache,yellowish_skin,dark_urine,nausea,loss_of_appetite,pain_behind_the_eyes,back_pain,constipation,abdominal_pain,diarrhoea,mild_fever,yellow_urine,yellowing_of_eyes,acute_liver_failure,fluid_overload,swelling_of_stomach,swelled_lymph_nodes,malaise,blurred_and_distorted_vision,phlegm,throat_irritation,redness_of_eyes,sinus_pressure,runny_nose,congestion,chest_pain,weakness_in_limbs,fast_heart_rate,pain_during_bowel_movements,pain_in_anal_region,bloody_stool,irritation_in_anus,neck_pain,dizziness,cramps,bruising,obesity,swollen_legs,swollen_blood_vessels,puffy_face_and_eyes,enlarged_thyroid,brittle_nails,swollen_extremeties,excessive_hunger,extra_marital_contacts,drying_and_tingling_lips,slurred_speech,knee_pain,hip_joint_pain,muscle_weakness,stiff_neck,swelling_joints,movement_stiffness,spinning_movements,loss_of_balance,unsteadiness,weakness_of_one_body_side,loss_of_smell,bladder_discomfort,foul_smell_of urine,continuous_feel_of_urine,passage_of_gases,internal_itching,toxic_look_(typhos),depression,irritability,muscle_pain,altered_sensorium,red_spots_over_body,belly_pain,abnormal_menstruation,dischromic _patches,watering_from_eyes,increased_appetite,polyuria,family_history,mucoid_sputum,rusty_sputum,lack_of_concentration,visual_disturbances,receiving_blood_transfusion,receiving_unsterile_injections,coma,stomach_bleeding,distention_of_abdomen,history_of_alcohol_consumption,fluid_overload,blood_in_sputum,prominent_veins_on_calf,palpitations,painful_walking,pus_filled_pimples,blackheads,scurring,skin_peeling,silver_like_dusting,small_dents_in_nails,inflammatory_nails,blister,red_sore_around_nose,yellow_crust_ooze'

diseases=diseases.split(',')

arr=np.array([[0]*132])

symptoms=['movement_stiffness', 'diarrhoea', 'watering_from_eyes', 'receiving_unsterile_injections', 'red_sore_around_nose']



for i in symptoms:
        arr[0][diseases.index(i)]=1
    
DCp = Decisionclassifier.predict(arr)
DCpd=decision_pikle.predict(arr)


RFp = RandomForestclassifier.predict(arr)
RFpd=random_pikle.predict(arr)


NBp = NaiveBayesclassifier.predict(arr)
NBpd=naive_pikle.predict(arr)


