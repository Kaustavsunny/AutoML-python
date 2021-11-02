import streamlit as st 
import pandas as pd 

import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 

matplotlib.use('Agg')
import seaborn as sns

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics  import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# AUTO ML
""")

image=Image.open('C:/Users/admin/Desktop/python/ML/im1.jpg')
st.image(image,caption='ML',use_column_width=True)




def main():
    st.title("Auto ML App")
    st.text('using streamlit')
    activities=['EDA','Plot','Model Building']
    choice=st.sidebar.selectbox('Select Activity',activities)
    
    if choice=='EDA':
        st.subheader('Exploletory Data Analysis')
        data=st.file_uploader('Uploaded Dataset',type=['csv','txt'])
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head())
            if st.checkbox('Show Shape'):
                st.write(df.shape)
            if st.checkbox('Show Columns'):
                all_columns=df.columns.to_list()
                st.write(all_columns)
            if st.checkbox('Selected Columns To Show'):
                selected_columns=st.multiselect('Select Columns',all_columns)
                new_df=df[selected_columns]
                st.dataframe(new_df)
            if st.checkbox('Show Summary'):
                st.write(df.describe())
            if st.checkbox('Show Value Counts'):
                st.write(df.iloc[:,-1].value_counts())  
    elif choice=='Plot':
        st.subheader('Data Visualization')
        data=st.file_uploader('Uploaded Dataset',type=['csv','txt'])
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head())
        if st.checkbox('Correletion With Seaborn'):
            st.write(sns.heatmap(df.corr(),annot=True))
            st.pyplot()
        if st.checkbox('Pie Chart'):
            all_columns=df.columns.to_list()
            columns_to_plot=st.selectbox('Select 1 Column:',all_columns)
            pie_plot=df[columns_to_plot].value_counts().plot.pie(autopct='%1.1f%%')
            st.write(pie_plot)
            st.pyplot()
        all_columns_names=df.columns.tolist()
        type_of_plot=st.selectbox('Select Types Of Plot',['area','bar','line','hist','box','kde'])
        selected_columns_names=st.multiselect('Select Column To Plot',all_columns_names)
        if st.button('Generate Plot'):
            st.success('Generate Coustomize Plot Of {} for {}'.format(type_of_plot,all_columns_names)) 

            if type_of_plot=='area':
                cust_data=df[selected_columns_names]
                st.area_chart(cust_data)
            elif type_of_plot=='bar':
                cust_data=df[selected_columns_names]
                st.bar_chart(cust_data)
            elif type_of_plot=='line':
                cust_data=df[selected_columns_names]
                st.line_chart(cust_data) 
            elif type_of_plot:
                cust_plot=df[selected_columns_names].plot(kind=type_of_plot)
                st.write(cust_plot)
                st.pyplot()    



    


    elif choice=='Model Building':

        st.subheader('Building ML Model')
        data=st.file_uploader('Uploaded Dataset',type=['csv','txt'])
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head())
            X=df.iloc[:,0:-1].values
            y=df.iloc[:,-1].values
            #X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
            
            st.subheader("Pick Your Algorithm") 
            choose_model=st.selectbox(label=' ',options=[' ','KNN','Logistic Regression'])
            if(choose_model=='KNN'):
                from sklearn.model_selection import train_test_split
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
                from sklearn.neighbors import KNeighborsClassifier
                KNN=KNeighborsClassifier()
                KNN.fit(X_train,y_train)
                st.subheader('Model Test Accuracy Score:')
                st.write(str(accuracy_score(y_test,KNN.predict(X_test))*100)+ '%')
            if(choose_model=='Logistic Regression'):
                from sklearn.model_selection import train_test_split
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
                from sklearn.linear_model import LogisticRegression
                KNN=LogisticRegression()
                KNN.fit(X_train,y_train)
                st.subheader('Model Test Accuracy Score:')
                st.write(str(accuracy_score(y_test,KNN.predict(X_test))*100)+ '%')   
            
            #RandomForestClassifier= KNeighborsClassifier()
            #RandomForestClassifier.fit(X_train,y_train)

            #st.subheader('Model Test Accuracy Score:')
            #st.write(str(accuracy_score(y_test,RandomForestClassifier.predict(X_test))*100)+ '%')

if __name__=='__main__':
    main()