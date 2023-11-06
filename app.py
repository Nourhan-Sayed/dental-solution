import streamlit as st 
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier  # Import KNN
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns  # Import Seaborn




uploaded_file = st.file_uploader(label="Upload a file", type=['csv', 'xlsx'])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        df = pd.read_excel(uploaded_file)

st.sidebar.title("Select the Dental Disease")

menu = ["Periodontal diseases", "Caries diseases"]
choice = st.sidebar.selectbox("Select a tab", menu)

# Periodontal tab content
if choice == "Periodontal diseases":
    st.title("Periodontal Disease")
    selected_option = st.sidebar.selectbox("choose available correlation problems", 
                                       ["gingivitis and pd", "periodontitis and cal"])
    
    if selected_option == "gingivitis and pd":
        df['gingivitis'] = df['probing depth'].apply(lambda x: 0 if x > 4 else 1)

        depth_counts = df['probing depth'].value_counts().sort_index()

        # Create a bar plot to visualize the number of people with different probing depths
        fig, ax = plt.subplots()
        ax.bar(depth_counts.index, depth_counts.values)
        ax.set_xlabel('Probing Depth')
        ax.set_ylabel('Number of People')
        ax.set_title('Number of People with Different Probing Depths')

        # Display the chart in Streamlit
        st.pyplot(fig)

        # Create a stacked bar plot to visualize the number of people with different gingivitis levels for each probing depth
        fig, ax = plt.subplots(figsize=(8, 6))
        df.groupby(['probing depth', 'gingivitis']).size().unstack().plot(kind='bar', stacked=True, ax=ax)
        ax.set_xlabel('Probing Depth')
        ax.set_ylabel('Number of People')
        ax.set_title('Number of People with Different Gingivitis Levels for Each Probing Depth')
        ax.legend(title='Gingivitis')

        # Rotate the x-axis labels
        plt.xticks(rotation=45)

        # Display the chart
        plt.show()


        # Display the chart in Streamlit
        st.pyplot(fig)

    elif selected_option == "periodontitis and cal":
        df['peridontitis'] = df['Clinical Attachment Loss'].apply(lambda x: 0 if x > 0 else 1)
        periodontitis_counts = df['peridontitis'].value_counts()

        fig, ax = plt.subplots()
        ax.bar(periodontitis_counts.index, periodontitis_counts.values)
        ax.set_xlabel('Peridontitis')
        ax.set_ylabel('Number of People')
        ax.set_title('Number of People with and without Periodontitis')
        st.pyplot(fig)

        cal_counts = df.groupby('peridontitis')['Clinical Attachment Loss'].value_counts().unstack(0)

        fig, ax = plt.subplots()
        cal_counts.plot(kind='bar', stacked=True, ax=ax)
        ax.set_xlabel('Clinical Attachment Loss')
        ax.set_ylabel('Number of People')
        ax.set_title('Number of People with Different Clinical Attachment Loss for Each Periodontitis Group')
        ax.legend(title='Peridontitis')
        st.pyplot(fig)

    df
    df.to_excel('simulated_data_edited.xlsx', index=False)

# Caries tab content
elif choice == "Caries diseases":  # Corrected the typo in the tab name
    st.title("Caries Disease")
    selected_option_caries = st.sidebar.selectbox("choose from available correlation problems", ["caries and hollow tooth after 6 mon", "caries and endo"])
    selected_option2 = st.sidebar.selectbox("choose from available machine learning models", ["SVM", "Decision tree", "Random Forest", "KNN", "Naive Bayes"])
    # Checkbox to display the confusion matrix
    display_confusion_matrix = st.sidebar.checkbox("Display Confusion Matrix")
    additional_metrics = st.sidebar.checkbox("Show Additional metrics")




    if selected_option_caries == "caries and hollow tooth after 6 mon":
        if selected_option2 == "SVM":
            X = df[['current caries stage']]
            y = df['hollow tooth after 6 months']

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # Create an SVM model and fit it to the training data
            model = SVC(kernel='linear', C=1)  # You can choose different kernels and C values
            model.fit(X_train, y_train)
            # Make predictions using the trained model on the test data
            y_pred = model.predict(X_test)
            # Calculate and print the accuracy
            accuracy_svm = accuracy_score(y_test, y_pred)
            print(f'Accuracy: {accuracy_svm:.2f}')
            confusion1 = confusion_matrix(y_test, y_pred)
            st.markdown(f"<h3>SVM Accuracy: {accuracy_svm:.2f}</h3>", unsafe_allow_html=True)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)

            if display_confusion_matrix:
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion1, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                st.pyplot(plt)
                
            
            
        elif selected_option2 == "Decision tree":
            X2 = df[['current caries stage']]
            y2 = df['hollow tooth after 6 months']
            # Split the data into training and testing sets
            X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
            # Create an SVM model and fit it to the training data
            model = DecisionTreeClassifier()
            model.fit(X_train2, y_train2)
            # Make predictions using the trained model on the test data
            y_pred2 = model.predict(X_test2)
            # Calculate and print the accuracy
            accuracy_dec = accuracy_score(y_test2, y_pred2)
            print(f'Accuracy: {accuracy_dec:.2f}')
            confusion2 = confusion_matrix(y_test2, y_pred2)
            st.markdown(f"<h3>Decision Tree Accuracy: {accuracy_dec:.2f}</h3>", unsafe_allow_html=True)
            classification_rep = classification_report(y_test2, y_pred2, output_dict=True)

            if display_confusion_matrix:
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion2, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                st.pyplot(plt)

        elif selected_option2 == "Random Forest":
            X3 = df[['current caries stage']]
            y3 = df['hollow tooth after 6 months']
            # Split the data into training and testing sets
            X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=42)
            # Create a Random Forest model and fit it to the training data
            model_rf = RandomForestClassifier()
            model_rf.fit(X_train3, y_train3)
            # Make predictions using the trained model on the test data
            y_pred_rf = model_rf.predict(X_test3)
            # Calculate and print the accuracy
            accuracy_rf = accuracy_score(y_test3, y_pred_rf)
            print(f'Random Forest Accuracy: {accuracy_rf:.2f}')
            confusion3 = confusion_matrix(y_test3, y_pred_rf)
            st.markdown(f"<h3>Random Forest Accuracy: {accuracy_rf:.2f}</h3>", unsafe_allow_html=True)
            classification_rep = classification_report(y_test3, y_pred_rf, output_dict=True)
            if display_confusion_matrix:
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion3, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                st.pyplot(plt)
        elif selected_option2 == "KNN":
            X4 = df[['current caries stage']]
            y4 = df['hollow tooth after 6 months']
            # Split the data into training and testing sets
            X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, test_size=0.2, random_state=42)
            # Create a KNN model and fit it to the training data
            model_knn = KNeighborsClassifier()
            model_knn.fit(X_train4, y_train4)
            # Make predictions using the trained model on the test data
            y_pred_knn = model_knn.predict(X_test4)
            # Calculate and print the accuracy
            accuracy_knn = accuracy_score(y_test4, y_pred_knn)
            classification_rep = classification_report(y_test4, y_pred_knn, output_dict=True)

            print(f'KNN Accuracy: {accuracy_knn:.2f}')
            confusion4 = confusion_matrix(y_test4, y_pred_knn)
            st.markdown(f"<h3>KNN Accuracy: {accuracy_knn:.2f}</h3>", unsafe_allow_html=True)

            if display_confusion_matrix:
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion4, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                st.pyplot(plt)
        if selected_option2 == "Naive Bayes":
            X5 = df[['current caries stage']]
            y5 = df['hollow tooth after 6 months']
            # Split the data into training and testing sets
            X_train5, X_test5, y_train5, y_test5 = train_test_split(X5, y5, test_size=0.2, random_state=42)
            # Create a Naive Bayes model (e.g., Gaussian Naive Bayes) and fit it to the training data
            model_nb = GaussianNB()
            model_nb.fit(X_train5, y_train5)
            # Make predictions using the trained model on the test data
            y_pred_nb = model_nb.predict(X_test5)
            # Calculate and print the accuracy
            accuracy_nb = accuracy_score(y_test5, y_pred_nb)
            classification_rep = classification_report(y_test5, y_pred_nb, output_dict=True)

            print(f'Naive Bayes Accuracy: {accuracy_nb:.2f}')
            confusion5 = confusion_matrix(y_test5, y_pred_nb)
            st.markdown(f"<h3>Naive Bayes Accuracy: {accuracy_nb:.2f}</h3>", unsafe_allow_html=True)
            if display_confusion_matrix:
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion5, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                st.pyplot(plt)

    def display_confusion_matrix(y_true, y_pred):
        confusion = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(plt)
        # Additional metrics
    if additional_metrics:
        st.markdown("<h2>Classification Report:</h2>", unsafe_allow_html=True)
        # Convert the classification report dictionary to a DataFrame
        classification_df = pd.DataFrame(classification_rep)
        # Display the DataFrame as a table
        st.table(classification_df)

##-------------------------------------------------------


    elif selected_option_caries == "caries and endo":

        if selected_option2 == "SVM":
                X = df[['current caries stage']]
                y = df['endo']

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                # Create an SVM model and fit it to the training data
                model = SVC(kernel='linear', C=1)  # You can choose different kernels and C values
                model.fit(X_train, y_train)
                # Make predictions using the trained model on the test data
                y_pred = model.predict(X_test)
                # Calculate and print the accuracy
                accuracy_svm = accuracy_score(y_test, y_pred)
                print(f'Accuracy: {accuracy_svm:.2f}')
                confusion1 = confusion_matrix(y_test, y_pred)
                st.markdown(f"<h3>SVM Accuracy: {accuracy_svm:.2f}</h3>", unsafe_allow_html=True)
                classification_rep = classification_report(y_test, y_pred, output_dict=True)

                if display_confusion_matrix:
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(confusion1, annot=True, fmt='d', cmap='Blues')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix')
                    st.pyplot(plt)
                    
                
                
        elif selected_option2 == "Decision tree":
            X2 = df[['current caries stage']]
            y2 = df['endo']
            # Split the data into training and testing sets
            X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
            # Create an SVM model and fit it to the training data
            model = DecisionTreeClassifier()
            model.fit(X_train2, y_train2)
            # Make predictions using the trained model on the test data
            y_pred2 = model.predict(X_test2)
            # Calculate and print the accuracy
            accuracy_dec = accuracy_score(y_test2, y_pred2)
            print(f'Accuracy: {accuracy_dec:.2f}')
            confusion2 = confusion_matrix(y_test2, y_pred2)
            st.markdown(f"<h3>Decision Tree Accuracy: {accuracy_dec:.2f}</h3>", unsafe_allow_html=True)
            classification_rep = classification_report(y_test2, y_pred2, output_dict=True)

            if display_confusion_matrix:
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion2, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                st.pyplot(plt)

        elif selected_option2 == "Random Forest":
            X3 = df[['current caries stage']]
            y3 = df['endo']
            # Split the data into training and testing sets
            X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=42)
            # Create a Random Forest model and fit it to the training data
            model_rf = RandomForestClassifier()
            model_rf.fit(X_train3, y_train3)
            # Make predictions using the trained model on the test data
            y_pred_rf = model_rf.predict(X_test3)
            # Calculate and print the accuracy
            accuracy_rf = accuracy_score(y_test3, y_pred_rf)
            print(f'Random Forest Accuracy: {accuracy_rf:.2f}')
            confusion3 = confusion_matrix(y_test3, y_pred_rf)
            st.markdown(f"<h3>Random Forest Accuracy: {accuracy_rf:.2f}</h3>", unsafe_allow_html=True)
            classification_rep = classification_report(y_test3, y_pred_rf, output_dict=True)
            if display_confusion_matrix:
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion3, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                st.pyplot(plt)
        elif selected_option2 == "KNN":
            X4 = df[['current caries stage']]
            y4 = df['endo']
            # Split the data into training and testing sets
            X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, test_size=0.2, random_state=42)
            # Create a KNN model and fit it to the training data
            model_knn = KNeighborsClassifier()
            model_knn.fit(X_train4, y_train4)
            # Make predictions using the trained model on the test data
            y_pred_knn = model_knn.predict(X_test4)
            # Calculate and print the accuracy
            accuracy_knn = accuracy_score(y_test4, y_pred_knn)
            classification_rep = classification_report(y_test4, y_pred_knn, output_dict=True)

            print(f'KNN Accuracy: {accuracy_knn:.2f}')
            confusion4 = confusion_matrix(y_test4, y_pred_knn)
            st.markdown(f"<h3>KNN Accuracy: {accuracy_knn:.2f}</h3>", unsafe_allow_html=True)

            if display_confusion_matrix:
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion4, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                st.pyplot(plt)
        if selected_option2 == "Naive Bayes":
            X5 = df[['current caries stage']]
            y5 = df['endo']
            # Split the data into training and testing sets
            X_train5, X_test5, y_train5, y_test5 = train_test_split(X5, y5, test_size=0.2, random_state=42)
            # Create a Naive Bayes model (e.g., Gaussian Naive Bayes) and fit it to the training data
            model_nb = GaussianNB()
            model_nb.fit(X_train5, y_train5)
            # Make predictions using the trained model on the test data
            y_pred_nb = model_nb.predict(X_test5)
            # Calculate and print the accuracy
            accuracy_nb = accuracy_score(y_test5, y_pred_nb)
            classification_rep = classification_report(y_test5, y_pred_nb, output_dict=True)

            print(f'Naive Bayes Accuracy: {accuracy_nb:.2f}')
            confusion5 = confusion_matrix(y_test5, y_pred_nb)
            st.markdown(f"<h3>Naive Bayes Accuracy: {accuracy_nb:.2f}</h3>", unsafe_allow_html=True)
            if display_confusion_matrix:
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion5, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                st.pyplot(plt)

    def display_confusion_matrix(y_true, y_pred):
        confusion = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(plt)
        # Additional metrics
    if additional_metrics:
        st.markdown("<h2>Classification Report:</h2>", unsafe_allow_html=True)
        # Convert the classification report dictionary to a DataFrame
        classification_df = pd.DataFrame(classification_rep)
        # Display the DataFrame as a table
        st.table(classification_df)
        

