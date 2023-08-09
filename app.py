import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from pycaret.classification import load_model, predict_model, setup, get_config
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

import pickle
import pandas as pd
import numpy as np

import re
import string
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression




# Define function to clean text
def clean_text(text):
    # remove links starting with 'http' or 'www'
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    # replace '\r' and/or '\n' with spacing
    text = re.sub('[\n\r]+', '\n', text)
    text = text.replace('\n', ' ')
    # remove emojis
    return emoji.replace_emoji(text, replace='')

# streamlit shell (layouts etc)
# set webpage name and icon
st.set_page_config(
    page_title='Riding the Data Trail',
    page_icon='🚲',
    layout='wide',
    initial_sidebar_state='expanded'
    )


st.markdown("<h1 style='text-align: center; color: black;'>🚴🏻‍♂️ Riding the Data Trail 🚴🏻‍♀️</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black;'>Text-Based Price Classifier for Road Bikes on Carousell</h2>", unsafe_allow_html=True)

# top navigation bar
selected = option_menu(
    menu_title = None,
    options = ['Get to know your Road Bike!', 'Road Bike Pricing Predictor'],
    icons = ['gear-wide-connected', 'currency-exchange'],
    default_index = 0,
    orientation = 'horizontal',
    styles={
        'nav-link-selected': {'background-color': '##d93354'}
        }
    )


st.markdown('<div style="text-align: center;">++++++++++++++++++++</div>', unsafe_allow_html=True)

if selected == 'Get to know your Road Bike!':
    # title
    st.markdown("<h1 style='text-align: center; color: black;'>Parts of a Road Bike</h1>", unsafe_allow_html=True)

    # add an image
    st.image('bicycle-components.jpg', caption='Road Bike', use_column_width=True)
    
    # external links
    st.write('Resources for further reading')
    st.markdown("* [Understanding groupsets](https://www.bikeradar.com/advice/buyers-guides/road-bike-groupsets-everything-you-need-to-know/)")
    st.markdown("* [Choosing your wheelset](https://www.rei.com/learn/expert-advice/bike-wheels-how-to-choose.html)")
    st.markdown("* [Choosing a suitable bike frame material](https://www.bicycling.com/bikes-gear/a21784287/bike-frame-materials-explained/)")
    st.markdown("* [How to set up drop bars](https://bike.bikegremlin.com/920/setting-up-road-bike-bars/)")
    
    

elif selected == 'Road Bike Pricing Predictor':
    # title
    st.markdown("<h1 style='text-align: center; color: black;'>Road Bike Pricing Classifier</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h2 style='text-align: center; color: black;'>User Input</h2>", unsafe_allow_html=True)

        title = st.text_input("Title", placeholder="Input Title")
        item_condition = st.selectbox("Item Condition", ['Brand new', 'Like new', 'Lightly used', 'Well used', 'Heavily used'])
        condition_subtext = st.selectbox("Item Condition", ['Never used. May come with original packaging or tag.', 'Used once or twice. As good as new.', 'Used with care. Flaws, if any, are barely noticeable.', 'Has minor flaws or defects.', 'Has obvious signs of use or defects.'])
        st.write('For x1 to x5, input 5 keywords from your title')
        x1 = st.text_input("x1", placeholder="examples: bike, canyon, giant, merida, pinarello, specialized, trek")
        x2 = st.text_input("x2", placeholder="examples: bike, carbon, cf, dogma, madone, size, tcr")
        x3 = st.text_input("x3", placeholder="examples: bike, bontrager, carbon, mm, reacto, shimano, size")
        x4 = st.text_input("x4", placeholder="examples: advanced, bike, carbon, di2, oclv, shimano, ultimate")
        x5 = st.text_input("x5", placeholder="examples: bike, dura, endurace, frame, pro, shimano, ultegra")

        df = pd.read_csv('../data/final_for_pycaret.csv')
        df=df.drop(columns='price')
        categorical_cols = ['title', 'item_condition', 'deal_method', 'post_date', 'category_type', 'post_type', 'condition_subtext',
                        'mailing_option', 'meetup_option', 'meetup_location', 'seller_id', 'seller_join_date',
                        'seller_response', 'seller_verif', 'verified_by_email', 'verified_by_facebook', 'verified_by_mobile', 'brands', 
                        'lemma_posts', 'x1', 'x2', 'x3' ,'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10' ]

        # One-Hot Encoding with 'handle_unknown' set to 'ignore'
        encoder_ignore = OneHotEncoder(handle_unknown='ignore').fit(df[categorical_cols])
        encoded_df = encoder_ignore.transform(df[categorical_cols]).toarray()
    
        # Train the Nearest Neighbors model
        nn_model = NearestNeighbors(n_neighbors=1).fit(encoded_df)

        def find_nearest_neighbor(title, item_condition, condition_subtext, x1, x2, x3, x4, x5):
            # Create a complete DataFrame with placeholders
            complete_input = pd.DataFrame([df.mode().loc[0]], columns=df.columns)
            
            # Set null values for the remaining columns that are not user input fields
            for col in df.columns:
                if col not in ['title', 'item_condition', 'condition_subtext', 'x1', 'x2', 'x3', 'x4', 'x5']:
                    complete_input[col] = ""
                

            # Replace the user-provided columns with the user input
            complete_input['title'] = title
            complete_input['item_condition'] = item_condition
            complete_input['condition_subtext'] = condition_subtext
            complete_input['x1'] = x1
            complete_input['x2'] = x2
            complete_input['x3'] = x3
            complete_input['x4'] = x4
            complete_input['x5'] = x5

            # One-Hot Encode the categorical columns of the user input DataFrame
            complete_input_encoded = encoder_ignore.transform(complete_input[categorical_cols]).toarray()
        
            nn_model = NearestNeighbors(n_neighbors=1).fit(encoded_df)
            # Use the trained Nearest Neighbors model to find the nearest neighbor in the dataset
            distances, indices = nn_model.kneighbors(complete_input_encoded)

            # Complete the user input DataFrame with the data from the nearest neighbor
            for col in df.columns:
                if col not in ['title', 'item_condition', 'condition_subtext', 'x1', 'x2', 'x3', 'x4', 'x5']:
                    complete_input[col] = df.loc[indices[0], col].values
    
            return complete_input
            
    with col2:
        st.markdown("<h2 style='text-align: center; color: black;'>Results</h2>", unsafe_allow_html=True)


    if st.button("Submit"):
    
        # input_data = prepare_input_data(title, item_condition, condition_subtext, x1, x2, x3, x4, x5)
        # Test the function
        input_data_encoded=find_nearest_neighbor(title, item_condition, condition_subtext, x1, x2, x3, x4, x5)
        # st.write(input_data_encoded) #this outputs the row with user inputs and knn of other columns

        #model=load_model('best_model')
        model=load_model('../code/best_model')

        # Make predictions using the loaded model and the transformed input data
        predictions = model.predict(input_data_encoded)

        # Get raw probabilities using the trained model
        raw_probs = predict_model(model, data=input_data_encoded, raw_score=True)
        # Output the probabilities for each class
        #class_probabilities = raw_probs.iloc[0, 1:]
        #for class_name, probability in zip(class_probabilities.index, class_probabilities.values):
        #    st.write(f"Probability of Class '{class_name}': {probability:.4f}")
        with col2:

            st.write("The predicted price class is:", predictions[0])
            # st.write("Predicted Price:", predictions['Label'][0])
            # Create a DataFrame
            class_df = pd.DataFrame({
                "Class": [0, 1, 2],
                "Price Range": ["$0- $600", "$601-$1200", "$1201-$2000"]
            })

            # Display the DataFrame as a table in Streamlit
            st.dataframe(class_df, hide_index=True)
        
        
        #st.write(predictions)
        # st.write("Predicted Price:", predictions['Label'][0])
