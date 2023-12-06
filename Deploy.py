#pip install streamlit
#streamlit run CapstoneStreamlit.py
#pip install streamlit-card
#pip install emoji
#!pip install easyocr
#!pip install openai
#!pip install python-dotenv
#!openai migrate
#!pip install openai==0.28
#pip install spacy
#pip install hydralit_components
#pip install streamlit_option_menu

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import emoji
import pickle
import string
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import re
from streamlit_option_menu import option_menu

# -------------------- Styling -------------------- #
# To display logos
import base64

# Display the center logo
center_logo_path = 'logo_title.PNG'
center_logo = open(center_logo_path, 'rb')

center_logo_data = center_logo.read()
center_logo_base64 = base64.b64encode(center_logo_data).decode("utf-8")

# Display the right logo
right_logo_path = 'authority_logo.png'
right_logo = open(right_logo_path, 'rb')

right_logo_path = right_logo.read()
right_logo_base64 = base64.b64encode(right_logo_path).decode("utf-8")

st.markdown(
    f"""
    <div style="margin-top: -10px; margin-right: 600px; margin-left: 700px; height: 10px; width: 160px">
        <img src="data:image/png;base64,{right_logo_base64}" alt="Right Logo" width= 170px height= 100px>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div style="margin-top: 60px; margin-right: 50px; margin-left: 210px; height: 300px; width: 300px">
        <img src="data:image/png;base64,{center_logo_base64}" alt="Center Logo" width= 300px height= 300px>
    </div>
    """,
    unsafe_allow_html=True
)




def STEP1(FullData):
# ----- First remove the emojis: ----- #

    def remove_emojis(text):
        # Remove emojis using the emoji library
        text = emoji.demojize(text)

        # Remove any remaining emoji characters using regular expressions
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        cleaned_text = emoji_pattern.sub(r'', text)
        cleaned_text = re.sub(r':[a-zA-Z_-]+:', '', text)


        return cleaned_text
        
    # remove the emojies from text in the data frame
    for index, row in FullData.iterrows():
        cleaned_text = remove_emojis(row['text'])
        FullData.at[index, 'text'] = cleaned_text
        
# ----- Second remove the stopwords: ----- #


    # Iterate through the rows of the 'text' column
    for index, row in FullData.iterrows():
        stopwords = pd.read_pickle('C:/Users/Razan/Desktop/T5/stopwords.pkl')
        # Split the row into a list of words
        words = row['text'].split()
        
        # Remove words from the row if they are in the stopwords list
        words = [word for word in words if word not in stopwords]
        
        # Join the remaining words back into a string
        modified_text = ' '.join(words)
        
        # Update the 'text' column with the modified row
        FullData.at[index, 'text'] = modified_text
        
# ----- Third clean the data: ----- #

    #Remove repeated letters
    def remove_repeated_letters(word):
        pattern = re.compile(r'(.)\1+')
        return pattern.sub(r'\1', word)

        FullData['text'] = FullData['text'].apply(remove_repeated_letters)
    
    #Unifying the shape of letters
    FullData['text'] = FullData['text'].replace("آ", "ا")
    FullData['text'] = FullData['text'].replace("إ", "ا")
    FullData['text'] = FullData['text'].replace("أ", "ا")
    FullData['text'] = FullData['text'].replace("ؤ", "و")
    FullData['text'] = FullData['text'].replace("ئ", "ي")
    
    #Detangling the word "اعلان" from the letters
    FullData['text'] = FullData['text'].str.replace(r'(\S)اعلان(\S)', r'\1 اعلان \2')
    
    #Remove unrelated # and @mentions
    def remove_after_sym(text):
        # Replace hashtags and mentions with whitespace
        text = re.sub(r"#\S+|@\S+", " ", text)

        # Remove words after '#' and '@' except if the word is 'إعلان' or 'اعلان'
        for sym in ['#', '@']:
            text = re.sub(rf"{sym}\S+(?=(?:\W\S+)*[إع]علان)", "", text)

        return text

    # Apply the function to the 'text' column
    FullData['text'] = FullData['text'].apply(remove_after_sym)
    
    #Replacing the 'https://colab.research.google.com/' with <رابط >
    FullData['text'] = FullData['text'].apply(lambda x: re.sub(r'http\S+', "<رابط>", x))

    #Remove punctuation, hashtags, and diacritics
    
    # Remove noise which includes punctuation, hashtags, and diacritics
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.٪,'{}~¦+|!”…“–ـ'''
    english_punctuations = string.punctuation
    punctuations_list = arabic_punctuations + english_punctuations

    arabic_diacritics = re.compile("""
                                 ّ    | # Tashdid
                                 َ    | # Fatha
                                 ً    | # Tanwin Fath
                                 ُ    | # Damma
                                 ٌ    | # Tanwin Damm
                                 ِ    | # Kasra
                                 ٍ    | # Tanwin Kasr
                                 ْ    | # Sukun
                                 ـ     # Tatwil/Kashida

                             """, re.VERBOSE)
    def remove_diacritics(text):
        text = re.sub(arabic_diacritics, '', text) #it will replace the diacritics with an empty space
        return text

    def remove_punctuations(text):
        translator = str.maketrans('', '', punctuations_list)
        return text.translate(translator)   
    
        FullData['text'] = FullData['text'].apply(remove_diacritics)
        FullData['text'] = FullData['text'].apply(remove_punctuations)
    
    #Remove English letters, numbers, and Arabic numbers.
    def remove_english_letters(text):
        return re.sub(r'[a-zA-Z0-9٠-٩]', '', text)

        FullData['text'] = FullData['text'].apply(remove_english_letters)
    
    #Remove duplicated spaces.
    def remove_duplicate_spaces(text):
        return re.sub(r'\s+', ' ', text)

        FullData['text'] = FullData['text'].apply(remove_duplicate_spaces)

# ----- Fourth vectorize the data: ----- # 

    # Vectorize the text data
    vocab_size = 100000  # to be suitable with the AraBertv0.2 model
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(FullData['text'])  

    max_length = 100

    def vectorize_text(text):
        sequences = tokenizer.texts_to_sequences([text])  # Convert text to sequences
        sequences_padded = pad_sequences(sequences, maxlen=max_length)  # Pad sequences
        sequences_padded = np.expand_dims(sequences_padded, axis=-1)  # Add axis for compatibility
        sequences_padded = sequences_padded.astype('float32')  # Convert to float32
        return sequences_padded  # Return the vectorized sequence

    FullData['vectorized_text'] = FullData['text'].apply(vectorize_text) 
    
    return FullData

def STEP2(FullData):
    class_names = {0: "Advertisement", 1: "Not_Advertisement"}
    model = tf.keras.models.load_model('C:/Users/Razan/Desktop/T5/FromScratchModel/FromScratchModel.h5')
    
    def predict_class(text):

        predicted_class = model.predict(text)


        predicted_class = predicted_class[0][0]
        nearest_key = min(class_names.keys(), key=lambda x: abs(x - predicted_class))
        predicted_class_name = class_names[nearest_key]
        print(predicted_class_name)
        return predicted_class_name

    # Apply the prediction function to each row and save the predicted class
    FullData['predicted_class'] = FullData['vectorized_text'].apply(predict_class)

    # Filter the rows where the predicted class is 'Advertisement'
    FullData = FullData[FullData['predicted_class'] == 'Advertisement']

    # Drop the 'predicted_class' column
    FullData = FullData.drop('predicted_class', axis=1)

    return FullData

def STEP3(FullData):
    # Iterate through each row in the first dataset
    for index, row in FullData.iterrows():
        LicensedUsers = pd.read_excel('C:/Users/Razan/Desktop/T5/LicensedUsers.xlsx')
    
        userid = row['user_id']

        # Check if the userid exists in the second dataset
        if userid in LicensedUsers['user_id'].values:
            FullData.loc[index, 'license_status'] = 'مُرخص'
        else:
            FullData.loc[index, 'license_status'] = 'غير مُرخص'

    return FullData

def STEP4(FullData):
    import easyocr
    def process_images(FullData):
        
        reader = easyocr.Reader(['ar'])  # Specifying Arabic for OCR

        for index, row in FullData.iterrows():
            image_urls = [row['image_1_url'], row['image_2_url'], row['image_3_url'], row['image_4_url']] # Twitter alows uploading 4 images in each tweet.
            condition_satisfied = False

            # ---------- Check the TEXT conditions ---------- #

            # If the advertiser followed advertising compliance text and licensed, drop the row because they are not our target:
            if ('اعلان' in row['text'] or 'إعلان' in row['text']) and 'YES' in LicensedUsers[LicensedUsers['user_id'] == row['user_id']]['Licensed'].values:  

                condition_satisfied = True # Advertising compliance text and License✅ then drop it because it is not our target

            # If the advertiser followed advertising compliance text but NOT licensed, then against the law:
            elif ('اعلان' in row['text'] or 'إعلان' in row['text']) and 'NO' in LicensedUsers[LicensedUsers['user_id'] == row['user_id']]['Licensed'].values:  
                  condition_satisfied = False
             
             

             # ---------- Check the IMAGE conditions ---------- #

             # ---------- When the advertiser didn't write "اعلان" in the text we will check the images ---------- #

            # Check the image condition if the text condition is not satisfied
            if not condition_satisfied: # the if statement will be executed if condition_satisfied is False
                for image_url in image_urls:
                    if pd.notnull(image_url):  # Check if the image URL is not null
                        try:
                            response = requests.get(image_url)
                            img = cv2.imdecode(np.array(bytearray(response.content), dtype=np.uint8), -1)

                            result = reader.readtext(img)
                            predicted_easy_ocr = " ".join([x[1] for x in result])  # Extracted text

                            # If the advertiser followed advertising compliance text and licensed, drop the row because they are not our target:
                            if re.search(r'(?:#)?[إأا]علان', predicted_easy_ocr, re.IGNORECASE) and 'YES' in LicensedUsers[LicensedUsers['user_id'] == row['user_id']]['Licensed'].values:  
                              
                              condition_satisfied = True # Advertising compliance text and License✅ then drop it because it is not out target

                              break  # Exit the loop if the condition is satisfied

                            # If the advertiser followed advertising compliance text but NOT licensed, then against the law:  
                            elif re.search(r'(?:#)?[إأا]علان', predicted_easy_ocr, re.IGNORECASE) and 'NO' in LicensedUsers[LicensedUsers['user_id'] == row['user_id']]['Licensed'].values:  #To check the second condition 2️⃣❎
                        
                              condition_satisfied = False
                              break  # Exit the loop if the condition is satisfied
                              
                        except Exception as e:
                            print(f"Error processing image: {image_url}")
                            print(e)

            if condition_satisfied:
                FullData = FullData.drop(index) # Advertising compliance text and License✅ then drop it because it is not out target
            else:
                FullData.loc[index, 'AgainstOrNot'] = 'YES' 

        return FullData

    # Process the images and get the results
    results = process_images(FullData)

    return FullData

def STEP5(FullData):

    openai.api_key = "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"
    
    delimiter = "####"
    system_message = f"""
    You will take arabic tweet contain ad and you will give me the name of the company, if any. \
    {delimiter}
    give me the name of company in this arabic tweet  \
    for example: arabic tweet = عدسات نايس افضل عدسات الوانهم جميله ومريحه في العين
    company name = نايس
    example 2: arabic tweet =  العطور هذي فخمه فخمه ثقو فيني ما مدحتها الا وانا مجربتها للطلب
    company name = not defined
    """

    def get_completion_from_messages(messages,
                                 model="gpt-3.5-turbo",
                                 temperature=0,
                                 max_tokens=500):
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message["content"]

    detectedNames = []

    for index, row in FullData.iterrows():
        text = row['text']  # Access the 'text' column value

        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': f"{delimiter}{text}{delimiter}"}
        ]
        response = get_completion_from_messages(messages)

        # Store the output in the "detectedNames" column
        if response:
            detectedNames.append(response)
        else:
            detectedNames.append("Not defined")

    # Add the new column to the dataset
    FullData['org_names'] = detectedNames

    return FullData
   
                
def start_the_process(FullData):
    FullData = STEP1(FullData) #Preprocess
    FullData = STEP2(FullData) #Model
    FullData = STEP3(FullData) #Check the license
    FullData = STEP4(FullData) #Check the word "اعلان"
    FullData = STEP5(FullData) #Check the org name
    return FullData  
    
    
def main():

    # Load the X application and Snapchat datasets
    df_x = pd.read_excel('C:/Users/Razan/Desktop/T5/TwitterData.xlsx')
    df_snapchat = pd.read_csv('C:/Users/Razan/Desktop/T5/SnapToS3.csv')


    # horizontal menu
    selected = option_menu(None, ["","X", "Snapchat", ""], 
    icons=["bracket","twitter", "snapchat","bracket"], 
    menu_icon="cast", default_index=0, orientation="horizontal")
    
    
    # Display the selected dataset
    if selected == 'X':
        df = start_the_process(df_x.copy())
        
        # Make the link column clickable
        if 'tweet_url' in df.columns:
            df['tweet_url'] = df['tweet_url'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')
        
        # Rename a column
        df = df.rename(columns={'user_id': 'إسم المستخدم', 'original_text': 'نص التغريده', 'tweet_url': 'رابط التغريده', 'license_status': 'حالة الرخصه', 'org_names':'اسم الشركة'})

        
        subset_data = df[['اسم الشركة','حالة الرخصه', 'رابط التغريده', 'نص التغريده', 'إسم المستخدم']]
        #after_data_choice(subset_data)
            # Create a selectbox for the filter option within the option menu
        filter_option = option_menu(
            "",
            ["المرخصين","الغير مرخصين"],
            icons=["check","x"],
            default_index=0
        )
        

        
        # Filter the DataFrame based on the license status

        if filter_option == "المرخصين":
            subset_data = subset_data[subset_data['حالة الرخصه'] != 'غير مُرخص']
            subset_data = subset_data.drop(columns=['اسم الشركة', 'حالة الرخصه'])
            st.write(subset_data.to_html(escape=False, index=False), unsafe_allow_html=True)
            
        elif filter_option == "الغير مرخصين":
            subset_data = subset_data[subset_data['حالة الرخصه'] != 'مُرخص']
            subset_data = subset_data.drop(columns=['حالة الرخصه'])
            
            # Apply CSS styling to move the table to the right
            st.markdown(
                """
                <style>
                .dataframe {
                    margin-left: auto;
                    margin-right: auto;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
        
            # Display the filtered dataset
            st.write(subset_data.to_html(escape=False, index=False), unsafe_allow_html=True)
    

        
       
    elif selected == 'Snapchat':
        df = start_the_process(df_snapchat.copy())
        
        # Make the profile link column clickable
        if 'profile_url' in df.columns:
            df['profile_url'] = df['profile_url'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')
        
        # Rename a column
        df = df.rename(columns={'user_id': 'إسم المستخدم', 'profile_url': 'رابط الحساب', 'license_status': 'حالة الرخصه'})

        subset_data = df[['حالة الرخصه', 'رابط الحساب', 'إسم المستخدم']]
        
        
        
        

        # Apply CSS styling to move the table to the right
        st.markdown(
            """
            <style>
            .dataframe {
                margin-left: auto;
                margin-right: auto;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.write(subset_data.to_html(escape=False, index=False), unsafe_allow_html=True)
    



if __name__ == '__main__' :
    main()


