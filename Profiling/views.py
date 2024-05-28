from collections import defaultdict
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import math
import json
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import warnings
warnings.simplefilter('ignore')
import plotly.express as px
from django.shortcuts import render
import cv2
import numpy as np
from keras.models import load_model
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import tensorflow as tf
from keras.preprocessing.image import load_img
from PIL import Image
import pandas as pd  # Import pandas to use DataFrame functionalities
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, authenticate, logout
from datetime import datetime
import tempfile
from .models import Product
from django.http import JsonResponse,HttpResponse
from .forms import ProductForm,QuantityForm,CameraForm,UploadFileForm
from django.contrib import messages
from django.shortcuts import get_object_or_404
from django.http import HttpResponseBadRequest
import numpy as np
from scipy.sparse.linalg import svds





#-----------------------------------------------------------------------------------------------------------------------------------------


total=0

columns=['index','ProductID','ProductName','ProductBrand','Gender','Price(INR)','PrimaryColor']
df = pd.read_csv("data(1).xls")
sedf= pd.read_csv("seasondata.csv")

df1=df.iloc[:400000,0:]

df1 = df.dropna()

# Checking the number of samples in your dataset
num_samples = len(df)

# Calculating the number of samples for the training set
train_size = int(0.7 * num_samples)  # 70% of the data for training

# Splitting the dataset into train and test sets
train_data, test_data = train_test_split(df, test_size=0.3, train_size=train_size, random_state=42)

train_data_grouped = train_data.groupby(['ProductName', 'Gender']).agg({'age': 'count'}).reset_index()
train_data_grouped.rename(columns={'age': 'count'}, inplace=True)

train_data_sort = train_data_grouped.sort_values(['count', 'ProductName'], ascending=[0, 1]) 
train_data_sort['rank'] = train_data_sort['count'].rank(ascending=0, method='first') 
popularity_recommendations = train_data_sort.head(5) 



def season(request):
    return render(request,'season_menu.html')

def summer(request):
    return render(request,'season/summer.html')

def fall(request):
    return render(request,'season/fall.html')

def spring(request):
    return render(request,'season/spring.html')

def winter(request):
    return render(request,'season/winter.html')




def recommend(age):     
    user_recommendations = popularity_recommendations  
    user_recommendations['age_based_recommendation'] = age 
      
 
    cols = user_recommendations.columns.tolist() 
    cols = cols[-1:] + cols[:-1] 
    user_recommendations = user_recommendations[cols] 
          
    return user_recommendations

df_CF = pd.concat([train_data, test_data]).reset_index()

pivot_df = df_CF.pivot_table(index=['age','Gender'],columns ='ProductName', values = 'ratings').fillna(0)

# Filter the DataFrame for males
df_men = df_CF[df_CF['Gender'] == 'Men']

# Create the pivot table for males
pivot_df_men = df_men.pivot_table(index=['age', 'Gender'], columns='ProductName', values='ratings').fillna(0)

# Filter the DataFrame for males
df_women = df_CF[df_CF['Gender'] == 'Women']

# Create the pivot table for males
pivot_df_women = df_women.pivot_table(index=['age', 'Gender'], columns='ProductName', values='ratings').fillna(0)

# Filter the DataFrame for males
df_boys = df_CF[df_CF['Gender'] == 'Boys']

# Create the pivot table for males
pivot_df_boys = df_boys.pivot_table(index=['age', 'Gender'], columns='ProductName', values='ratings').fillna(0)

# Filter the DataFrame for males
df_girls = df_CF[df_CF['Gender'] == 'Girls']

# Create the pivot table for males
pivot_df_girls = df_girls.pivot_table(index=['age', 'Gender'], columns='ProductName', values='ratings').fillna(0)

pivot_df['user_index'] = np.arange(0, pivot_df.shape[0], 1)

pivot_df.set_index(['user_index'], inplace=True)



# Convert pivot table to numpy array
pivot_array = pivot_df.values

# Perform SVD
U, sigma, Vt = svds(pivot_array, k=10)

pivot_array_women = pivot_df_women.values

# Perform SVD for 'Women' gender category
U_women, sigma_women, Vt_women = svds(pivot_array_women, k=10)

pivot_array_men = pivot_df_men.values

# Perform SVD for 'Women' gender category
U_men, sigma_men, Vt_men = svds(pivot_array_men, k=10)

pivot_array_boys = pivot_df_boys.values

# Perform SVD for 'Women' gender category
U_boys, sigma_boys, Vt_boys = svds(pivot_array_boys, k=10)

pivot_array_girls = pivot_df_girls.values

# Perform SVD for 'Women' gender category
U_girls, sigma_girls, Vt_girls = svds(pivot_array_girls, k=10)

sigma = np.diag(sigma)

sigma_women = np.diag(sigma_women)

sigma_men = np.diag(sigma_men)

sigma_boys = np.diag(sigma_boys)

sigma_girls = np.diag(sigma_girls)

user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
preds_df = pd.DataFrame(user_predicted_ratings, columns = pivot_df.columns)

# Compute predictions for the 'Women' gender category
user_predicted_ratings_women = np.dot(np.dot(U_women, sigma_women), Vt_women)
preds_df_women = pd.DataFrame(user_predicted_ratings_women, columns=pivot_df_women.columns)

# Compute predictions for the 'Women' gender category
user_predicted_ratings_men = np.dot(np.dot(U_men, sigma_men), Vt_men)
preds_df_men = pd.DataFrame(user_predicted_ratings_men, columns=pivot_df_men.columns)

# Compute predictions for the 'Women' gender category
user_predicted_ratings_boys = np.dot(np.dot(U_boys, sigma_boys), Vt_boys)
preds_df_boys = pd.DataFrame(user_predicted_ratings_boys, columns=pivot_df_boys.columns)

# Compute predictions for the 'Women' gender category
user_predicted_ratings_girls = np.dot(np.dot(U_girls, sigma_girls), Vt_girls)
preds_df_girls = pd.DataFrame(user_predicted_ratings_girls, columns=pivot_df_girls.columns)


s=[]
preicted_lists=[]

warnings.filterwarnings('ignore')

# Define gender dictionary
gender_dict = {0:'Men', 1:'Women'}

model = load_model("model.h5")



def homepage(request):
    return render(request, 'homepage.html')





def getFaceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    faceBoxes = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, faceBoxes

def age_gender_detector(request):
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"

    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"

    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    video = cv2.VideoCapture(0)

    padding = 20

    output_dir = "Dirs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    last_capture_time = datetime.now()

    while True:
        hasFrame, vidFrame = video.read()

        if not hasFrame:
            cv2.waitKey()
            break

        frame, faceBoxes = getFaceBox(faceNet, vidFrame)

        if not faceBoxes:
            print("No face detected")
        else:
            current_time = datetime.now()
            if (current_time - last_capture_time).seconds >= 5:
                for faceBox in faceBoxes:
                    face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                               max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]
                    # Save the image to a temporary file
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_img_file:
                        temp_img_path = temp_img_file.name
                        cv2.imwrite(temp_img_path, face)
                    # Pass the temporary file path to the predict function
                    predict(temp_img_path, pivot_df, preds_df, num_recommendations)
                    # Remove the temporary file
                    os.unlink(temp_img_path)
                    last_capture_time = current_time

        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                   max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(blob)
            genderPred = genderNet.forward()
            gender = genderList[genderPred[0].argmax()]

            ageNet.setInput(blob)
            agePred = ageNet.forward()
            age = ageList[agePred[0].argmax()]

            labelGender = "{}".format("Gender : " + gender)
            labelAge = "{}".format("Age : " + age + "Years")
            cv2.putText(frame, labelGender, (faceBox[0], faceBox[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, labelAge, (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Age-Gender Detector", frame)
        
        # Listen for keyboard input
        key = cv2.waitKey(1)
        if key == 49:  # If '1' is pressed
            return redirect('homepage')
            break

    # Release the video capture and destroy any OpenCV windows
    video.release()
    cv2.destroyAllWindows()

    return render(request, 'camera.html', {'video': video})





def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']

            # Save the uploaded file to a temporary location
            file_path = "E:\\SSP\\Strategic\\save" + file.name
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            # Parse the CSV file
            df = pd.read_csv(file_path)

            # Extract age and gender information
            # (Assuming the CSV file has 'Age' and 'Gender' columns)
            ages = df['Age']
            genders = df['Gender']

            # Make predictions for each row in the CSV file
            s = []
            num_recommendations = 20
            for age, gender in zip(ages, genders):
                if gender == 'male':
                    pivot_df = pivot_df_men
                    preds_df = preds_df_men
                elif gender == 'female':
                    pivot_df = pivot_df_women
                    preds_df = preds_df_women
                recommend_items(age, pivot_df, preds_df, num_recommendations)  # Implement this function based on your recommendation logic
                messages.success(request, 'File uploaded successfully!')
    else:
        form = UploadFileForm()

    return render(request, 'upload.html', {'form': form})




def off(request, video):
    # Release the webcam feed
    video.release()
    # Close any OpenCV windows
    cv2.destroyAllWindows()




# Define a function to predict gender and age from an image
def predict(image_path, pivot_df, preds_df, num_recommendations):
    # Load the image
    img = load_img(image_path, color_mode="grayscale")
    img = img.resize((128, 128), 3)  # Using numerical value for anti-aliasing
    img = np.array(img)/255

    # Make predictions using the model
    pred = model.predict(img.reshape(1, 128, 128, 1))
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])
    if(pred_age<=18):
        if(pred_gender=='Men'):
            pred_gender='Boys'
        elif(pred_gender=='Women'):
            pred_gender='Girls'
    agemin = pred_age - 5
    agemax = pred_age + 5
    print("Predicted Gender:", pred_gender, "Predicted Age:", str(agemin), "-", str(agemax))
    age = pred_age
    gen=pred_gender
    
    # Select the appropriate pivot_df and preds_df based on the predicted gender
    if pred_gender == 'Men':
        pivot_df = pivot_df_men
        preds_df = preds_df_men
    elif pred_gender == 'Women':
        pivot_df = pivot_df_women
        preds_df = preds_df_women
    elif pred_gender == 'Boys':
        pivot_df = pivot_df_boys
        preds_df = preds_df_boys
    elif pred_gender == 'Girls':
        pivot_df = pivot_df_girls
        preds_df = preds_df_girls
    # Call the recommend_items function
    recommend_items(age, pivot_df, preds_df, num_recommendations) 
    
    
    
    
    
# Define the recommend_items function
def recommend_items(age,pivot_df, preds_df, num_recommendations):
    global temp
    if age < 1 or age > len(preds_df):
        return  # Skip recommendation if age is out of range
    user_idx = age - 1
    sorted_user_predictions = preds_df.iloc[user_idx].sort_values(ascending=False)
    temp =pd.concat([sorted_user_predictions], axis=1)
    
    temp.index.name = 'Recommended Items'
    temp.columns = ['user_predictions']
    s.append(temp)

        
# Define the directory containing the images
image_directory = "Dirs"
# Get a list of all files in the directory
image_files = os.listdir(image_directory)

num_recommendations = 20
# Define the number of recommendations
# Iterate through each image in the directory
for image_file in image_files:
    # Construct the full path to the image
    image_path = os.path.join(image_directory, image_file)
    
    # Perform prediction and recommendation for the current image
    predict(image_path, pivot_df, preds_df, num_recommendations)


def recommend_menu(request):
    return render(request, 'menu.html')







def placement(request):
    return render(request,'image_map.html')








def get_price_from_product_name(product_name):
    product_row = df[df['ProductName'] == product_name]

    if not product_row.empty:
    # Get the price of the product
        price = product_row['Price (INR)'].iloc[0]
        return price
def get_category_from_product_name(product_name):
    product_row = df[df['ProductName'] == product_name]

    if not product_row.empty:
    # Get the price of the product
        category = product_row['category'].iloc[0]
        return category

def get_price_from_season_name(product_name):
    product_row = sedf[sedf['ProductName'] == product_name]

    if not product_row.empty:
    # Get the price of the product
        price = product_row['Price'].iloc[0]
        return price
def get_category_from_season_name(product_name):
    product_row = sedf[sedf['ProductName'] == product_name]

    if not product_row.empty:
    # Get the price of the product
        category = product_row['category'].iloc[0]
        return category

def get_id_from_season_name(product_name):
    product_row = sedf[sedf['ProductName'] == product_name]

    if not product_row.empty:
    # Get the price of the product
        idd = product_row['id'].iloc[0]
        return idd


def combine_recommendations(request):
    # Combine recommendations from multiple images
    combined_df = pd.concat(s)
    combined_df = combined_df.drop_duplicates()
    p=[]
    # Sort the combined DataFrame based on 'user_predictions'
    sorted_combined_df = combined_df.sort_values(by='user_predictions', ascending=False)
    p.append(sorted_combined_df)
    sorted_combined_df=sorted_combined_df.head(20)
    just_names = sorted_combined_df.index
    # Initialize total cost price
    erack_price = 0
    # Prepare product details list
    product_details = []
    for i, product_name in enumerate(just_names):
        # Get the price of the product
        price = get_price_from_product_name(product_name)
        
        # Get the category of the product
        category = get_category_from_product_name(product_name)
        product_details.append({
            'id': i + 1,  # Assigning IDs starting from 1
            'name': product_name,
            'price': price,
            'category': category
        })
        
        # Increment the total price
        erack_price += price
    context = {
        'product_details': product_details,
        'erack_price': erack_price
    }
    
    # Render the template with the provided context
    return render(request, 'combined_recommendations.html', context)


def erack(request):
    # Combine recommendations from multiple images
    combined_df = pd.concat(s)
    # Sort the combined DataFrame based on 'user_predictions'
    sorted_combined_df = combined_df.sort_values(by='user_predictions', ascending=False)
    sorted_combined_df=sorted_combined_df.drop_duplicates()
    entrance = sorted_combined_df.iloc[0:21]
    just_names = entrance.index
    
    # Initialize total cost price
    erack_price = 0

    # Prepare product details list
    product_details = []
    
    # Iterate through each recommended product
    for i, product_name in enumerate(just_names):
        # Get the price of the product
        price = get_price_from_product_name(product_name)
        
        # Get the category of the product
        category = get_category_from_product_name(product_name)
        
        # Append product details to the list

        
        # Append product details to the list
        product_details.append({
            'id': i + 1,  # Assigning IDs starting from 1
            'name': product_name,
            'price': price,
            'category': category
        })
        
        # Increment the total price
        erack_price += price


    context = {
        'product_details': product_details,
        'erack_price': erack_price
    }
    
    # Render the template with the provided context
    return render(request, 'entrance_racks.html', context)


def brack(request):
    # Combine recommendations from multiple images
    combined_df = pd.concat(s)
    # Sort the combined DataFrame based on 'user_predictions'
    sorted_combined_df = combined_df.sort_values(by='user_predictions', ascending=False)
    sorted_combined_df=sorted_combined_df.drop_duplicates()
    entrance = sorted_combined_df.iloc[131:200]
    just_names = entrance.index
    
    # Initialize total cost price
    erack_price = 0

    # Prepare product details list
    product_details = []
    
    # Iterate through each recommended product
    for i, product_name in enumerate(just_names):
        # Get the price of the product
        price = get_price_from_product_name(product_name)
        
        # Get the category of the product
        category = get_category_from_product_name(product_name)
        
        # Append product details to the list
        product_details.append({
            'id': i + 1,  # Assigning IDs starting from 1
            'name': product_name,
            'price': price,
            'category': category
        })
        
        # Increment the total price
        erack_price += price


    context = {
        'product_details': product_details,
        'erack_price': erack_price
    }
    
    # Render the template with the provided context
    return render(request, 'backrack.html', context)


def crack(request):
    # Combine recommendations from multiple images
    combined_df = pd.concat(s)
    # Sort the combined DataFrame based on 'user_predictions'
    sorted_combined_df = combined_df.sort_values(by='user_predictions', ascending=False)
    sorted_combined_df=sorted_combined_df.drop_duplicates()
    entrance = sorted_combined_df.iloc[21:51]
    just_names = entrance.index
    
    # Initialize total cost price
    erack_price = 0

    # Prepare product details list
    product_details = []
    
    # Iterate through each recommended product
    for i, product_name in enumerate(just_names):
        # Get the price of the product
        price = get_price_from_product_name(product_name)
        
        # Get the category of the product
        category = get_category_from_product_name(product_name)
        
        # Append product details to the list
        product_details.append({
            'id': i + 1,  # Assigning IDs starting from 1
            'name': product_name,
            'price': price,
            'category': category
        })
        
        # Increment the total price
        erack_price += price


    context = {
        'product_details': product_details,
        'erack_price': erack_price
    }
    
    # Render the template with the provided context
    return render(request, 'central_rack.html', context)

def left(request):
    # Combine recommendations from multiple images
    combined_df = pd.concat(s)
    # Sort the combined DataFrame based on 'user_predictions'
    sorted_combined_df = combined_df.sort_values(by='user_predictions', ascending=False)
    sorted_combined_df=sorted_combined_df.drop_duplicates()
    entrance = sorted_combined_df.iloc[51:91]
    just_names = entrance.index
    
    # Initialize total cost price
    erack_price = 0

    # Prepare product details list
    product_details = []
    
    # Iterate through each recommended product
    for i, product_name in enumerate(just_names):
        # Get the price of the product
        price = get_price_from_product_name(product_name)
        
        # Get the category of the product
        category = get_category_from_product_name(product_name)
        
        # Append product details to the list
        product_details.append({
            'id': i + 1,  # Assigning IDs starting from 1
            'name': product_name,
            'price': price,
            'category': category
        })
        
        # Increment the total price
        erack_price += price


    context = {
        'product_details': product_details,
        'erack_price': erack_price
    }
    
    # Render the template with the provided context
    return render(request, 'left_rack.html', context)

def right(request):
    # Combine recommendations from multiple images
    combined_df = pd.concat(s)
    # Sort the combined DataFrame based on 'user_predictions'
    sorted_combined_df = combined_df.sort_values(by='user_predictions', ascending=False)
    sorted_combined_df=sorted_combined_df.drop_duplicates()
    entrance = sorted_combined_df.iloc[91:131]
    just_names = entrance.index
    
    # Initialize total cost price
    erack_price = 0

    # Prepare product details list
    product_details = []
    
    # Iterate through each recommended product
    for i, product_name in enumerate(just_names):
        # Get the price of the product
        price = get_price_from_product_name(product_name)
        
        # Get the category of the product
        category = get_category_from_product_name(product_name)
        
        # Append product details to the list
        product_details.append({
            'id': i + 1,  # Assigning IDs starting from 1
            'name': product_name,
            'price': price,
            'category': category
        })
        
        # Increment the total price
        erack_price += price


    context = {
        'product_details': product_details,
        'erack_price': erack_price
    }
    
    # Render the template with the provided context
    return render(request, 'right_rack.html', context)

def add_to_inventory(request):
    if request.method == 'POST':
        # Create instances of the forms
        product_form = ProductForm(request.POST)
        
        # Process individual item additions
        if product_form.is_valid():
            product_form.save()  # Save the product
            return redirect('/homepage/placement#')  # Redirect to wherever appropriate
            
        # Process selected items
        elif 'selected_products' in request.POST:
            selected_product_ids = request.POST.getlist('selected_products')
            for product_id in selected_product_ids:
                product_name = request.POST.get('name' + product_id)
                product_price = request.POST.get('price' + product_id)
                prototal_price=product_price
                
                # Save selected products with quantities
                product = Product(name=product_name, price=product_price,prototal=product_price)
                product.save()
            return redirect('/homepage/placement#')  # Redirect to wherever appropriate

    return redirect('/homepage/placement#')


def add_to_inventory_back(request):
    if request.method == 'POST':
        # Create instances of the forms
        product_form = ProductForm(request.POST)
        
        # Process individual item additions
        if product_form.is_valid():
            product_form.save()  # Save the product
            return redirect('/homepage/recommendation')  # Redirect to wherever appropriate
            
        # Process selected items
        elif 'selected_products' in request.POST:
            selected_product_ids = request.POST.getlist('selected_products')
            for product_id in selected_product_ids:
                product_name = request.POST.get('name' + product_id)
                product_price = request.POST.get('price' + product_id)
                prototal_price=product_price
                
                # Save selected products with quantities
                product = Product(name=product_name, price=product_price,prototal=product_price)
                product.save()
            return redirect('/homepage/recommendation')  # Redirect to wherever appropriate

    return redirect('/homepage/recommendation')

def inventory(request):
    products = Product.objects.all()
    total_price = sum(product.prototal for product in products)

    if request.method == 'POST':
        for product in products:  # Loop through all products to handle each form individually
            # Construct field names based on product ID
            quantity_field_name = f'number_{product.id}'
            product_id_field_name = f'product_id_{product.id}'

            # Check if the form fields exist in the request.POST data
            if quantity_field_name in request.POST and product_id_field_name in request.POST:
                quantity = int(request.POST[quantity_field_name])  # Extract quantity value
                product_id = int(request.POST[product_id_field_name])  # Extract product ID

                product = Product.objects.get(pk=product_id)
                product.prototal = product.price * quantity
                product.save()

        # Recalculate total price after updating product quantities
        total_price = sum(product.prototal for product in products)
        return redirect('inventory')

    return render(request, 'display_inventory.html', {'products': products, 'total_price': total_price})


def delete(request):
    if request.method == 'POST':
        # Get the list of selected product IDs from the form
        selected_product_ids = request.POST.getlist('selected_products')

        # Iterate through the selected product IDs
        for product_id in selected_product_ids:
            # Retrieve the product instance to be deleted
            product = get_object_or_404(Product, pk=product_id)

            # Delete the product instance
            product.delete()

        # Redirect back to the inventory page after deletion
        return redirect('inventory')

    # Handle cases where the request method is not POST
    return HttpResponseBadRequest()

    


def payment(request):
    return render(request,'payment.html')
def inside_payment(request):
    total_price=0
    products = Product.objects.all()
    total_price = sum(product.prototal for product in products)
    return render(request,'inside_payment.html',{'price':total_price})

def success(request):
    return render(request,'success.html')

def aboutus(request):
    return render(request,'about.html')

#______________________________________________________________________________________________________________________________________




def add_to_inventory_season(request):
    if request.method == 'POST':
        # Create instances of the forms
        product_form = ProductForm(request.POST)
        
        # Process individual item additions
        if product_form.is_valid():
            product_form.save()  # Save the product
            return redirect('/homepage/tanktop')  # Redirect to wherever appropriate
            
        # Process selected items
        elif 'selected_products' in request.POST:
            selected_product_ids = request.POST.getlist('selected_products')
            for product_id in selected_product_ids:
                product_name = request.POST.get('name' + product_id)
                product_price = request.POST.get('price' + product_id)
                prototal_price=product_price
                
                # Save selected products with quantities
                product = Product(name=product_name, price=product_price,prototal=product_price)
                product.save()
            return redirect('/homepage/season')  # Redirect to wherever appropriate

    return redirect('/homepage')



def tanktop(request):
    tanktop_products = sedf[sedf['Category'] == 'tanktops']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/tanktop.html', context)

def skirts(request):
    tanktop_products = sedf[sedf['Category'] == 'skirts']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/tanktop.html', context)
def shorts(request):
    tanktop_products = sedf[sedf['Category'] == 'shorts']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/tanktop.html', context)

def strawhats(request):
    tanktop_products = sedf[sedf['Category'] == 'Strawhats']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/tanktop.html', context)

def sunglasses(request):
    tanktop_products = sedf[sedf['Category'] == 'Sunglasses']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/tanktop.html', context)

def Breathablefabric(request):
    tanktop_products = sedf[sedf['Category'] == 'Breathablefabric']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/tanktop.html', context)

def flipflops(request):
    tanktop_products = sedf[sedf['Category'] == 'flipflops']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/tanktop.html', context)

def Breathablefabric(request):
    tanktop_products = sedf[sedf['Category'] == 'Breathablefabric']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/tanktop.html', context)

def sandals(request):
    tanktop_products = sedf[sedf['Category'] == 'sandals']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/tanktop.html', context)

def brightcoloredclothing(request):
    tanktop_products = sedf[sedf['Category'] == 'brightcoloredclothing']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/tanktop.html', context)
def canvas(request):
    tanktop_products = sedf[sedf['Category'] == 'canvas']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/tanktop.html', context)
def belts(request):
    tanktop_products = sedf[sedf['Category'] == 'belts']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/tanktop.html', context)


def coverups(request):
    tanktop_products = sedf[sedf['Category'] == 'coverups']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/tanktop.html', context)

def scarves(request):
    tanktop_products = sedf[sedf['Category'] == 'scarves']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/scarves.html', context)

def scarves(request):
    tanktop_products = sedf[sedf['Category'] == 'scarves']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/scarves.html', context)

def pastel(request):
    tanktop_products = sedf[sedf['Category'] == 'pastel']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/scarves.html', context)
def lightjacket(request):
    tanktop_products = sedf[sedf['Category'] == 'lightjacket']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/scarves.html', context)
def floral(request):
    tanktop_products = sedf[sedf['Category'] == 'floral']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/scarves.html', context)
def denim(request):
    tanktop_products = sedf[sedf['Category'] == 'denim']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/scarves.html', context)
def printedskirts(request):
    tanktop_products = sedf[sedf['Category'] == 'printedskirts']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/scarves.html', context)
def tshirts(request):
    tanktop_products = sedf[sedf['Category'] == 'tshirts']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/scarves.html', context)
def knitwear(request):
    tanktop_products = sedf[sedf['Category'] == 'knitwear']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/scarves.html', context)
def crossbody(request):
    tanktop_products = sedf[sedf['Category'] == 'crossbody']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/scarves.html', context)
def whitesneakers(request):
    tanktop_products = sedf[sedf['Category'] == 'whitesneakers']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/scarves.html', context)
def jewel(request):
    tanktop_products = sedf[sedf['Category'] == 'jewel']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/scarves.html', context)

def sweater(request):
    tanktop_products = sedf[sedf['Category'] == 'sweater']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/sweater.html', context)
def boots(request):
    tanktop_products = sedf[sedf['Category'] == 'boots']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/sweater.html', context)
def coats(request):
    tanktop_products = sedf[sedf['Category'] == 'coats']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/sweater.html', context)
def leatherjacket(request):
    tanktop_products = sedf[sedf['Category'] == 'leatherjacket']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/sweater.html', context)
def corduroypants(request):
    tanktop_products = sedf[sedf['Category'] == 'corduroypants']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/sweater.html', context)
def beanies(request):
    tanktop_products = sedf[sedf['Category'] == 'beanies']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/sweater.html', context)
def berets(request):
    tanktop_products = sedf[sedf['Category'] == 'berets']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/sweater.html', context)
def cardigans(request):
    tanktop_products = sedf[sedf['Category'] == 'cardigans']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/sweater.html', context)
def flannelshirts(request):
    tanktop_products = sedf[sedf['Category'] == 'flannelshirts']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/sweater.html', context)
def leggings(request):
    tanktop_products = sedf[sedf['Category'] == 'leggings']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/sweater.html', context)
def suedehandbags(request):
    tanktop_products = sedf[sedf['Category'] == 'suedehandbags']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/sweater.html', context)


def Jacket(request):
    tanktop_products = sedf[sedf['Category'] == 'Jacket']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/Jacket.html', context)
def WoolCoat(request):
    tanktop_products = sedf[sedf['Category'] == 'WoolCoat']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/Jacket.html', context)
def CashmereSweater(request):
    tanktop_products = sedf[sedf['Category'] == 'CashmereSweater']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/Jacket.html', context)
def SnowBoots(request):
    tanktop_products = sedf[sedf['Category'] == 'SnowBoots']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/Jacket.html', context)
def TurtleneckSweater(request):
    tanktop_products = sedf[sedf['Category'] == 'TurtleneckSweater']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/Jacket.html', context)
def DarkColoredJeans(request):
    tanktop_products = sedf[sedf['Category'] == 'DarkColoredJeans']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/Jacket.html', context)
def WoolSkirts(request):
    tanktop_products = sedf[sedf['Category'] == 'WoolSkirts']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/Jacket.html', context)
def VelvetDresses(request):
    tanktop_products = sedf[sedf['Category'] == 'VelvetDresses']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/Jacket.html', context)
def OvertheKneeBoots(request):
    tanktop_products = sedf[sedf['Category'] == 'OvertheKneeBoots']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/Jacket.html', context)
def StatementBoots(request):
    tanktop_products = sedf[sedf['Category'] == 'StatementBoots']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/Jacket.html', context)
def Gloves(request):
    tanktop_products = sedf[sedf['Category'] == 'Gloves']
    
    # Initialize total price
    total_price = 0
    
    # Prepare product details list
    product_details = []
    
    # Iterate through tanktop products
    for index, row in tanktop_products.iterrows():
        # Get product details
        product_id = row['ID']
        product_name = row['ProductName']
        product_price = int(row['Price'])
        product_category = row['Category']
        
        # Append product details to the list
        product_details.append({
            'id': product_id,
            'name': product_name,
            'price': product_price,
            'category': product_category
        })
        
        # Increment total price
        total_price += product_price
    
    # Prepare context for rendering template
    context = {
        'product_details': product_details,
        'total_price': total_price
    }
    
    # Render the template with the provided context
    return render(request, 'sub/Jacket.html', context)

