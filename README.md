## Project description
Check The Mole is a skin lesions classifier which takes the picture of the mole/lesion and tells if it looks more like a non-cancerous thing or like a cancerous one. 

This project is inspired by my personal experience as a parent when I had quite a stressful time waiting for test results for my kid. Having an approximation of how dangerous and urgent the lesion is reduces the stress of waiting for clear result. It also can help the specialists to be alert and extra attentive, providing earlier and more accurate skin cancer diagnostics.

Demo is here: https://share.streamlit.io/nataliakra/check_the_mole/main/streamlit_app_CNN_binary_new.py
## Dataset
The dataset for this repo can be found [here](https://drive.google.com/file/d/1qQhg56kZ8Tqh4Cv0a7Tjvpnx6YcnOzPG/view?usp=sharing).
The initial dataset is [Kaggle Skin Cancer MNIST](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000). During the data analysis I found that the set already contains augmented images so that one lesion (tracked with lesion_id in HAM10000_metadata.csv) may have from one to six corresponding pictures. 
All images belong to 7 classes: Actinic keratosis (akiec), Basal Cell Carcinoma (bcc), Benign Keratosis Like Lesion (bkl), Dermatofibroma (df), Melanoma (mel), Melanocytic Nevi (mv) and Vascular Lesion (vasc). The dataset is very unbalanced with one class represented by 54% of all images and two smallest classes represented by only 0.5% and 0.8%.
### Tackling Unbalanced Dataset
To tackle unbalanced dataset, I did the following:
-	Added more data by including images of Actinic Keratosis, Basal Cell Carcinoma and Melanoma from [Mendeley Data Skin Lesion Dataset](https://data.mendeley.com/datasets/zr7vgbcyr2/1). In the 1_Skin_lesions_Preprocess.ipynb file you can see that I only used three classes mentioned above and did not include the rest of the images as their classes did not correspond to the initial classes.
-	Tracked augmented images in the initial dataset, left the augmented images only for under sampled classes and removed the augmented ones from the largest class
-	Tried to use class_weights when building a model – this did not help much really
-	Tried to undersample the largest class – this worked well, but I did not really want to lose data as the dataset is not big
-	Tried to remove two smallest classes – again, it worked ok and the models could learn the remaining 5 classes with around 62% accuracy (CNN and transfer learning MobileNet) but it did no seem to be good enough for the project task
-	Ended up combining classes into 2 big classes, cancerous and non-cancerous. This was my to-go option for this project
## Preprocessing
As I used the Keras ImageDataGenerator with flow_from_directory option, I did all the preprecessing inside it (resize, put into array, scale). Building the appropriate directory structure is shown in 2_Skin_lesions_Dataset.ipynb. 
Please note that the data subset can slightly differ because of the random way of sampling data for training, validation and testing. If you would like to train the  model on the exactly same dataset as I did, please use the 2classes_df.csv file. 

## Modelling
Due to dataset challenges I had to try different models, including CNN and pretrained options like ResNet50 and MobileNet, with different number of classes. I ended up using the CNN for 2 classes to get better performance. The modelling code can be found in 3_Skin_lesions_Modelling.ipynb file. 
## Evaluation
The final model showed around 89% for precision and 90% for recall. Note that recall is the most important measurement in this case because we want to make sure that all cancerous lesions are recognized correctly and are not missed. The model evaluations including f1 score, confusion matrix and prediction on the batch dataset can be found in 4_Skin_lesions_Model_evaluation.ipynb
## App
Streamlit_app_CNN_binary.py along with model file (best_CNN_Datagen_2cl_categ.hdf5) contains everything you need to run the little demo app.
## Future improvements
The plan for project improvements includes gathering more data to make the dataset more balanced, implement transfer learning to get higher recall and to build an app better UI.
## Files description
### Data
-	HAM10000_metadata.csv – dataframe for the main dataset
-	metadata_additional.csv – dataframe for the additional data
-	2classes_df.csv – dataframe for final dataset which was used to parse images
### Preprocessing and modelling
-	1_Skin_lesions_Preprocess.ipynb
-	2_Skin_lesions_Dataset.ipynb
-	3_Skin_lesions_Modelling.ipynb
-	4_Skin_lesions_Model_evaluation.ipynb
-	
### App
-	Streamlit_app_CNN_binary.py
-	best_CNN_Datagen_2cl_categ.hdf5
Thank you!
 
