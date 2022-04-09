
## practical rm -rf caltech101/BACKGROUND_Google model = InceptionV3(weights='imagenet',


import numpy as np
from numpy.linalg import norm
import pickle
from tqdm import tqdm, tqdm_notebook
import os
import time

from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tf.keras.preprocessing import image
# from tf.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D


## Earlier >> #model = ResNet50(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
def model_picker(name):
    if (name == 'vgg16'):
        model = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(224, 224, 3),
                      pooling='max')
    elif (name == 'vgg19'):
        model = VGG19(weights='imagenet',
                      include_top=False,
                      input_shape=(224, 224, 3),
                      pooling='max')
    elif (name == 'mobilenet'):
        model = MobileNet(weights='imagenet',
                          include_top=False,
                          input_shape=(224, 224, 3),
                          pooling='max',
                          depth_multiplier=1,
                          alpha=1)
    elif (name == 'inception'):
        model = InceptionV3(weights='imagenet',
                            include_top=False,
                            input_shape=(224, 224, 3),
                            pooling='max')
    elif (name == 'resnet'):
        model = ResNet50(weights='imagenet',
                         include_top=False,
                         input_shape=(224, 224, 3),
                        pooling='max')
    elif (name == 'xception'):
        model = Xception(weights='imagenet',
                         include_top=False,
                         input_shape=(224, 224, 3),
                         pooling='max')
    else:
        print("Specified model not available")
    print("----Type(model---",type(model)) #<class 'keras.engine.functional.Functional'>
    return model


def extract_features(img_path, model):
    #
    input_shape = (224, 224, 3)
    img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    #
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    print("type(normalized_features)---",type(normalized_features))
    print("---normalized_features.shape---",normalized_features.shape)
    print("---normalized_features.ndim---",normalized_features.ndim)
    print("len(normalized_features)-----",len(normalized_features))
    print("normalized_features)-----",normalized_features[0:5])
    return normalized_features


# Now it’s time to extract features for the entire dataset. First, we get
# all the filenames with this handy function, which recursively looks
# for all the image files (defined by their extensions) under a
# directory:

extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def get_file_list(root_dir):
    file_list = []
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
    print("---file_list----",file_list)
    return file_list

# Nearest Neighbours 
import numpy as np
import pickle
from tqdm import tqdm, tqdm_notebook
import random
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import PIL
from PIL import Image
from sklearn.neighbors import NearestNeighbors

import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline


# Helper function to get the classname
def classname(str):
    return str.split('/')[-2]

# Helper function to get the classname and filename
def classname_filename(str):
    return str.split('/')[-2] + '/' + str.split('/')[-1]


# Helper functions to plot the nearest images given a query image
def plot_nn_images(filenames, distances):
    print("----filenames----",filenames)
    import os
    from datetime import datetime
    dt_time_now = datetime.now()
    hour_now = dt_time_now.strftime("_%m_%d_%Y_%H")

    images = []
    for filename in filenames:
        print("----filename----",filename)

        images.append(mpimg.imread(filename))
    plt.figure(figsize=(20, 10))
    columns = 4
    print("---plot_nn_images--len(images----",len(images))
    print("---plot_nn_images--type(images[0]----",type(images[0]))

    for iter_n, image in enumerate(images):
        print("---plot_nn_images-->>iter_n---",iter_n)
        ax = plt.subplot(len(images) / columns + 1, columns, iter_n + 1)
        if iter_n == 0:
            ax.set_title("Query Image\n" + classname_filename(filenames[iter_n]))
            init_QueryImgName = classname_filename(filenames[iter_n]) # TODO -'out_put_dir/_Faces/image_0015.jpg_.pdf'
            img_className = init_QueryImgName.split('/')[0]
            img_fileName = init_QueryImgName.split('/')[1]
            print("--plot_nn_images-img_className-",img_className)
            print("--plot_nn_images-img_fileName-",img_fileName)

            img_className_dir_path = '../output_dir/_QueryImg_ClassName_/'+str(img_className)
            #hourly_save_dir = os.path.join(f'{hour_now}',img_className_dir_path)
            if not os.path.exists(img_className_dir_path):
                os.makedirs(img_className_dir_path)
            plt.imsave(str(img_className_dir_path)+"/"+str(img_fileName)+"_.png", image)  #TODO -- Fix this getting 2KB PDF Files
        else:
            ax.set_title("Similar Image\n" + classname_filename(filenames[iter_n]) + "\nDistance: " + str(float("{0:.2f}".format(distances[iter_n]))))
            init_QueryImgName = classname_filename(filenames[iter_n]) # TODO -'out_put_dir/_Faces/image_0015.jpg_.pdf'
            img_className = init_QueryImgName.split('/')[0]
            img_fileName = init_QueryImgName.split('/')[1]
            
            img_className_dir_path = '../output_dir/_SimilarImg_ClassName_/'+str(img_className)+'_nn_distance_'+ str(float("{0:.2f}".format(distances[iter_n])))
            if not os.path.exists(img_className_dir_path):
                os.makedirs(img_className_dir_path)
            plt.imsave(str(img_className_dir_path)+str(img_fileName)+"_.png", image) 

        plt.imshow(image)
        plt.gcf().set_dpi(300)
        plt.show()#block=False)

        #plt.savefig(str(hourly_plots_dir)+"/"+str(pdf_fileName)+"_.pdf", format='pdf', dpi=300)
        # To save the plot in a high definition format i.e. PDF, uncomment the following line:
        #plt.savefig('results/' + str(random.randint(0,10000))+'.pdf', format='pdf', dpi=1000)
        # We will use this line repeatedly in our code.

def wrapper_plot_nn_images(feature_list,filenames):
    #
    num_images = 100 # this is Not defined in ORIGINAL BOOK Code - so probably its the TOTAL IMAGES Count 
    for iter_k in range(6):
        print("-wrapper_plot_nn_images--iter---",iter_k)
        # random_image_index = random.randint(0, num_images) # this is Not defined in ORIGINAL BOOK Code - so probably its the TOTAL IMAGES Count 
        # print("---random_image_index--",random_image_index)
        random_image_index = 200
        distances, indices = neighbors.kneighbors([feature_list[random_image_index]])
        # Don't take the first closest image as it will be the same image
        #print("----FileName with DISTANCE ==iter_j in range(1,4)-which is NOT index == 0 --Same IMAGE--",filenames[indices[0][iter_k]])
        similar_image_paths = [filenames[random_image_index]] + [filenames[indices[0][iter_j]] for iter_j in range(1, 4)]
        plot_nn_images(similar_image_paths, distances[0]) # original book code 

        """
        You can also experiment with different distances like Minkowski
        distance, Manhattan distance, Jaccardian distance, and weighted
        Euclidean distance (where the weight is the contribution of each
        feature as explained in pca.explained_variance_ratio_ ).
        """


if __name__ == "__main__":
    import tensorflow
    import time
    #
    print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

    # model_architecture = 'resnet'
    # model = model_picker(model_architecture)

    # normalized_features = extract_features('./known_face_dir/trump/trump_1.png', model)
    # normalized_features1 = extract_features('./known_face_dir/trump/trump_2.jpeg', model)

    # path to the your datasets
    # root_dir = '../datasets/caltech101'
    # #root_dir = "./known_face_dir/"
    # #get_file_list(root_dir)
    # filenames = sorted(get_file_list(root_dir))

    # feature_list = []
    # for i in tqdm_notebook(range(len(filenames))):
    #     feature_list.append(extract_features(filenames[i], model))  
    # print("---len(feature_list----",len(feature_list))
    # print("---len(feature_list[0]----",len(feature_list[0]))
    # # Save the lists in Pickle format
    # pickle.dump(feature_list, open('pickle_files/features_caltech101.pickle', 'wb'))
    # pickle.dump(filenames, open('pickle_files/filenames_caltech101.pickle','wb'))

    filenames = pickle.load(open('../pickle_files/filenames_caltech101.pickle','rb'))
    feature_list = pickle.load(open('../pickle_files/features_caltech101.pickle','rb'))    
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute',metric='euclidean').fit(feature_list)

    wrapper_plot_nn_images(feature_list,filenames)
   

    # below own hacked code 
    for iter_k in range(25,30,1):
        distances, indices = neighbors.kneighbors([feature_list[iter_k]])
        print("---iter_k--",iter_k)
        # print("---distances--",type(distances))
        # print("---indices--",type(indices))
        print("---distances--",distances.shape)
        print("---indices--",indices.shape)
        print("---distances--",distances)
        print("---indices--",indices)
        print("----filenames[iter_k]----",filenames[iter_k])
        plt.imshow(mpimg.imread(filenames[iter_k]))
        plt.show()#block=False)
        time.sleep(0.3)
        plt.close('all')
        print("----FileName of the DISTANCE == 0 -which is Same IMAGE--",filenames[indices[0][0]])
        plt.imshow(mpimg.imread(filenames[indices[0][0]])) # Same IMAGE as the Image Shown Above 
        plt.show()#block=False)
        time.sleep(0.3)
        #plt.close('all')

        print("----FileName of the 1st NN----",filenames[indices[0][1]])
        plt.imshow(mpimg.imread(filenames[indices[0][1]])) # 1st Nearest Neighbour of Image Shown Above 
        plt.show(block=False)
        #plt.close('all')

        print("----FileName of the 2nd NN----",filenames[indices[0][2]])
        plt.imshow(mpimg.imread(filenames[indices[0][2]])) # 2nd Nearest Neighbour of Image Shown Above 
        plt.show(block=False)
        plt.close('all')
  
  












