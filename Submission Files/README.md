# Deep Learning Project: Dehazing of Images using Conditional GANs

## Installing Required Libraries
1. Change directory to our project directory
```cd CSF425-DEEP-LEARNING-PROJECT-TASK-2```

2. Install requirements.txt
```pip install -r requirements.txt```

## Instructions for Executing Testing
1. Testing code is present in the file ```testing_code_template-task2.py```.

2. As told, you only need to change TEST_DATASET_HAZY_PATH to path of folder with hazy images and TEST_DATASET_OUTPUT_PATH to path of folder you want to save generated images to.

3. [Optional] By default, we use model-1 for testing and we have commented out the model giving the best performance according to our experiments (model - 5), because it's architecture is more complex. If you wish to test the second model, you can uncomment the lines ```model = GeneratorModel5()```, ```weightsPath = "generator_resnet.pth"``` and comment the lines ```model = GeneratorModel1()```, ```weightsPath = "generator_cgan.pth"```. 

4. Run the file.
```python testing_code_template-task2.py```


## Instructions for Executing Training

1. We conduct data-augmentation by creating a low-haze and medium-haze image from each clean image. The code for this is integrated into the training file. Once your dataset folder has 22,857 images after running augmentation, it won't augment on further runs

2. Set the variable ```root_dir``` in the train.py file to the directory where the train and val folders are present

3. [Optional] By default, we use model-1 for training and we have commented out the model giving the best performance according to our experiments (model - 5), because it's architecture is more complex. If you wish to test the second model, you can uncomment the line ```model = GeneratorModel5()``` and comment the line ```model = GeneratorModel1()```.

4. [Optional] You can modify the number of epochs as the ```num_epochs``` variable passed to the trainer class at the bottom of the train.py file. Moreover, you can also set ```use_l1_loss``` to false if you wish to omit it.
   
5. Run the train.py file.
```python train.py```

6. The code will show you that it is augmenting image files upto image number 7619. Afterwards, it automatically starts training.
