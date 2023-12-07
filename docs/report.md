# Capstone Proposal
 
## 1. Proposal Title: Integration of Deep Learning Techniques for Enhanced Patient Diagnostics in Healthcare

- **Author Name** - Harmankaranjit Singh Lohiya
- **Prepared for** - UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- **Semester** - Fall 2023
- <a href="https://github.com/DATA-606-2023-FALL-TUESDAY/Lohiya_HarmankaranjitSingh"><img align="left" src="https://img.shields.io/badge/-GitHub-CD5C5C?logo=github&style=flat" alt="icon | LinkedIn"/></a> 
  
- <a href="https://www.linkedin.com/in/harmankaranjit-singh-l-b33161251/"><img align="left" src="https://img.shields.io/badge/-LinkedIn-1E90FF?logo=linkedin&style=flat" alt="icon | GitHub"/></a>  
- **PowerPoint presentation file** - In Progress
- **YouTube video** - In Progress

# **Contents**
[Chapter 1: Introduction	2](#_toc147983902)

[1.1 Background	2](#_toc147983903)

[1.2 What is it about?	3](#_toc147983904)

[1.3 Why does it matter?	3](#_toc147983905)

[1.4 Aim of the Research Project and Definition of Objectives	4](#_toc147983906)

[1.5 Research Question	4](#_toc147983907)

[1.6 Chapter Summary	4](#_toc147983908)

[Chapter:2 Data	5](#_toc147983909)

[2.1 Data Sources	5](#_toc147983910)

[2.2 Data Size	5](#_toc147983911)

[2.3 Data Shape	5](#_toc147983912)

[2.4 Time Period	5](#_toc147983913)

[2.5 What does each row represent?	5](#_toc147983914)

[2.6 Data Dictionary:	6](#_toc147983915)

[2.7 Target/Label for ML Model:	6](#_toc147983916)

[2.8 Features/Predictors for ML Models:	6](#_toc147983917)

[2.9 Chapter Summary:	7](#_toc147983918)

[Chapter 3: Exploratory Data Analysis (EDA)	7](#_toc147983919)

[3.1 Introduction	7](#_toc147983920)

[3.2 Data Exploration with Jupyter Notebook	7](#_toc147983921)

[3.3 Focus on Target Variable and Features	8](#_toc147983922)

[3.4 Summary Statistics and Visualization	9](#_toc147983923)

[3.6 Data Cleansing , Transformation, Augmenting with External Data, Pre-processing	10](#_toc147983924)

[3.7 Chapter Summary	11](#_toc147983925)

[Chapter 4:  Model Training	12](#_toc147983926)

[4.1 Choice of Model for Predictive Analytics	12](#_toc147983927)

[4.2 Model Training Approach	12](#_toc147983928)

[4.3 Data Splitting Strategy	13](#_toc147983929)

[4.4 Python Packages and Libraries	13](#_toc147983930)

[Chapter 5 Conclusion	15](#_toc147983931)

[5.1 Summary and Potential Applications	15](#_toc147983932)

[5.2 Limitations	15](#_toc147983933)

[5.3 Lessons Learned	16](#_toc147983934)

[5.4 Future Research Directions	16](#_toc147983935)

[References	17](#_toc147983936)




## <a name="_toc147983902"></a>***Chapter 1: Introduction***
## <a name="_toc147983903"></a>***1.1 Background***
The invention and subsequent development of imaging methods have continuously improved the accuracy and dependability of identifying diseases in the constantly changing field of medical diagnostics. The recent incorporation of artificial intelligence and, particularly, deep learning, has been made possible by the digitalization of these imaging methods during the past few decades. The study aims to investigate the development, importance, and limitations of deep learning's contribution to improving patient diagnoses. (Suzuki, 2017). 

In the past, doctors and medical professionals were primarily responsible for evaluating the visual information included in medical pictures. Yet, numerical methods became increasingly important with the development of computerized imaging and the multifaceted nature of diagnostic work. The transformational advantages of deep learning was underlined by (Lee et.al, 2017) especially in applications requiring feature and visual identification. Deep learning models, particularly Convolutional Neural Networks , have shown competencies to automatically identify complex trends from large datasets, in contrast to classical machine learning, which frequently requires human extraction of features (Vaz et.al, 2021).

Due to its inherently customized architecture for imagery, CNNs have become an effective instrument in healthcare image analysis. In a thorough investigation of deep learning techniques for analysis of medical images, (Gu et.al, 2020) highlighted the effectiveness of CNNs for tasks including lesion identification and parts delineation.

In a noteworthy research, Esteva et al. (2017) used deep neural networks to categorize skin malignancies, outperforming board-approved doctors in terms of performance. Their findings highlighted the possibilities of deep learning as a robust primary diagnostic tool rather than merely a supplemental one.

Rajpurkar et al. (2018) also investigated the use of deep learning methods for deciphering chest CT scans. Their CheXNeXt model, which can detect up to fourteen abnormalities and is comparable to experienced medical professionals, was developed on a dataset of more than ten thousand images from chest X-rays.

Deep learning's inclusion in healthcare diagnosis holds forth the prospect of enhanced precision and cost effectiveness. Obermeyer and Emanuel (2016) investigated the application of artificial intelligence in healthcare and came to the conclusion that such mathematical instruments, when properly utilized, could substantially decrease errors made by humans, speed up testing processes, and even predict patient routes allowing proactive treatments.

Although the use of deep learning for medical investigations has unquestionably bright futures, there are still significant obstacles to overcome. Data security is still of utmost importance, particularly in light of the delicate characteristics of medical information. Deep learning algorithms' 'black box' character also raises readability problems, raising concerns regarding their incorporation into crucial healthcare assessments (Obermeyer & Emanuel, 2016).
## <a name="_toc147983904"></a>***1.2 What is it about?***
The approach of artificial intelligence known as "deep learning" is distinguished by its capacity to extract information from enormous volumes of data, uncovering relationships and trends that may be opaque to individual specialists. Methods of deep learning, notably CNNs, have shown tremendous promise in the healthcare industry for processing complex medical pictures to identify and diagnose disorders. The goal of this research is to improve the diagnostic process by utilizing deep learning to make it more precise, effective, and quick.
## <a name="_toc147983905"></a>***1.3 Why does it matter?***
A quick and precise assessment could mean what separates death and life in the complex world of healthcare. While successful, conventional diagnostic techniques frequently rely on the knowledge of medical experts and can occasionally be flawed by mistakes made by individuals. Additionally, there is a pressing demand for automated, effective, and exact methods of diagnosis given the expanding amount of health information. With its aptitude for managing huge data sets and generating insightful conclusions, artificial intelligence presents a viable answer to these problems. The process of diagnosis might be revolutionized by using deep learning algorithms, assuring early identification and intervention, which may substantially enhance the health of patients and reduce healthcare expenses.

## <a name="_toc147983906"></a>***1.4 Aim of the Research Project and Definition of Objectives***
1. To improve the precision and accuracy of disease diagnosis from medical imagery. 
1. To reduce the time taken to diagnose diseases, enabling timely interventions. 
1. To study state-of-the-art approaches of AI techniques. 
1. To develop a model that can be fine-tuned or adapted to diagnose a variety of diseases or conditions. 
1. To ensure that the model's predictions can be understood and rationalized by healthcare professionals.
## <a name="_toc147983907"></a>***1.5 Research Question*** 
1. What can be done to improve the precision and effectiveness of illness detection from medical images using deep learning techniques, particularly CNNs?
1. What current issues in medical diagnostics need to be addressed, and how may deep learning help?
1. What steps can we take to make sure that healthcare practitioners can understand and justify the predictions that deep learning models make?
1. How will integrating deep learning techniques affect the whole healthcare system, particularly in terms of patient outcomes and treatment effectiveness?
## <a name="_toc147983908"></a>***1.6 Chapter Summary***
A fresh era of scientific development has begun with the use of deep learning methods, notably CNNs, to investigations in the field of health care. This project intends to improve the health of patients, shorten the process of diagnosis, and add up to a more effective healthcare industry by constructing a prototype capable of quickly and accurately identifying illnesses using medical pictures. Numerous advantages might result from this integration, from better patient care to huge cost reductions for healthcare organizations. The combination of neural training and medical examinations, while the globe keeps embracing AI, is proof of the virtually endless potential of technology advancement in medicine.


## <a name="_toc147983909"></a>***Chapter:2 Data*** 
The accuracy and usefulness of the training data have a considerable impact on the effectiveness of the algorithm, particularly in medical diagnoses. The raw data set used in this study is thoroughly examined in the following section with an emphasis on its source, size, framework, and quality. Understanding the peculiarities of the information set guarantees the framework's construction and assessment stages are tackled with clarity and precision.
## <a name="_toc147983910"></a>***2.1 Data Sources***
Diverse data in the vast area of healthcare imaging make it easier to conduct research, particularly for advanced machine learning applications. Chest X-ray scans have been selected as the specific sample for this investigation. Images from the dataset are divided into two main groups: normal and pneumonia. Given the frequency and importance of pneumonia identification, this set of data is very pertinent to the goals of the study.
## <a name="_toc147983911"></a>***2.2 Data Size***
Despite not being as vast as some other collections, the set of data is adequately comprehensive for our needs. It consists of a group of Ten allocated photos for validation, a set of 250 images for testing, and 814 images for training. This breakdown makes sure that the strategy is evenly distributed, enabling thorough training of models while yet keeping plenty of data for testing and validation.
## <a name="_toc147983912"></a>***2.3 Data Shape***
The data set's arrangement is rather simple. Each photo has a caption stating that it depicts a normal condition or a pneumonia condition because there are two distinct types. As a result, each entry in the collection comprises of an image and its associated label.
## <a name="_toc147983913"></a>***2.4 Time Period***
This dataset contains photos and descriptions that were gathered between the years 2010 and 2020 and span a decade. Due to the wide range of disease trends and developments throughout this long period of time, an adequate sample of patients is ensured.
## <a name="_toc147983914"></a>***2.5 What does each row represent?***
The dataset's entries and rows each refer to a particular chest X-ray picture. There is a label next to the image stating whether it belongs in the "normal" category or the "pneumonia" category.
## <a name="_toc147983915"></a>***2.6 Data Dictionary:***
The data dictionary acts as a guide, elucidating the nature and structure of the dataset. 

- **Columns Name**:
  - Image\_Path
  - Label
- **Data Type**:
  - Image\_Path: String (Path to the image)
  - Label: String (Category of the image: "normal" or "pneumonia")
- **Definition**:
  - Image\_Path: Denotes the location or filename of the X-ray image.
  - Label: Indicates the diagnostic classification of the X-ray image, either as 'normal' or 'pneumonia'.
- **Potential Values**:
  - Image\_Path: Filenames such as "xray\_00123.jpg".
  - Label: Either "normal" or "pneumonia".
## <a name="_toc147983916"></a>***2.7 Target/Label for ML Model:***
Each row and entry in the dataset refers to a specific chest X-ray image. Whether the photograph is in the "normal" category or the "pneumonia" category is indicated by a label next to it.
## ` `***<a name="_toc147983917"></a>2.8 Features/Predictors for ML Models:***
The X-ray pictures, which are referred to by the "Image\_Path" column, will serve as the CNN model's main attribute. The algorithm will examine and interpret the complex features in these photos in order to generate its forecasts.

The Chest X-ray a database, as a whole, provides an extensive and thorough compilation of X-ray pictures, making it the perfect option for this study. Its size and level of detail guarantee that the algorithm used for deep learning has a wealth of data to draw from, increasing the likelihood that it will provide correct diagnosis.
## <a name="_toc147983918"></a>***2.9 Chapter Summary:***
The selected dataset, with its emphasis on pneumonia identification, serves as a reminder of the value of prompt and precise diagnoses in the medical field. Despite being straightforward, its structure gives the model the depth it needs to comprehend and foretell pneumonia instances from chest X-ray pictures. The exploration in this chapter paves the way for the ensuing model-building and assessment processes, which offer intriguing insights that may influence the direction of medical diagnostics in the future.
# <a name="_toc147983919"></a>**Chapter 3: Exploratory Data Analysis (EDA)**
## <a name="_toc147983920"></a>**3.1 Introduction**
`	`Any dataset-driven project must start with exploratory data analysis. To grasp the data's structure, peculiarities, and trends requires a thorough investigation. The need of making sure the information is widely accepted and ready to go has increased with the popularity of imaging in healthcare and its use in AI. This investigation is made easier by the use of tools like Jupyter Notebook, which allows for a collaborative and continuous study. Chest X-ray pictures are used in our work to leverage the potential of deep modeling for medical diagnoses.
## <a name="_toc147983921"></a>**3.2 Data Exploration with Jupyter Notebook**
An free to use web tool called Jupyter Notebook facilitates the development and distribution of files with active coding, problems, visuals, and text. Jupyter Notebook has been used in our research to load and examine the data set. This engaging tool allows for quick representations and calculations, which makes it easier to comprehend the dataset's complexities.

![](report_img_1)![](Aspose.Words.de6df561-5f44-446c-848f-47ba8a9a1346.002.png)

![](Aspose.Words.de6df561-5f44-446c-848f-47ba8a9a1346.003.png)

![](Aspose.Words.de6df561-5f44-446c-848f-47ba8a9a1346.004.png)

## <a name="_toc147983922"></a>**3.3 Focus on Target Variable and Features**
The diagnostic result is the key variable. Each X-ray picture is specifically assigned to one of two groups:

1. **Normal**: This indicates that there were no discernible symptoms of the illness on the X-ray imaging.
1. **Pneumonia**: The X-ray scan showed features and characteristics typical of pneumonia sufferers.

The X-ray picture represents the most important tool we have at our disposal. Each pixel value (in grayscale) in these photographs is represented as a matrix, with values ranging from 0 to 255. CNN algorithm will discover whether variations in these values of pixels, slopes, borders, and sequences are associated with a certain diagnosis..
## <a name="_toc147983923"></a>**3.4 Summary Statistics and Visualization**
It is easier to comprehend the structure, principal tendency, and dispersion of the data when short statistics are produced. Identifying the pattern of distribution of normal vs pneumonia photos, overall image dimensions, pixel density contributions, etc. in the setting of our picture dataset would be necessary. Data may be understood more easily through representations. While strong programs like Matplotlib are available, Plotly Express provides interactive charts that are very useful for our dataset. It would be useful to visualize the intensity of the pixels patterns of a portion of normal and pneumonia photos as well as any other pertinent data.

![](Aspose.Words.de6df561-5f44-446c-848f-47ba8a9a1346.005.png)

![](Aspose.Words.de6df561-5f44-446c-848f-47ba8a9a1346.006.png)

![](Aspose.Words.de6df561-5f44-446c-848f-47ba8a9a1346.007.png)

![](Aspose.Words.de6df561-5f44-446c-848f-47ba8a9a1346.008.png)
## <a name="_toc147983924"></a>**3.6 Data Cleansing , Transformation, Augmenting with External Data, Pre-processing**
It is crucial to guarantee the accuracy of the data. Data cleaning is a crucial stage in this procedure for this investigation. As an example, lacking values might be a critical issue. The statistical accuracy might be substantially hampered by a poorly loaded image or a missing description. Such irregularities must be found and properly fixed. Another possible hazard is the existence of identical columns or photos. Repeated data can induce bias by forcing an automated algorithms to incorrectly emphasize particular patterns, which results in overfitting. Therefore, it is essential to find and remove repetitions in order to retain the dataset's validity.

- **Missing Values:** Checking for any missing descriptions or failure loading images is necessary.
- **Duplicate Rows:** Overfitting might result from duplicate photos that bias the training process. They must be located and eliminated.

The resulting data set gets further processed by Dataset Transformation, which adapts it to match the unique requirements of the ensuing AI projects. For a more detailed study, certain aspects may need to be divided, while others may require integration to provide a more comprehensive view. Though the majority of our dataset consists of picture routes and their labels, in order to increase the model's resilience, various changes such data augmentations may be added to the photos throughout the training stage.

The concept of augmentation with additional information adds a broad perspective to the investigation. Despite being thorough, the X-ray photos and comments only cover a portion of the person's medical record. The size and breadth of the study can be considerably improved by incorporating supplementary source data. The set of images used for the present research concentrates on the X-ray pictures. Adding individual characteristics, prior medical records, or even geography information, nevertheless, can provide extra information in larger circumstances.

Descriptive Information If the dataset includes technical labels or terms, initial processing is a necessary step. Raw data composed of text may be erratic and cluttered. Such anomalies can be removed by normalizing the content, which requires putting it in a regular format. To further simplify the data, inconsequential terms known as stopwords that don't significantly contribute meaning to the text are removed. The issuance of token helps the text's preparedness for modeling by dividing it into separate phrases or characters.

Last but not least, maintaining a "Tidy" Dataset is essential for a quick and successful analysis. A compact set consists of a single in which each row represents a unique observation and each column denotes a special characteristic of that information. For the purposes of this investigation, each row should outline a distinct X-ray picture. Our dataset follows this rule and makes the succeeding steps more seamless with its two columns, "Image\_Path" and "Label". The framework, dispersion, and features of our dataset have now been thoroughly understood thanks to the EDA procedure. These insights will improve modeling and guarantee that the model generated by deep learning is accuracy and reliability.
## <a name="_toc147983925"></a>**3.7 Chapter Summary** 
The Exploratory Data Analysis was covered in detail in this chapter. this study  analyzed and comprehended the dataset's composition using tools like Jupyter Notebook and Plotly Express, concentrating on the distinction between 'Normal' and 'Pneumonia' diagnosis. Data cleansing to remove duplicates and missing information, as well as assuring the dataset's orderliness, were crucial tasks. This thorough investigation prepares our research for an educated and effective modeling step.

.
## <a name="_toc147983926"></a>**Chapter 4:  Model Training**
Accuracy in the field of diagnostic imaging is crucial since it determines whether someone will live or die. In this field, it is essential to guarantee the accuracy of algorithmic learning frameworks. The chapter examines the efficiency of the model after it has been trained to identify illnesses from chest X-ray pictures. Given the consequences of a wrong diagnosis, the decision becomes much more important in the field of diagnostic imaging. 
## <a name="_toc147983927"></a>**4.1 Choice of Model for Predictive Analytics**
Predictive statistics has been assigned to the CNN. Because of its distinctive design, CNNs have demonstrated a notable level of success in image classification challenges. They are especially appropriate for our dataset of chest X-rays since they are skilled at establishing ordered spaces of characteristics effortlessly and continually from pictures. CNNs' hierarchical structure gives its algorithm the ability to distinguish between many aspects of a picture, such as boundaries, surfaces, and more complicated designs, which improves its ability to diagnose problems.
## <a name="_toc147983928"></a>**4.2 Model Training Approach**
Labeled data are fed into a CNN during training. Every X-ray picture in this situation is sent over the network and assigned both the label "Normal" or "Pneumonia." The disparity among the prediction of the model and the real label—also known as the error—is calculated. The inbuilt settings are then adjusted as a result of the mistake being back-propagated via the network. Up until the model's forecasts are adequately accurate, this procedure is performed repeatedly (iterations).

![](Aspose.Words.de6df561-5f44-446c-848f-47ba8a9a1346.009.png)
## <a name="_toc147983929"></a>**4.3 Data Splitting Strategy**
There are training and testing sets for the set of data. It has been decided to employ an eighty percent to twenty split, where eighty percent of the data is used for training and twenty percent is set aside for testing. Through this division, the model is given enough data for learning from while simultaneously maintaining an objective sample of data to assess its effectiveness.
## <a name="_toc147983930"></a>**4.4 Python Packages and Libraries**
Various Python packages bolster the model training and evaluation process:

- **TensorFlow**: ML models could be easily created and trained using this an open-source project platform. Keras, its excellent API, makes it easier to define, train, and assess deep computational models.
- **scikit-learn**: It is a well-known algorithms package that provides tools for dividing datasets, assessing simulations, and other tasks.
- **OpenCV**: By guaranteeing the X-ray pictures are properly transformed before being fed into the computer model, this library aids in image processing jobs.

**4.5 Development Environment**

The selection of a training environment gets crucial considering the computing requirements of deep learning models, particularly CNNs. Personal laptops  perform simple calculations and beginning data exploration, but the hard process of model training requires more stable conditions. For this project, a powerful laptop with a graphical user interface stands out as the best option. Its easy compatibility with Jupyter notebooks adds to its allure.

**4.6 Performance Metrics and Comparison**

To gauge the model's efficiency, several metrics are employed:

- Accuracy: It measures the percentage of correct predictions out of all predictions made.
- Confusion Matrix: A table used to describe the performance of a classification model on a set of data for which the true values are known.
- Classification Report: It provides a detailed breakdown of precision, recall, and F1-score for each class.

![](Aspose.Words.de6df561-5f44-446c-848f-47ba8a9a1346.010.png)

![](Aspose.Words.de6df561-5f44-446c-848f-47ba8a9a1346.011.png)

![](Aspose.Words.de6df561-5f44-446c-848f-47ba8a9a1346.012.png)

![](Aspose.Words.de6df561-5f44-446c-848f-47ba8a9a1346.013.png)

Possible overfitting is revealed by contrasting the algorithms efficiency on the training set with the test set. The algorithm may be over-fitting if it works very well on the data used as training but badly on the data being tested. On the other hand, if it does inadequately in both, it may be underfitting and require architectural changes.

**4.7 Chapter Summary**

A thorough strategy is essential when training an artificial intelligence learning framework, particularly for important jobs like medical diagnosis. Each stage is essential to ensure the model's dependability and accuracy, from choosing the proper model and setting to adopting suitable metrics for assessment. The complexity of these procedures was clarified in this chapter, setting the foundation for the real-world execution of the code and subsequent diagnostics.

## <a name="_toc147983931"></a>**Chapter 5 *Conclusion***
## <a name="_toc147983932"></a>***5.1 Summary and Potential Applications***
In this study, we set out on an attempt to use convolutional neural networks , a subset of methods from deep learning, to improve medical diagnosis using chest X-ray pictures. This work has demonstrated that it is feasible to discern between healthy and pneumonia-affected X-ray pictures with an exceptional level of precision. The applications of this study are numerous and profound.

**Medical Diagnostics**: Radiologists can use this framework as an additional diagnostic tool to assist them validate their medical conclusions or to identify prospective instances they may have missed.

**Telemedicine:** Our methodology can fill the gap in rural areas where specialist health care might not be easily accessible, providing prompt and precise diagnosis.

**Screening:** Rapid patient screening becomes crucial in pandemic or epidemic situations. Our algorithm may be used to swiftly recognize and rank people in need of urgent treatment.
## <a name="_toc147983933"></a>***5.2 Limitations***
This study is not without its limitations like anyother research:

**Dataset Diversity:** On a particular collection of X-ray pictures, our model was trained. When exposed to pictures from various machines, different demographics, or other lung conditions, the performance can change.

**False Positives and Negatives:** Although the model has shown remarkable accuracy, there is still a chance of incorrect diagnoses, which can have serious medical repercussions.

**Generalization:** Large volumes of data are necessary for deep learning models, particularly CNNs, to generalize effectively. Despite being large, our dataset might not fully reflect all the subtleties and unusual occurrences of the condition.
## <a name="_toc147983934"></a>***5.3 Lessons Learned***
Throughout this research:

- We have emphasized the importance of data quality and preprocessing in determining how well deep learning models function.
- The significance of model evaluation to receive a whole picture of the model's performance, including precision, recall, and F1 score in addition to measures like accuracy.
- Deep learning's promise and skills to revolutionize medical diagnosis

## <a name="_toc147983935"></a>***5.4 Future Research Directions***
The groundwork established by this study opens up a number of possibilities for further investigation:

- **Incorporating More Diseases**: Making the model more adaptable as a diagnostic tool by extending it to recognize more respiratory conditions from X-ray pictures.
- **Transfer Learning**: utilizing trained models already, improving the precision and speed of our diagnostic instrument.
- **Model Interpretability**: Even if our model is capable of making predictions, knowing the reasons behind its choices might increase confidence in its prognoses. Future studies can examine methods for improving the interpretability of neural networks.
- **Real-world Application**: integrating our approach with hospital systems or telemedicine platforms to get from study to practical implementation.
# <a name="_toc147983936"></a>**References** 

Suzuki, K., 2017. Overview of deep learning in medical imaging. *Radiological physics and technology*, *10*(3), pp.257-273.

Lee, J.G., Jun, S., Cho, Y.W., Lee, H., Kim, G.B., Seo, J.B. and Kim, N., 2017. Deep learning in medical imaging: general overview. *Korean journal of radiology*, *18*(4), pp.570-584.

Vaz, J.M. and Balaji, S., 2021. Convolutional neural networks (CNNs): Concepts and applications in pharmacogenomics. *Molecular diversity*, *25*(3), pp.1569-1584.

Gu, R., Wang, G., Song, T., Huang, R., Aertsen, M., Deprest, J., Ourselin, S., Vercauteren, T. and Zhang, S., 2020. CA-Net: Comprehensive attention convolutional neural networks for explainable medical image segmentation. IEEE transactions on medical imaging, 40(2), pp.699-711.

Esteva, A., Kuprel, B., Novoa, R.A., Ko, J., Swetter, S.M., Blau, H.M. and Thrun, S., 2017. Dermatologist-level classification of skin cancer with deep neural networks. nature, 542(7639), pp.115-118

Rajpurkar, P., Irvin, J., Ball, R.L., Zhu, K., Yang, B., Mehta, H., Duan, T., Ding, D., Bagul, A., Langlotz, C.P. and Patel, B.N., 2018. Deep learning for chest radiograph diagnosis: A retrospective comparison of the CheXNeXt algorithm to practicing radiologists. PLoS medicine, 15(11), p.e1002686.

Obermeyer, Z. and Emanuel, E.J., 2016. Predicting the future—big data, machine learning, and clinical medicine. The New England journal of medicine, 375(13), p.1216.
