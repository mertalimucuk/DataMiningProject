Tarım Verileri Üzerine Makine Öğrenmesi Analizi
Proje Tanıtımı:
Bu proje, tarım verilerini kullanarak ürün verimini (GrainYield) tahmin etmeyi amaçlayan kapsamlı bir veri madenciliği çalışmasıdır. Proje, ham verinin temizlenmesinden, çeşitli makine öğrenmesi modellerinin karşılaştırılmasına ve en iyi performans gösteren modelin belirlenmesine kadar uzanan bir süreci kapsamaktadır.

Veri Seti:
Veri Kaynağı: Projede, tarımsal verileri içeren bir XLSX dosyası kullanılmıştır.

Veri Temizleme: Veri setindeki gürültülü (noisy) ve boş veriler temizlenerek, sonraki analizler için sağlam bir temel oluşturulmuştur. Bu temizleme işlemleri, Data_Cleaning.ipynb dosyasında detaylandırılmıştır.

Uygulanan Yöntemler
Bu projede, veri analizi ve makine öğrenmesi alanında birçok önemli adım atılmıştır:

Veri Ön İşleme:

Veri setindeki gürültülü değerler ve eksik (null) veriler belirlenip temizlenmiştir.

Veri, model eğitimine hazır hale getirilmiştir.

Sınıf dengesizliğini gidermek için SMOTE (Synthetic Minority Over-sampling Technique) tekniği uygulanmıştır.

Model Geliştirme ve Karşılaştırma:

Projede, lojistik regresyon, karar ağaçları, Random Forest, destek vektör makineleri (SVM) ve yapay sinir ağları (MLP) gibi birçok farklı makine öğrenmesi algoritması kullanılmıştır.

Bu modellerin performansı ROC eğrileri, doğruluk (accuracy), kesinlik (precision), geri çağırma (recall), F1 puanı ve MCC gibi çeşitli metrikler kullanılarak değerlendirilmiştir.

Çapraz Doğrulama (Cross-Validation):

Modellerin genelleştirme yeteneğini değerlendirmek için 5 Katlı Stratified K-Fold çapraz doğrulama yöntemi uygulanmıştır.

Proje Dosyaları
Data_Cleaning.ipynb: Veri temizleme ve ön işleme adımlarının bulunduğu Jupyter Notebook dosyası.

Data_processed_cleaned.xlsx: Temizlenmiş ve işlenmeye hazır hale getirilmiş veri seti.

Feature_selection.ipynb: Özellik seçimi ve makine öğrenmesi modellerinin karşılaştırılmasının yapıldığı ana dosya. Bu dosya, farklı algoritmaların performansını ve ROC eğrilerini görselleştirir.

Image_Results/: Çıkan sonuçların (ROC Eğrileri, Confusion Matrix) görselleştirildiği görüntüler bu dizinde yer almaktadır.

Sonuçlar
Analizler sonucunda en yüksek performansı gösteren algoritma belirlenmiş ve bu modelin doğruluk matrisi (confusion matrix) ile performans grafikleri oluşturulmuştur. Bu sonuçlar, veri setinin tarımsal verim tahmini için başarılı bir şekilde kullanılabileceğini göstermektedir.

[ENG]

Machine Learning Analysis on Agricultural Data
Project Introduction
This project is a comprehensive data mining study aimed at predicting crop yield (GrainYield) using agricultural data. The process covers everything from cleaning raw data to comparing various machine learning models and identifying the best-performing one.

Dataset
Data Source: The project utilizes an XLSX file containing agricultural data.

Data Cleaning: Noisy values and missing (null) data in the dataset were cleaned to create a solid foundation for subsequent analyses. These cleaning operations are detailed in the Data_Cleaning.ipynb file.

Methodology
Several key steps in data analysis and machine learning were taken in this project:

Data Preprocessing
Noisy values and missing (null) data were identified and handled.

The data was prepared for model training.

SMOTE (Synthetic Minority Over-sampling Technique) was applied to address class imbalance.

Model Development and Comparison
The project used multiple machine learning algorithms, including Logistic Regression, Decision Trees, Random Forest, Support Vector Machines (SVM), and Multi-Layer Perceptrons (MLP).

Model performance was evaluated using various metrics such as ROC curves, accuracy, precision, recall, F1 score, and MCC.

Cross-Validation
5-Fold Stratified K-Fold cross-validation was used to assess the models' generalization ability.

Project Files
Data_Cleaning.ipynb: A Jupyter Notebook file containing the data cleaning and preprocessing steps.

Data_processed_cleaned.xlsx: The cleaned and processed dataset, ready for analysis.

Feature_selection.ipynb: The main file for feature selection and machine learning model comparison. This notebook also visualizes model performance and ROC curves.

Image_Results/: This directory contains images of the output, such as ROC Curves and the Confusion Matrix.

Results
The analysis identified the highest-performing algorithm. The confusion matrix and performance graphs for this model were created to demonstrate that the dataset can be successfully used for agricultural yield prediction.
