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
