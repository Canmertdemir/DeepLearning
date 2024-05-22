# Zaman Serileri Özelinde Derin Öğrenme Modellerinin Karşılaştırılması

Bu proje, Individual Household Electric Power Consumption veri kümesi üzerinde farklı derin öğrenme modellerinin eğitimini ve sonrasında bu modelin kullanımını içerir.

1. **Gereksinimler**
   - Python 3.7
   - pyTorch
   - NumPy
   - Matplotlib
   - Scikit-learn
2. **Kurulum**

   Gereksinimleri yüklemek için aşağıdaki komutları kullanın:

   - conda create -n pytorch python=3.7
   - conda activate pytorch
   - conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
   - conda install -c anaconda cudatoolkit(caffe2_nvrtc.dll not found )
   - conda install python=3.7.4
   - conda install scikit-learn

*Test*
import torch
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


3. Veri Kümesi ve İşlenmesi

   Individual Household Electric Power Consumption veri kümesi https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption linkinden indirilerek yüklenir.

   Veri setinin orijinal halinde tüm değişkenler obje olarak tutulmuştur. Objeler ile regresyon işlemi yapılamayacağı için bu değişkenler tam sayılara ve ondalık sayılara dönüştürülmüştür.
Date ve Time verileri iki ayrı değişkende tutulmuştur. Bu değişkenler birleştirilerek DateTime olarak bir yeni değişken elde edilmiştir ve indeks olarak atama yapılmıştır.
Makine öğrenmesi modellerinin konvolisyon gibi bir özelliği olmadığı için özellik mühendisliği kullanılarak Date ve Time değişkenlerinin içinden yıl, ay, gün, saat, dakika gibi anlamlı özellikler ayıklanarak modele verilmiştir.
Derin öğrenme modellerinde bazı modellerde özellik çıkarımı kullanılmıştır, CNN gibi konvolüsyon ile özellik çıkarımı yapan mimarilerde bu işlemden kaçınılmıştır.
Sayısal değişkenlerden oluşan bu veri setinde herhangi bir encoder kullanılmamıştır.

5. Eğitim

   Modeli eğitmek için klasörlerin altında bulunan 'model_ismi.py' dosyalarını kullanınız.

6. Bulgular
  - Zaman serilerinde makine öğrenmesi modelleri Derin öğrenme modelleri kadar iyi sonuç vermemektedir.
  - Makine öğrenmesi modellerinden LightGbm ve XGBoost en iyi çalışan zaman serisi modelleri olmasına rağmen iyi sonuç vermemiştir.
  - Tam bağlı sinir ağı regresyon modeli derin öğrenme modelleri arasında en az performans gösteren modeldir.
  - Bu veri seti özelinde RNN ile LSTM yaklaşık olarak aynı performansı göstermiştir.
  - LSTM ve CNN zaman serisi modelleri en iyi performans veren modellerdir.
  - CNN, LSTM’e göre daha iyi sonuç vermiştir. 
  - En iyi performans veren model CNN Zaman Serisi Regresyon modelidir.   


7.Tavsiyeler

1. **Makine Öğrenmesi Modelleri (LightGBM ve XGBoost):**
   - LightGBM ve XGBoost gibi makine öğrenmesi modelleri, genel amaçlı regresyon ve sınıflandırma problemlerinde oldukça başarılıdır, ancak zaman serisi verilerinde yapısal bağımlılıkları ve trendleri yakalamada yetersiz kalabilirler. Zaman serisi verileri için özelleştirilmiş parametreler ve özellik mühendisliği (feature engineering) uygulamak, performanslarını artırabilir.

2. **Tam Bağlı Sinir Ağı (Fully Connected Neural Network - FCNN):**
   - FCNN'ler, zaman serisi verilerinde genellikle iyi performans göstermeyebilir çünkü zaman serisi verilerinde ardışıklık (sequence) ve bağımlılıklar önemlidir. Bu tür bağımlılıkları yakalamak için genellikle RNN veya LSTM gibi modeller tercih edilir.

3. **RNN ve LSTM:**
   - RNN (Recurrent Neural Network) ve LSTM (Long Short-Term Memory) modelleri, zaman serisi verilerinde ardışık bağımlılıkları yakalamada başarılıdır. Bu iki modelin benzer performans göstermesi, veri setinin özelliklerinden kaynaklanabilir. RNN'ler kısa süreli bağımlılıkları öğrenmede iyi iken, LSTM'ler uzun süreli bağımlılıkları öğrenmede avantajlıdır.

4. **LSTM ve CNN:**
   - LSTM ve CNN (Convolutional Neural Network) modelleri, zaman serisi verilerinde öne çıkmaktadır. LSTM'ler zaman içindeki bağımlılıkları ve trendleri öğrenmede iyidir. CNN'ler ise zaman serisi verilerindeki lokal kalıpları ve özellikleri yakalamada başarılıdır. 
   - CNN'in LSTM'e göre daha iyi sonuç vermesi, veri setinde lokal kalıpların daha belirgin olabileceğini gösterir. Zaman serisi verilerinde genellikle birden fazla dönemsel kalıp (seasonal pattern) veya trend bulunabilir ve CNN bu kalıpları daha etkili şekilde yakalayabilir.

5. **En İyi Performans Veren Model: CNN Zaman Serisi Regresyon Modeli:**
   - CNN modelinin en iyi performansı göstermesi, zaman serisi verisindeki lokal özelliklerin önemini vurgular. CNN'ler zaman serisi verilerinde konvolüsyon katmanları sayesinde lokal özellikleri ve kalıpları iyi yakalayabilir.

### Tavsiyeler:
1. **Model Seçimi:**
   - Zaman serisi problemlerinde, klasik makine öğrenmesi modelleri yerine, verideki ardışık bağımlılıkları ve trendleri daha iyi yakalayan derin öğrenme modellerini (özellikle LSTM ve CNN) tercih edin.

2. **Özellik Mühendisliği:**
   - Zaman serisi verilerinde ek özellikler (feature engineering) oluşturmak, modellerin performansını artırabilir. Örneğin, mevsimsellik (seasonality), trend, ve lag özellikleri oluşturulabilir.

3. **Model Kombinasyonları:**
   - CNN ve LSTM gibi modelleri birleştirerek (örneğin, CNN-LSTM hibrit modeller) daha iyi performans elde edebilirsiniz. CNN'in lokal kalıpları yakalama yeteneği ile LSTM'in uzun vadeli bağımlılıkları öğrenme kapasitesi bir arada kullanılabilir.

4. **Hiperparametre Optimizasyonu:**
   - Modellerin hiperparametrelerini optimize ederek (örneğin, öğrenme hızı, katman sayısı, nöron sayısı, vb.), model performansını daha da iyileştirebilirsiniz.

5. **Model Değerlendirme:**
   - Modelleri değerlendirirken, zaman serisi verilerinin doğasına uygun metrikler kullanın. Örneğin, Mean Absolute Error (MAE), Mean Squared Error (MSE) gibi metrikler yanı sıra, zaman serisi için özel metrikler (örneğin, Mean Absolute Percentage Error - MAPE) de kullanılabilir.

Bu tavsiyeler doğrultusunda zaman serisi veri setinizde daha iyi performans elde edebilirsiniz.

### Sonuçlar:
### Makine Öğrenmesi Modelleri:
1. **Model Çeşitliliği ve Birleştirme:**
   - Daha fazla makine öğrenmesi modeli kullanarak ve bu modelleri bir araya getirerek (ensembling) model performansını artırabilirsiniz. Örneğin, Random Forest, Gradient Boosting, CatBoost gibi modelleri deneyebilirsiniz.
   - Ensemblling teknikleri olarak bagging, boosting veya stacking yöntemlerini kullanabilirsiniz.

2. **Özellik Mühendisliği:**
   - Hareketli ortalama (moving average) ve pencere oluşturma (windowing) gibi teknikler ile zaman serisi verilerinden yeni özellikler oluşturabilirsiniz.
   - Lag özellikleri, mevsimsellik özellikleri, ve trend analizleri yaparak modele ek veri sağlanabilir.
   - Zaman serisi verilerinde outlier (aykırı değer) tespiti ve giderilmesi de performansı artırabilir.

### Derin Öğrenme Modelleri:
1. **Tam Bağlı Sinir Ağı (FCNN):**
   - Hiperparametre optimizasyonunu daha derinlemesine yaparak performansı artırabilirsiniz. Örneğin, öğrenme hızı, optimizasyon algoritması, katman sayısı, nöron sayısı gibi parametreler optimize edilebilir.
   - Dropout ve batch normalization gibi tekniklerle modelin aşırı öğrenmesi (overfitting) engellenebilir.

2. **RNN (Recurrent Neural Network):**
   - RNN modeline daha fazla katman ekleyerek ve nöron sayısını artırarak modelin kapasitesini artırabilirsiniz.
   - Özellik mühendisliği ile RNN modeline daha fazla bilgi(Fourier dönüşümleri vb.) sağlanarak performans iyileştirilebilir.
   - Sequence length, batch size, ve öğrenme hızı gibi hiperparametrelerin optimizasyonu yapılmalıdır.

3. **LSTM (Long Short-Term Memory):**
   - LSTM katman sayısını ve nöron sayısını artırarak modelin performansını artırabilirsiniz. Ancak, daha fazla katman eklerken modelin aşırı öğrenmemesine dikkat edilmelidir.
   - LSTM'nin dikkat (attention) mekanizmaları ile kombinasyonu performansı daha da artırabilir.

4. **CNN (Convolutional Neural Network):**
   - CNN modeline daha fazla katman ekleyerek lokal kalıpları daha iyi yakalayabilirsiniz. Ancak, katman sayısını artırırken aşırı öğrenme (overfitting) riskine dikkat edilmelidir.
   - Convolution ve pooling katmanlarının sayısı ve boyutları optimize edilerek performans iyileştirilebilir.
   - Özellikle zaman serisi verilerinde 1D Convolution katmanları sayısı arttırılabilir.
   - Bu çalışma özelinde low-level özellik türeten 1d Convolution bir katman kullanılmıştır.

### Genel Tavsiyeler:
1. **Model Kombinasyonu:**
   - CNN ve LSTM gibi modelleri birleştirerek (örneğin, CNN-LSTM hibrit modeller) daha iyi performans elde edebilirsiniz. Bu sayede hem lokal kalıplar hem de uzun vadeli bağımlılıklar yakalanabilir.

2. **Veri Ön İşleme:**
   - Zaman serisi verilerinde veri ön işleme adımları çok önemlidir. Veriyi normalize etmek, outlier değerleri tespit edip temizlemek ve eksik verileri doldurmak performansı ciddi şekilde etkileyebilir.
   - Verideki trendleri ve mevsimsellikleri belirleyip modele bu bilgileri sağlamak da önemlidir.

3. **Model Değerlendirme:**
   - Modelleri değerlendirirken zaman serisi verilerinin doğasına uygun metrikler kullanın. Örneğin, Mean Absolute Error (MAE), Mean Squared Error (MSE) gibi metrikler yanı sıra, Mean Absolute Percentage Error (MAPE) gibi metrikler kullanılabilir.
   - Zaman serisi verilerinde cross-validation yaparken, verinin ardışık yapısını bozmamak için zaman temelli cross-validation yöntemlerini kullabilir.

### Referanslar:   
   https://coderzcolumn.com/tutorials/data-science/how-to-remove-trend-and-seasonality-from-time-series-data-using-python-pandas
   https://tr.d2l.ai/chapter_convolutional-neural-networks/index.html
   https://coderzcolumn.com/tutorials/artificial-intelligence/pytorch-lstm-networks-for-time-series-regression-tasks
   https://www.geeksforgeeks.org/ensemble-methods-in-python/
   https://machinelearningmastery.com/building-a-regression-model-in-pytorch/
   https://machinelearningmastery.com/building-a-regression-model-in-pytorch/
