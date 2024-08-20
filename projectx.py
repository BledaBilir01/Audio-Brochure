import cv2 #Opencv kutuphanesi cagırma
import torch #PyTorch kutuphanesi cagırma
import pygame  #Playsound kutuphanesi cagırma
import time #Time kutuphanesi cagırma
import os#Os kutuphanesi cagırma

# Mevcut dizini al
current_directory = os.path.dirname(os.path.abspath(__file__))

karsilama_sound_path = os.path.join(current_directory, "sesdosyalari", "hosgeldiniz.wav")#Uygulamayi calistirirken calan sesin yolu
tarama_sound_path = os.path.join(current_directory, "sesdosyalari", "taramatamam.wav")#Tarama yaparken calan sesin yolu
bekunis_sound_path = os.path.join(current_directory, "sesdosyalari", "sesbekunis.wav")#"Bekunis" isimli ilacin ozelliklerini barindiran sesin yolu
arveles_sound_path = os.path.join(current_directory, "sesdosyalari", "sesarveles.wav")#"Arveles" isimli ilacin ozelliklerini barindiran sesin yolu
flessi_sound_path = os.path.join(current_directory, "sesdosyalari", "sesflessi.wav")#"Flessi" isimli ilacin ozelliklerini barindiran sesin yolu
crebros_sound_path = os.path.join(current_directory, "sesdosyalari", "sescrebros.wav")#"Crebros" isimli ilacin ozelliklerini barindiran sesin yolu
diabacore_sound_path = os.path.join(current_directory, "sesdosyalari", "sesdiaba.wav")#"Diabacore" isimli ilacin ozelliklerini barindiran sesin yolu
bitis_sound_path = os.path.join(current_directory, "sesdosyalari", "bitis.wav")#Uygulamayi bitirirken calan sesin yolu

def play_diabacore_sound():#"Diabacore" isimli ilacın ozelliklerini barindiran ses ve sesi calıstirmak icin gerekli fonksiyon
    pygame.mixer.music.load(diabacore_sound_path)
    pygame.mixer.music.play()

def play_karsilama_sound():#Uygulamayi calistirirken calan ses ve sesi calistirmak icin gerekli fonksiyon
    pygame.mixer.music.load(karsilama_sound_path)
    pygame.mixer.music.play()

def play_tarama_sound():#Tarama yaparken calan ses ve ses ve sesi calistirmak icin gerekli fonksiyon
    pygame.mixer.music.load(tarama_sound_path)
    pygame.mixer.music.play()

def play_bekunis_sound():#"Bekunis" isimli ilacin ozelliklerini barindiran ses ve sesi calistirmak icin gerekli fonksiyon
    pygame.mixer.music.load(bekunis_sound_path)
    pygame.mixer.music.play()

def play_arveles_sound():#"Arveles" isimli ilacin ozelliklerini barindiran ses ve sesi calistirmak icin gerekli fonksiyon
    pygame.mixer.music.load(arveles_sound_path)
    pygame.mixer.music.play()

def play_flessi_sound():#"Flessi" isimli ilacin ozelliklerini barindiran ses ve sesi calistirmak icin gerekli fonksiyon
    pygame.mixer.music.load(flessi_sound_path)
    pygame.mixer.music.play()

def play_crebros_sound():#"Crebros" isimli ilacin ozelliklerini barindiran ses ve sesi calistirmak icin gerekli fonksiyon
    pygame.mixer.music.load(crebros_sound_path)
    pygame.mixer.music.play()

def play_bitis_sound():#Uygulamayi kapatirken calan ses ve sesi calistirmak için gerekli fonksiyon
    pygame.mixer.music.load(bitis_sound_path)
    pygame.mixer.music.play()

pygame.mixer.init()# Pygame ses sistemini başlat
    
play_karsilama_sound()#"hosgeldiniz" isimli sesi calistir

path = os.path.join(current_directory, "veriseti_ve_goruntuler", "best.pt")#Google Colab ile egitilen ve Goruntu isleme için gerekli model dosyasinin bulundugu yol

model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True) # Burada egittigimiz modeli cagirmak ve kullanmak  için gerekli fonksiyonlar var

cap = cv2.VideoCapture(0)# Kameramizi aktif etmek icin kullanilir

object_detected = False  #Nesne yakalamayi kapali yapar
start_time = 0  #Uygulama calisirken sureyi sıfırlamaya yarar

while True:
    ret, frame = cap.read()#Kameradan kare okumak icin
    frame = cv2.resize(frame, (1020, 500))#Kameradan gelen goruntuyu istenilen boyutta gosterir
    
    results = model(frame)#Nesnenin hangisi oldugu konusunda tahminleme yapar
    detections = results.xyxy[0]#Kare icerisine alinan nesnenin koordinatlarini verir (x_min, y_min, x_max, y_max)

    for det in detections:
        class_label = int(det[5])#Sınıf ismi icin
        confidence = float(det[4])#Guven skoru

        if confidence > 0.5:#Eger guven skoru 0.5 uzerindeyse islemi yapmak icin
            class_name = model.names[class_label]#model isimlerini cekmek icin
            x, y, w, h = int(det[0]), int(det[1]), int(det[2] - det[0]), int(det[3] - det[1])#Nesne tespiti yapildiktan sonra kare icerisine alinması icin gerekli kisim
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)#Karenin rengini ve boyutunu ayarlar
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)#Nesnenin adının rengini ve boyutunu ayarlar

            if not object_detected:#Eger tespit edilen herhangi bir obje yoksa
                object_detected = True#Nesne yakalamayi acik yapar
                start_time = time.time()#Sureyi baslatir

        else:#varsa eger
            object_detected = False #Nesne yakalamayi kapali yapar
    
    if object_detected and time.time() - start_time > 3:#Eger tespit edilen bir veya daha fazla obje varsa
                if class_name.lower() == "bekunis":#Eger tespit edilen nesnenin ismi "bekunis" ise
                    object_detected = False#Nesne yakalamayi kapali yapar
                    play_tarama_sound()#"tarama" isimli sesi calistir
                    time.sleep(3)#3 saniye bekle
                    play_bekunis_sound()#"bekunis" isimli sesi calistir
                    
                elif class_name.lower() == "arveles":
                    object_detected = False
                    play_tarama_sound()#"tarama" isimli sesi calistir
                    time.sleep(3)#3 saniye bekle
                    play_arveles_sound()#"arveles" isimli sesi calistir
                    
                elif class_name.lower() == "flessi":
                    object_detected = False#Nesne yakalamayi kapali yapar
                    play_tarama_sound()#"tarama" isimli sesi calistir
                    time.sleep(3)#3 saniye bekle
                    play_flessi_sound()#"flessi" isimli sesi calistir
                    
                elif class_name.lower() == "crebros":
                    object_detected = False#Nesne yakalamayi kapali yapar
                    play_tarama_sound()#"tarama" isimli sesi calistir
                    time.sleep(3)#3 saniye bekle
                    play_crebros_sound()#"crebros" isimli sesi calistir
                    
                elif class_name.lower() =="diabacore":
                    object_detected = False#Nesne yakalamayi kapali yapar
                    play_tarama_sound()#"tarama" isimli sesi calistir
                    time.sleep(3)#3 saniye bekle
                    play_diabacore_sound()#"diabacore" isimli sesi calistir                 
                                        
    cv2.imshow("projectx", frame)#Kameradan alınan goruntuyu yansitmak icin
    
    if cv2.waitKey(1) & 0xFF == 27:#eger "esc" butonuna basılırsa
        play_bitis_sound()#"bitis" isimli sesi calistir
        time.sleep(4)
        break

cap.release()#Goruntuyu serbest bırakmak icin
cv2.destroyAllWindows()#tum pencereleri kapatmak icin


