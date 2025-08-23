from base64 import b64encode
from base64 import b64decode
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

def encryption(stringData, stringKey):

    #merubah key dan data dari string kedalam bentuk byte
    key = stringKey.encode("utf-8")
    data = stringData.encode("utf-8")
    
    cipher = AES.new(key, AES.MODE_GCM)                 #membuat object cipher untuk encripsi
    ciphertext, tag = cipher.encrypt_and_digest(data)   #mengenskripsi data
    nonce = cipher.nonce                                 #mengenerate iv
    
    #mengembalikan nilai enskripsi dalam bentuk string dengan format base64
    return b64encode(ciphertext+tag).decode('utf-8')+":"+b64encode(nonce).decode('utf-8')
    
def decryption(stringData, stringKey):
    
    #membagi data dengan iv
    txt = stringData.split(":")
    
    #merubah data, iv, dan key dari string format base64 ke byte
    data = b64decode(txt[0])
    iv = b64decode(txt[1])
    key = stringKey.encode("utf-8")
    
    #membagi text dengan tag
    text  = data[:-16]
    tag = data[-16:] 
    
    cipher = AES.new(key, AES.MODE_GCM, nonce=iv)   #membuat object cipher
    textasli = cipher.decrypt_and_verify(text, tag) #mendeskripsi data
    
    #mengembalikan nilai dekripsi dalam bentuk string
    return textasli.decode("utf-8")


test1 = encryption("moderate,Neg.,Normal,Trace,6.0,Hemolyzed,1.020,Neg.,Neg.,Neg.","opadfahadfladfaj")
print(test1)
print(decryption(test1,"opadfahadfladfaj"))
test1 = encryption("halo","opadfahadfladfaj")
print(test1)
print(decryption(test1,"opadfahadfladfaj"))