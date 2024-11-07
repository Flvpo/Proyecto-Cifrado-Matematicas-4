import pyaudio
import wave
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

# 1. Captura de Audio
def grabar_audio(duracion=5, archivo="voz.wav", tasa_muestreo=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=tasa_muestreo, input=True, frames_per_buffer=1024)
    
    frames = []
    print("Grabando...")
    for _ in range(0, int(tasa_muestreo / 1024 * duracion)):
        data = stream.read(1024)
        frames.append(data)
    print("Grabación completada.")
    
    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(archivo, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(tasa_muestreo)
        wf.writeframes(b''.join(frames))

# 2. Procesamiento de Voz y Aplicación de FFT
def procesar_voz(archivo="voz.wav"):
    tasa_muestreo, data = wavfile.read(archivo)
    
    # Normalización de la señal
    data = data / np.max(np.abs(data))
    
    # Aplicar FFT
    espectro = fft(data)
    frecuencias = np.abs(espectro)[:len(espectro)//2]  # Usar solo la mitad del espectro (simétrico)
    return frecuencias

# 3. Derivación de la Clave
def derivar_clave(frecuencias, tamano_clave=32):
    datos_frecuencia = ''.join(map(str, frecuencias[:tamano_clave]))
    clave = hashlib.sha256(datos_frecuencia.encode()).digest()
    return clave

# 4. Cifrado del Archivo
def cifrar_archivo(nombre_archivo, clave, archivo_cifrado="archivo_cifrado.enc"):
    with open(nombre_archivo, "rb") as f:
        datos = f.read()
    
    cipher = AES.new(clave, AES.MODE_CBC)
    datos_cifrados = cipher.encrypt(pad(datos, AES.block_size))
    
    with open(archivo_cifrado, "wb") as f:
        f.write(cipher.iv)  # Escribir IV al inicio
        f.write(datos_cifrados)
    print("Archivo cifrado guardado como:", archivo_cifrado)

# 5. Descifrado del Archivo
def descifrar_archivo(archivo_cifrado, clave, archivo_descifrado="archivo_descifrado"):
    with open(archivo_cifrado, "rb") as f:
        iv = f.read(16)  # Lee IV de 16 bytes
        datos_cifrados = f.read()
    
    cipher = AES.new(clave, AES.MODE_CBC, iv=iv)
    datos_descifrados = unpad(cipher.decrypt(datos_cifrados), AES.block_size)
    
    with open(archivo_descifrado, "wb") as f:
        f.write(datos_descifrados)
    print("Archivo descifrado guardado como:", archivo_descifrado)

# Ejecución Completa
# 1. Grabar audio
grabar_audio(duracion=5)

# 2. Procesar la señal de voz y aplicar FFT
frecuencias = procesar_voz()

# 3. Derivar clave de las frecuencias
clave = derivar_clave(frecuencias)

# 4. Cifrar un archivo con la clave generada
nombre_archivo = "archivo_para_cifrar.txt"  # Asegúrate de tener este archivo
cifrar_archivo(nombre_archivo, clave)

# 5. Descifrar el archivo con la clave derivada de la voz
descifrar_archivo("archivo_cifrado.enc", clave)
