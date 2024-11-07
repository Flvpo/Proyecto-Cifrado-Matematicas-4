import matplotlib
import matplotlib.pyplot as plt  # Asegúrate de importar pyplot
import numpy as np
import pyaudio as pa 
import struct 

matplotlib.use('TkAgg')

FRAMES = 10248  # Tamaño del paquete a procesar
FORMAT = pa.paInt16  # Formato de lectura INT 16 bits
CHANNELS = 1
Fs = 44100  # Frecuencia de muestreo típica para audio

p = pa.PyAudio()

stream = p.open(  # Abrimos el canal de audio con los parámetros de configuración
    format=FORMAT,
    channels=CHANNELS,
    rate=Fs,
    input=True,
    output=True,
    frames_per_buffer=FRAMES
)

# Creamos una gráfica con 2 subplots y configuramos los ejes
fig, (ax, ax1) = plt.subplots(2)

x_audio = np.arange(0, FRAMES, 1)
x_fft = np.linspace(0, Fs, FRAMES)

line, = ax.plot(x_audio, np.random.rand(FRAMES), 'r')
line_fft, = ax1.semilogx(x_fft, np.random.rand(FRAMES), 'b')

ax.set_ylim(-32500, 32500)
ax.set_xlim(0, FRAMES)

Fmin = 1
Fmax = 5000
ax1.set_xlim(Fmin, Fmax)

fig.show()

F = (Fs / FRAMES) * np.arange(0, FRAMES // 2)  # Creamos el vector de frecuencia

while True:

    data = stream.read(FRAMES)  # Leemos paquetes de longitud FRAMES
    dataInt = struct.unpack(str(FRAMES) + 'h', data)  # Convertimos los datos de bytes

    line.set_ydata(dataInt)  # Asignamos los datos a la curva de la variación temporal

    M_gk = abs(np.fft.fft(dataInt) / FRAMES)  # Calculamos la FFT y su magnitud

    ax1.set_ylim(0, np.max(M_gk) + 10)
    line_fft.set_ydata(M_gk)  # Asignamos la magnitud de la FFT a la curva del espectro

    M_gk = M_gk[0:FRAMES // 2]  # Tomamos la mitad del espectro
    Posm = np.where(M_gk == np.max(M_gk))  # Buscamos el pico máximo
    F_fund = F[Posm]  # Encontramos la frecuencia fundamental

    # Imprimir la Transformada de Fourier
    print("Valores de la Transformada de Fourier (FFT):")
    print(M_gk)  # Imprime los valores de la FFT calculada
    print(f"Frecuencia Dominante: {int(F_fund)} Hz")  # Imprimimos la frecuencia dominante

    fig.canvas.draw()
    fig.canvas.flush_events()
