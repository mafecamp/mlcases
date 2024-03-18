# README

Reconhecimento de sinais de mão (dos Estados Unidos) usando a biblioteca OpenCV/Mediapipe.

## Treino

Para treinar o modelo, usamos o Hand MNIST. Basicamente, ele possui dataframes com os pixels (28x28) de cada sinal de mão

## Uso

### Vídeo estático

```bash
python static.py large.task
```

### Webcam - tempo real

Ao rodar o script, ele irá abrir a webcam e reconhecer os sinais de mão. Para sair, basta pressionar a tecla `q`.

```bash
python app.py large.task
```
