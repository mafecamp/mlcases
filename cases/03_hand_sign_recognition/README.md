# README

Reconhecimento de sinais de mão (dos Estados Unidos) usando a biblioteca OpenCV/Mediapipe.

## Propósito

O propósito é reconhecer sinais de mão em tempo real, usando a webcam, e também em vídeos estáticos.

## Status

O modelo é capaz de reconhecer corretamente ~8 sinais de mão.

## Treino

Para treinar o modelo, usamos o Hand MNIST. Basicamente, ele traz dataframes com os pixels (28x28) de cada imagem com um sinal de mão, parte gerada de imagens reais e algumas outras geradas por augmentation.

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

## Ideias de melhoria

- Não foi feito qualquer tuning por limitação de máquina.
- Possivelmente, usando um upscale das imagens com ESRGAN, poderíamos melhorar a qualidade das imagens reconhecidas.