# Cruzamento de cães usando VAE

Propósito:

- Dadas duas imagens de cães de raças diferentes, gera imagens de cães que seriam o resultado da mistura das imagens entre eles usando VAEs, dando um vislumbre da mistura entre os dois cães.

## Status

O pipeline funciona, mas as imagens ainda estão muito embaçadas, precisando de treino/tuning mais cuidadoso, se possivel com mais infra (treinei em CPU pequena local).

## Como rodar

```bash
# mistura as imagens de dois cachorros
python app.py --dog1 data/samples/beagle.jpg --dog2 data/samples/pomeranian.jpg --show True
```

## Dataset

O dataset de treino utilizado foi o stanford dogs, que possui 21k imagens de cães com raças (não usamos o de 10k imagens).

## Treino

VAE feito com Keras, copiando em grande parte o protocolo [deste link](https://github.com/Data-Science-kosta/Variational-Autoencoder-for-Face-Generation/blob/master/VAE_celebFaces.ipynb), alterando vários pedaços deprecados e alguns da rede neural original que não faziam sentido.

As instruções estão nos notebooks:

1. `data_acquisition.ipynb`: preparo do dataset
2. `train.ipynb`: treino do VAE
