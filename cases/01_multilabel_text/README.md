# README

Dada uma entrada de texto com vários rótulos:

1. Treina rede neural para classificar novos casos;
2. Endpoint de previsão para classificar após treinado.

Simples uso de fine tuning do BERT português.

## Instalação

Rode:

```bash
pip install -r requirements.txt
```

## Caso de uso: previsão

```bash
# retorna classe para um unico item
python app01_text --input_text "texto a ser classificado" --model "bert_fine_tuned.pt"

# retorna classe para varios itens, documento com um item por linha
python app01_text --input_file "arquivo.txt" --model "bert_fine_tuned.pt"
```

## Caso de uso: retreino

Caso queira treinar com um novo caso de uso, basta rodar:

```bash
python app01_text --train_file "train.csv" --model "bert_fine_tuned.pt" --output_model "your_model.pt"
```

O formato do arquivo de entrada `train.csv` é um CSV, com as colunas:

1. `sentence` com a entrada de texto
2. `category` com uma ou mais categorias separadas por `,` e cercadas por aspas, exemplo:

```csv
sentence,category
O Brasil é um país com vícios e qualidades,"corrupção,desigualdade,belezas naturais"
O trabalho enobrece a pessoa,"trabalho,qualidade de vida"
Deus ajuda quem cedo madruga,"religião,qualidade de vida"
```
