# README

Classificação multi-label de texto a partir do BERT em português.

Casos de uso:

- Treino (fine tuning) de rede neural para classificar novos textos;
- Método para classificar textos com a rede neural que já passou por fine tuning.

Simples uso de fine tuning do BERT português.

## Status

- v1: completa. Para o caso em `data/default.csv`, o modelo prevê com boa acurácia.

## Instalação

Rode o comando para instalar as dependências, se possível num ambiente virtual.

```bash
# Crie um ambiente virtual para instalar as dependências
python3 -m venv .venv
source .venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

## Caso de uso: previsão

```bash
# retorna classe para um unico item
python app.py \
    --frase "Os bancos pagariam o governo caso os processos de indústria fossem realmente atendidos" \
    --confianca 0.3

# retorna classe para varios itens, documento com um item por linha
python app.py --arquivo text_example.txt --confianca 0.3 --tipo arquivo
```

## Caso de uso: retreino

Caso queira treinar com um novo caso de uso, basta rodar:

```bash
python train.py --train_file "train.csv" --model "bert_fine_tuned.pt" --output_model "your_model.pt"
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
