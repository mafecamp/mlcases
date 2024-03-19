from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from loguru import logger
from config import settings
import numpy as np
import torch
import argparse
import gdown
import os


def parse_labels(labels: list):
    """
    Parses the labels from the settings file.
    """
    id2label = {}

    for idx, label in enumerate(labels):
        id2label[idx] = label

    label2id = {v: k for k, v in id2label.items()}

    return id2label, label2id


parser = argparse.ArgumentParser(
    prog="multilabel text model",
    description="App de classificação de texto multi rótulos.",
)

parser.add_argument(
    "--confianca",
    type=float,
    help="A confiança mínima para a classificação, entre 0 e 1 (decimal).",
    required=True,
)
parser.add_argument("--frase", type=str, help="A frase que deve ser classificada")
parser.add_argument(
    "--arquivo",
    type=str,
    help="Arquivo com frases a serem classificadas (uma frase por linha)",
)
parser.add_argument(
    "--tipo", type=str, help="Tipo de entrada (arquivo ou frase)", default="frase"
)

parsed_args = parser.parse_args()
input_type = parsed_args.tipo
valid_types = ["frase", "arquivo"]

logger.info(f"Tipo de entrada: {input_type}")

if input_type not in valid_types:
    raise ValueError(f"Input type {input_type} not supported")

if input_type == "frase" and parsed_args.frase is None:
    raise ValueError("Frase não pode ser nula")

if input_type == "arquivo" and parsed_args.arquivo is None:
    raise ValueError("Arquivo não pode ser nulo")

threshold = parsed_args.confianca
labels = settings.prediction.labels
id2label, label2id = parse_labels(labels)

if not os.path.exists(settings.trained_model_path):
    # Baixa o modelo a primeira vez caso não exista
    logger.info("Baixando o modelo pré-treinado padrão pela primeira vez...")
    file_id = "1MfyowP9czztTx58dRn6xSFTNbwR2CmnL"
    gdown.download(id=file_id, output="model.zip", quiet=False)
    logger.info("Descompactando o modelo...")
    os.system("unzip model.zip -d ./models/")

new_model = AutoModelForSequenceClassification.from_pretrained(
    settings.trained_model_path,
    problem_type="multi_label_classification",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=settings.base_model_name
)


def predict_phrase(phrase):
    """
    Prevê uma frase dado um modelo e um tokenizador.
    """
    inputs = tokenizer(
        phrase, return_tensors="pt", padding=True, truncation=True, max_length=128
    )

    outputs = new_model(**inputs)

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(outputs.logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= threshold)] = 1
    predicted_labels = [
        id2label[idx] for idx, label in enumerate(predictions) if label == 1.0
    ]

    return predicted_labels


# create string from list itens
if input_type == "frase":
    predicted_labels = predict_phrase(phrase)
    labels_str = ", ".join(predicted_labels)
    logger.info(f"Rótulos: {labels_str}")
elif input_type == "arquivo":
    line_no = 1
    for line in open(parsed_args.arquivo, "r"):
        predicted_labels = predict_phrase(line)
        logger.info(f"Linha {line_no}: {predicted_labels}")
        line_no += 1

        # save as predictions
        with open("predictions.txt", "a") as f:
            predicted_labels_commasep = ", ".join(predicted_labels)
            # add quotes

            f.write(f'{line.strip()},"{predicted_labels_commasep}"\n')

    logger.info("Predições salvas em predictions.txt")
else:
    raise ValueError(
        f"Input type {input_type} not supported. Use one of: 'phrase' or 'file'"
    )
