# mlcases

Repositório para resoluções de problemas pontuais com machine learning e redes neurais.

Cada diretório possui as próprias instruções de instalação e como rodar nos casos de uso.

## Casos completos

- 01_multilabel_text: Fine tuning do bert para textos com vários rótulos (labels).
- 02_dog_breeding_vae: Variational Autoencoder para geração de imagens de cachorros.
- 03_hand_sign_recognition: Reconhecimento de letras (lingua de sinais estadunidense) usando MediaPipe.

## Instalando

Em todos os casos, o padrão de uso é:

- Instale as dependencias, possivelmente num ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Rode os apps conforme README.md

