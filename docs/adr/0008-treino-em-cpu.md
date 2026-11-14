# ADR 0008: Treino em CPU

- Status: aceito (altera a v1.0 do PLANO)
- Data: 2026-11-10
- Origem: [PLANO](../../PLANO.md), cabeçalho v1.1, §4.1 e §12

## Contexto

A v1.0 do plano assumia treino na GPU (RTX 3050 Mobile, 4 GB) com `tensorflow[and-cuda]`. Na prática o
`nvidia-smi` falhava: o driver estava instalado, mas o módulo do kernel não carregava. Resolver isso era uma ação
do usuário no sistema (Secure Boot/MOK ou reinício) e estava no caminho crítico, com Colab/Kaggle como plano B.

Em 10/11/2026 o usuário decidiu treinar na **CPU**: um i7-11370H (4 núcleos, 8 threads) com cerca de 8 GB de RAM
livres.

## Decisão

- O notebook usa `tensorflow==2.21.0` **sem** `[and-cuda]`, com `keras==3.15.1`, em Python 3.13 (o TensorFlow 2.21.0
  tem wheels cp310 a cp313, não 3.14). Está fixado em `notebook/pyproject.toml`.
- Os itens de driver NVIDIA, TF32 e Colab saem do caminho crítico.
- A CNN é dimensionada para a CPU. Na v1.2 ela recebe gráficos de **64x64x3** (cerca de 60 mil imagens de treino por
  amostragem com passo, no máximo ~1 GB de RAM em uint8) com meta de até 60 s por época, e o notebook roda a ablação
  de 4 configurações em vez de uma busca extensa. Os MLPs treinam sem dificuldade em CPU.
- A precisão continua float32, sem `mixed_float16`: o export ONNX exige a política float32. Os golden outputs são
  gerados na CPU, a mesma classe de hardware do `onnxruntime` da API.
- O hardware usado fica registrado em `manifest.training.hardware`, junto com as versões de Python, TensorFlow e
  Keras e o `seed`.
- `enable_op_determinism` não é usado na CNN: serializa o `tf.data` e não garante reprodução entre hardwares.
  Reprodutibilidade vem de `keras.utils.set_random_seed`, de média e desvio sobre 3 a 5 seeds nos MLPs e do registro
  das versões.

## Consequências

- O treino é mais lento que na GPU, e a CNN tende a ser menor ou a usar entrada reduzida, o que pode custar
  acurácia. O número final é o que está nos model cards; o acaso nas 3 classes de regime é 33,3%.
- Remove um bloqueio de ambiente (driver) do cronograma e torna o resultado reproduzível em qualquer máquina com
  Python 3.13 e memória suficiente.
- O CI continua sem treinar nem baixar dados (só lint estático do notebook). O "Restart & Run All" é local.
- Trechos do PLANO ainda descrevem a GPU (§3, §4.1, §12 e §15). O cabeçalho da v1.1 prevalece sobre eles.
