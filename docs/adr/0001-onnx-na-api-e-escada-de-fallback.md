# ADR 0001: ONNX Runtime na API e escada de fallback

- Status: aceito
- Data: 2026-11-10
- Origem: [PLANO](../../PLANO.md) §3, §5.4 e §14

## Contexto

O notebook treina com TensorFlow/Keras 3. Levar o TensorFlow para a imagem de inferência aumentaria a superfície de
ataque, o tempo de build e o cold start. Além disso, carregar `.keras` ou `.h5` pode executar código: o PLANO lista
CVE-2025-1550, CVE-2025-8747, CVE-2025-9905 e GHSA-c9rc-mg46-23w3 (inclusive bypass de `safe_mode`).

Exportar para ONNX tem riscos conhecidos (PLANO §3): `Model.export(format="onnx")` exige política de dtype float32,
e no backend TensorFlow o Keras usa o `tf2onnx`, cujo README procura novo mantenedor e testa só TensorFlow 2.13 a
2.15 e opsets 14 a 18. O projeto usa TensorFlow 2.21.0 e opset 17.

## Decisão

1. A API serve os modelos com `onnxruntime` (CPU), sem TensorFlow na imagem.
2. A API nunca desserializa `.keras`, `.h5`, pickle e similares. O git ignora `*.keras` e `*.h5`, e
   `tools/check_artifacts.py` rejeita esses formatos dentro de `artifacts/`.
3. O arquivo é lido para a memória, tem o SHA-256 conferido contra o manifesto e só então vai ao
   `InferenceSession` como bytes. Modelos com dados externos (`external_data`) são recusados por uma varredura do
   protobuf feita no próprio adaptador, sem importar o pacote `onnx` (CVE-2026-34447). Pesos `.npz` são lidos com
   `allow_pickle=False`.
4. A **escada de fallback** é declarada no manifesto, em `runtimes`, por ordem de preferência:
   1. `onnx`: clone de inferência em float32, opset 17.
   2. `numpy_mlp`: só para MLPs (`normalize`, `dense` com `relu`/`linear`/`softmax`, `denormalize`); pesos em `.npz`
      e camadas descritas no manifesto.
   3. `litert`: para a CNN; extra opcional `trinet-api[litert]`, importado sob demanda; o interpretador não é
      thread-safe, então cada chamada toma um lock.
5. Para cada degrau, o carregador confere o SHA-256, constrói o runtime e roda um warm-up com um lote de zeros
   (forma correta, valores finitos, vetor de probabilidades válido para classificadores). O primeiro que passa
   serve o modelo. Falhas ficam no log (`error` para integridade, `warning` para o resto).
6. Se nenhum degrau passar, o modelo vira `UnavailableModel` com um `reason` (`integrity_check_failed`,
   `invalid_manifest` ou `no_usable_runtime`). Só ele responde `404 model_not_available`; os outros continuam
   no ar e `/health/ready` passa a `degraded`.
7. Só a maior versão semver de cada modelo é considerada. Não há fallback silencioso para uma versão antiga.

## Consequências

- Imagem menor e sem TensorFlow; nenhum caminho de código da API carrega formatos que executam código.
- Os MLPs têm dois caminhos numéricos (ONNX e NumPy) que precisam concordar. A suíte de contrato do `ModelRuntime`
  (`api/tests/contract/`) roda os mesmos testes em todos os adaptadores, e o golden compara a saída com a do notebook.
- Que o `tf2onnx` exporte as redes finais está fora da matriz que ele declara testar. O spike com as redes reais
  decide; se a CNN falhar, resta o LiteRT, e se ele também falhar a CNN fica indisponível sem derrubar os tabulares.
- O adaptador LiteRT e os testes (com um `.tflite` mínimo gerado sem TensorFlow) existem, mas rodar o `.tflite`
  exportado pelo Keras com a CNN final, que habilita ops Flex, **não foi verificado**.
- O carregador não valida a estrutura do ONNX além do que o `onnxruntime` aceita e da varredura de dados externos.
- Rollback de um modelo exige remover ou reverter o diretório da versão ruim, não apenas implantar uma imagem
  antiga que ainda a contenha (veja o [runbook](../runbook.md#trocar-a-versão-de-um-modelo)).
