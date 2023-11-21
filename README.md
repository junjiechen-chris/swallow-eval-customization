# 概要

英語の事前学習済み大規模言語モデルから継続学習されたモデルの評価。

評価軸：
* 日本語能力が改善されるか？
* 英語能力が維持されるか？

# 準備：環境構築
各フレームワークに対し、別々の仮想環境を用意することを推奨します

```
python -m venv .venv_llm_jp_eval
python -m venv .venv_harness_jp
python -m venv .venv_harness_en
```
`jalm-evaluation-private/`にて
```
source .venv_llm_jp_eval/bin/activate
cd llm-jp-eval
pip install .
pip install protobuf
pip install sentencepiece
```
`jalm-evaluation-private/`にて
```
source .venv_harness_jp/bin/activate
cd lm-evaluation-harness-jp
pip install -e ".[ja]"
pip install sacrebleu
pip install sentencepiece
pip install protobuf
```
`jalm-evaluation-private/`にて
```
source .venv_harness_en/bin/activate
cd lm-evaluation-harness-en
pip install -e .
pip install sentencepiece
pip install protobuf
```

# 日本語の評価
* `llm-jp-eval` および `JP LM Evaluation Harness` の一部を採用
    * 多肢選択・自然言語推論・質問応答・文書読解・数学
    * 生成タスク: XLSum

## llm-jp-eval データセットの前処理
* まず[llm-jp-evalのREADME.md](https://github.com/llm-jp/llm-jp-eval/tree/main)に従って、データセットをダウンロードする  
* つぎに (a)公式設定 または (b)NLIタスクを日本語化 のいずれかの設定を選んで、前処理を実行する。  
  両者の違いは、NLIタスクのクラスラベルを(a)英語にするか または (b)日本語化するか である。  
  日本語に特化したLLM、特に指示チューニングしていないLLMの性能を評価する場合は (b)のほうが適切ではないかという説がある。  
  参考：[Stability AI 日本語大規模言語モデル「Japanese Stable LM Beta」シリーズをリリースしました](https://ja.stability.ai/blog/japanese-stable-lm-beta)

```
# (a)公式設定 の場合
前提と仮説の関係をentailment、contradiction、neutralの中から回答してください。

# (b)NLIタスクを日本語化 の場合
前提と仮説の関係を含意、矛盾、中立の中から回答してください。
```

```bash
cd llm-jp-eval

# (a)公式設定 の場合
python scripts/preprocess_dataset.py  \
--dataset-name all  \
--output-dir ./datasets

# (b)NLIタスクを日本語化 の場合
python scripts/preprocess_dataset.py  \
--dataset-name all  \
--output-dir ./datasets_nli_localize \
--localize_nli_verbalizer
```

## llm-jp-eval 評価の実行

`jalm-evaluation-private/`にて

llm-jp-evalのタスクで評価
```
bash scripts/evaluate_ja_llmjp.sh \
$MODEL_PATH \
$TOKENIZER_PATH \
$NUM_FEWSHOT \
$NUM_TESTCASE
```
全テストケースで評価する場合は、NUM_TESTCASEを`-1`にしてください。

## xlsum（自動要約）のタスクで評価

```
bash scripts/evaluate_ja_xlsum.sh \
$MODEL_PATH \
$NUM_FEWSHOT \
$NUM_TESTCASE
```
全テストケースで評価する場合は、`evaluate_ja_xlsum.sh`内の`--limit`を消してください。


## mgsm（数学）のタスクで評価

```
bash scripts/evaluate_ja_mgsm.sh \
$MODEL_PATH \
$NUM_FEWSHOT \
$NUM_TESTCASE
```
全テストケースで評価する場合は、`evaluate_ja_mgsm.sh`内の`--limit`を消してください。

結果は
`results/${MODEL_PATH}/ja/${task_name}_${NUM_FEWSHOT}shot_${NUM_TESTCASE}cases/`
に保存される。

# 英語の評価
* `llm-evaluation-harness` を採用
    * 常識推論: HellaSwag, WinoGrande, OpenBookQA
    * 世界知識: TriviaQA
    * 文書読解: SQuAD
    * 数学: GSM8K

`jalm-evaluation-private/`にて
```
bash scripts/evaluate_english.sh \
$MODEL_PATH \
$NUM_FEWSHOT \
$NUM_TESTCASE
```
全テストケースで評価する場合は、`evaluate_english.sh`内の`--limit`を消してください。

# ABCI上
* `rt_AG.small=1` と `rt_AF=1` で全タスク全テストケースで評価するスクリプトは `scripts/abci/rt_{AGsmall,AF}/qsub_all.sh` です。
`jalm-evaluation-private/`にて
```
bash scripts/abci/rt_{AGsmall,AF}/qsub_all.sh $MODEL_NAME_OR_PATH
```
