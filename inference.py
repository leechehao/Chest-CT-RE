import argparse

import mlflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking_uri", type=str, required=True, help="指定 MLflow 追蹤伺服器的 URI。")
    parser.add_argument("--run_id", type=str, required=True, help="MLflow 實驗運行的唯一標識符。")
    parser.add_argument("--text", type=str, required=True, help="待分析的文本。文本中須先加入特殊符號（$）在要關係萃取的實體兩側。")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    model = mlflow.pytorch.load_model(f"runs:/{args.run_id}/model")
    ents = model.hf_pipeline(args.text)
    print(ents)


if __name__ == "__main__":
    main()
